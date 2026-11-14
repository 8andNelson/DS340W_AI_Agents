"""
Hosted Postgres memory store, shared by the team (Supabase, Neon or any
Postgres: only a standard connection string is used, through psycopg).

The pipeline connects with the restricted memory_agent role
(MEMORY_DATABASE_URL): SELECT and INSERT, UPDATE on counters and state
only, no DELETE. Migrations need the admin connection
(MEMORY_ADMIN_DATABASE_URL), given only when running
`python -m src.memory migrate`.

Prepared statements are turned off so a transaction pooler (Supabase's
port 6543) works. A connection or timeout failure is raised as
RemoteUnavailable; any other database error belongs to the op itself.
"""
from contextlib import contextmanager
from datetime import datetime

from src.memory import schema
from src.memory.backends.base import MigrationError, RemoteUnavailable, check_op, migration_files

SCHEMA_NAME = "agent_memory"
CONNECT_TIMEOUT_SECONDS = 5


def _psycopg():
    # Imported here so the default SQLite backend never needs psycopg.
    import psycopg
    return psycopg


def encode(table: str, row: dict) -> dict:
    from psycopg.types.json import Jsonb

    types = schema.TABLES[table]
    out = {}
    for column, value in row.items():
        kind = types[column]
        if value is None:
            out[column] = None
        elif kind == "json":
            out[column] = Jsonb(value)
        elif kind == "list":
            out[column] = [str(v) for v in value]
        elif kind == "ts":
            out[column] = datetime.fromisoformat(value) if isinstance(value, str) else value
        elif kind == "bool":
            out[column] = bool(value)
        else:
            out[column] = value
    return out


def decode(table: str, row: dict) -> dict:
    out = {}
    for column, value in row.items():
        if column not in schema.TABLES[table]:
            continue
        out[column] = value.isoformat(timespec="seconds") if isinstance(value, datetime) else value
    return out


class PostgresStore:
    dialect = "postgres"

    def __init__(self, url: str):
        self.url = url

    @contextmanager
    def session(self):
        """One connection for several calls (sync sends many ops)."""
        if not self.url:
            raise RemoteUnavailable("no connection string is set (MEMORY_DATABASE_URL)")
        psycopg = _psycopg()
        from psycopg.rows import dict_row
        try:
            # autocommit: each read stands alone, and every write is an
            # explicit conn.transaction() block (a real BEGIN/COMMIT).
            conn = psycopg.connect(self.url, connect_timeout=CONNECT_TIMEOUT_SECONDS, row_factory=dict_row,
                                   prepare_threshold=None, autocommit=True)
        except psycopg.OperationalError as e:
            raise RemoteUnavailable(f"cannot connect to the memory store: {e}") from e
        try:
            yield conn
        except psycopg.OperationalError as e:
            raise RemoteUnavailable(f"lost the connection to the memory store: {e}") from e
        finally:
            conn.close()

    # --- Schema ---

    def migrate(self) -> list[str]:
        """Apply the migrations not yet applied (needs the admin connection)."""
        applied = []
        with self.session() as conn:
            with conn.transaction():
                conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
                conn.execute(f"CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.schema_migrations (version integer PRIMARY KEY, "
                             "name text NOT NULL, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())")
                conn.execute(f"REVOKE ALL ON {SCHEMA_NAME}.schema_migrations FROM PUBLIC")
            done = {r["version"]: r["checksum"]
                    for r in conn.execute(f"SELECT version, checksum FROM {SCHEMA_NAME}.schema_migrations").fetchall()}
            for version, name, sql, checksum in migration_files("postgres"):
                if version in done:
                    if done[version] != checksum:
                        raise MigrationError(f"{name} changed after it was applied to the hosted store")
                    continue
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute(f"INSERT INTO {SCHEMA_NAME}.schema_migrations (version, name, checksum) "
                                 "VALUES (%s, %s, %s)", (version, name, checksum))
                applied.append(name)
            # The store's schema version is the code's data-schema version
            # (schema.SCHEMA_VERSION), not the number of migration files.
            current = self.schema_version(conn)
            if current is not None and current > schema.SCHEMA_VERSION:
                raise MigrationError(f"the hosted store has schema version {current}, newer than this code's "
                                     f"{schema.SCHEMA_VERSION}: pull the latest code")
            conn.execute(f"INSERT INTO {SCHEMA_NAME}.memory_meta (key, value) VALUES ('schema_version', %s) "
                         "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (str(schema.SCHEMA_VERSION),))
        return applied

    def schema_version(self, conn=None) -> int | None:
        if conn is None:
            with self.session() as conn:
                return self.schema_version(conn)
        row = conn.execute(f"SELECT value FROM {SCHEMA_NAME}.memory_meta WHERE key = 'schema_version'").fetchone()
        return int(row["value"]) if row else None

    # --- Writes ---

    def apply(self, op: dict, conn=None) -> bool:
        check_op(op)
        if conn is None:
            with self.session() as conn:
                return self.apply(op, conn)
        sql = _psycopg().sql
        with conn.transaction():
            fresh = conn.execute(f"INSERT INTO {SCHEMA_NAME}.applied_ops (op_id) VALUES (%s) "
                                 "ON CONFLICT (op_id) DO NOTHING RETURNING op_id", (op["op_id"],)).fetchone()
            if not fresh:
                return False
            table = sql.Identifier(SCHEMA_NAME, op["table"])
            if op["type"] == "insert":
                for row in op["rows"]:
                    encoded = encode(op["table"], row)
                    conn.execute(sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
                        table, sql.SQL(", ").join(map(sql.Identifier, encoded)),
                        sql.SQL(", ").join(sql.Placeholder() * len(encoded))), list(encoded.values()))
            elif op["type"] == "set_state":
                conn.execute(f"UPDATE {SCHEMA_NAME}.memories SET state = %s, "
                             "superseded_by = COALESCE(%s, superseded_by), updated_at = now() WHERE id = %s",
                             (op["state"], op.get("superseded_by"), op["key"]["id"]))
            elif op["type"] == "usage_outcome":
                changed = conn.execute(f"UPDATE {SCHEMA_NAME}.memory_usage SET helped = %s "
                                       "WHERE id = %s AND helped IS NULL", (op["helped"], op["key"]["id"])).rowcount
                if changed:
                    column = sql.Identifier("times_helped" if op["helped"] else "times_hurt")
                    conn.execute(sql.SQL("UPDATE {table} SET {col} = {col} + 1, updated_at = now() WHERE id = %s").format(
                        table=sql.Identifier(SCHEMA_NAME, "memories"), col=column), (op["memory_id"],))
            elif op["type"] == "set_quality":
                from psycopg.types.json import Jsonb
                conn.execute(f"UPDATE {SCHEMA_NAME}.memories SET quality = %s, quality_reasons = %s, "
                             "updated_at = now() WHERE id = %s",
                             (op["quality"], Jsonb(op["quality_reasons"]), op["key"]["id"]))
            else:
                sets = sql.SQL(", ").join(sql.SQL("{0} = {0} + %s").format(sql.Identifier(c)) for c in op["counters"])
                where = sql.SQL(" AND ").join(sql.SQL("{} = %s").format(sql.Identifier(k)) for k in op["key"])
                conn.execute(sql.SQL("UPDATE {} SET {}, updated_at = now() WHERE {}").format(table, sets, where),
                             [*op["counters"].values(), *op["key"].values()])
        return True

    # --- Reads ---

    def fetch_memories(self, agent: str = None, kind: str = None, states: tuple = ("active",),
                       ids: list = None, limit: int = 50, conn=None, content_shas: list = None) -> list[dict]:
        if conn is None:
            with self.session() as conn:
                return self.fetch_memories(agent, kind, states, ids, limit, conn, content_shas)
        if (ids is not None and not ids) or (content_shas is not None and not content_shas):
            return []
        columns = ", ".join(schema.TABLES["memories"])
        where, params = [], []
        if agent:
            where.append("agent = %s")
            params.append(agent)
        if kind:
            where.append("kind = %s")
            params.append(kind)
        if states:
            where.append("state = ANY(%s)")
            params.append(list(states))
        if ids is not None:
            where.append("id = ANY(%s)")
            params.append(list(ids))
        if content_shas is not None:
            where.append("content_sha = ANY(%s)")
            params.append(list(content_shas))
        query = f"SELECT {columns} FROM {SCHEMA_NAME}.memories" + (" WHERE " + " AND ".join(where) if where else "")
        rows = conn.execute(query + " ORDER BY created_at DESC LIMIT %s", [*params, limit]).fetchall()
        return [decode("memories", r) for r in rows]

    # --- Role check (`python -m src.memory check-role`) ---

    def check_role(self) -> dict:
        """Connect as the configured role: it must read, and it must be
        refused DELETE. Nothing is changed (the DELETE matches no row and is
        rolled back)."""
        psycopg = _psycopg()
        with self.session() as conn:
            user = conn.execute("SELECT current_user AS u, 1 AS one").fetchone()
            conn.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.memories").fetchone()
            try:
                with conn.transaction(force_rollback=True):
                    conn.execute(f"DELETE FROM {SCHEMA_NAME}.memories WHERE false")
                delete_refused = False
            except psycopg.errors.InsufficientPrivilege:
                delete_refused = True
        return {"user": user["u"], "select_ok": user["one"] == 1, "delete_refused": delete_refused}
