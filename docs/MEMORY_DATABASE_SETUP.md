# Shared memory database (PostgreSQL) setup

The agents keep a memory of past runs: scripts that worked, failures, and
lessons. By default this memory is a local SQLite file
(`database/memory_local.db`), and you do not need to set anything up.

To share one memory between team members, the pipeline can use a hosted
PostgreSQL database instead (`MEMORY_BACKEND=postgres`). Any PostgreSQL 14
or newer works: Neon, Supabase, or your own server. The pipeline uses the
plain PostgreSQL protocol (`psycopg`, already in `requirements.txt`), not a
provider SDK.

Choose your part:

- **A. Join an existing team database:** most team members. You get a
  connection string from the person who set it up.
- **B. Create a new team database:** one person, once.
- **C. Try it on your own computer with Docker:** to test, with no account.

---

## How access is controlled

The database has two roles:

| Role | Who uses it | What it can do |
|---|---|---|
| admin (the provider's owner role, e.g. `neondb_owner` or `postgres`) | one person, only to run `migrate` | everything |
| `memory_agent` | every pipeline run, on every team member's computer | read and add memories; update only counters and state columns; **no `DELETE` or `TRUNCATE`** |

So a bug or a bad model answer cannot erase the team's memory. The
`memory_agent` role is created by the migrations
(`src/memory/migrations/postgres/002_roles.sql`), without a password; the
admin sets the password by hand.

Rules:

- Put **only** the `memory_agent` connection string in `.env`
  (`MEMORY_DATABASE_URL`). Never put the admin string in `.env`.
- Set the admin string (`MEMORY_ADMIN_DATABASE_URL`) only in the terminal
  where you run `migrate`, and close that terminal after.
- Share the `memory_agent` string privately (for example a direct message
  or a password manager), never in the repository, an issue, or a group
  channel.

---

## A. Join an existing team database

1. Get the `memory_agent` connection string from the person who set up the
   database (see part B, step 5).
2. In your `.env`, set:

   ```ini
   MEMORY_BACKEND=postgres
   MEMORY_DATABASE_URL=<the memory_agent connection string>
   MEMORY_CONTRIBUTOR=<your short name, not an email>
   ```

3. Check the role. It must be able to read and must not be able to delete:

   ```powershell
   python -m src.memory check-role
   ```

   Expected result:

   ```text
   {
     "user": "memory_agent",
     "select_ok": true,
     "delete_refused": true
   }
   OK: the role can read and cannot delete.
   ```

4. Send your local memories and run metrics to the team database, and
   check the connection:

   ```powershell
   python -m src.memory sync
   python -m src.memory status --check
   ```

The pipeline also syncs by itself at the start and end of every run.

**When the database cannot be reached**, a run does not stop. Writes wait
in a local outbox (`database/memory_local.db`) and are sent at the next
sync. Reads use the local copy if it was synced in the last day.

---

## B. Create a new team database (one person, once)

### 1. Create the database

**Neon** (https://neon.tech): create a project. On the project dashboard,
select **Connect** and copy the connection string of the owner role (for
example `neondb_owner`). It looks like:

```text
postgresql://neondb_owner:<password>@ep-example-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
```

**Supabase** (https://supabase.com): create a project. Select **Connect**
and copy a connection string for the `postgres` role. The session pooler
string looks like:

```text
postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

The tables go in their own schema, `agent_memory`, not `public`. On
Supabase, `public` is reachable through its web API; `agent_memory` is not,
and the migrations remove the web API roles' access to it.

### 2. Apply the migrations with the admin string

Set the admin string **in this terminal only**, then run `migrate`.

PowerShell:

```powershell
$env:MEMORY_ADMIN_DATABASE_URL = "<the admin connection string>"
python -m src.memory migrate
Remove-Item Env:MEMORY_ADMIN_DATABASE_URL
```

macOS / Linux:

```bash
export MEMORY_ADMIN_DATABASE_URL="<the admin connection string>"
python -m src.memory migrate
unset MEMORY_ADMIN_DATABASE_URL
```

Expected result:

```text
Applied: 001_init.sql, 002_roles.sql, 003_events.sql
```

You can run `migrate` again at any time. It applies only new migrations
(`Applied: nothing (the store is up to date)` when there are none). Run it
again after you pull a version of this repository that adds a file in
`src/memory/migrations/postgres/`.

### 3. Give the `memory_agent` role a password

Make a long random password:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Then, in the provider's SQL editor (or `psql` with the admin string), run:

```sql
ALTER ROLE memory_agent WITH LOGIN PASSWORD '<the password>';
```

A hex password has no characters that need escaping in a URL. If you use
another password, URL-encode its special characters (`@`, `:`, `/`, `#`,
`%`, `?`) in the connection string.

### 4. Make the `memory_agent` connection string

Use the admin string, with the user name and password changed:

- **Neon:**
  `postgresql://memory_agent:<password>@<same host>/<same database>?sslmode=require`
- **Supabase pooler:** the user name includes the project reference:
  `postgresql://memory_agent.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`
- **Your own server:**
  `postgresql://memory_agent:<password>@<host>:<port>/<database>`

Pooled connections work: the pipeline turns off prepared statements
(`prepare_threshold=None`).

### 5. Check the role and share the string

Put the `memory_agent` string in your own `.env` and run part A, steps 2 to 4.
`check-role` must print `OK`. If it says the role **can** delete, the string
in `.env` is the admin string: replace it.

Then give the `memory_agent` string to each team member privately.

To remove a member's access, set a new password (step 3) and give the
new string to the remaining members.

---

## C. Try it on your own computer with Docker

This makes a temporary PostgreSQL 16 database in Docker, for testing. It
needs Docker Desktop to be running.

> **Do not run `python -m src.memory sync` against a test database.** Sync
> sends your local outbox and marks it as sent. Memories sent to a test
> database are not sent to the team database later. `migrate` and
> `check-role` do not change your local files.

PowerShell:

```powershell
docker run -d --rm --name memory-test -e POSTGRES_PASSWORD=admin-test-pw -p 55432:5432 postgres:16
docker exec memory-test pg_isready -U postgres     # wait until "accepting connections"

$env:MEMORY_ADMIN_DATABASE_URL = "postgresql://postgres:admin-test-pw@localhost:55432/postgres"
python -m src.memory migrate
Remove-Item Env:MEMORY_ADMIN_DATABASE_URL

docker exec memory-test psql -U postgres -c "ALTER ROLE memory_agent WITH LOGIN PASSWORD 'agent-test-pw-0123456789abcdef';"

$env:MEMORY_DATABASE_URL = "postgresql://memory_agent:agent-test-pw-0123456789abcdef@localhost:55432/postgres"
python -m src.memory check-role
Remove-Item Env:MEMORY_DATABASE_URL

docker stop memory-test                             # deletes the test database
```

A variable set in the terminal is used before the value in `.env`, so these
commands do not change your `.env`.

---

## Commands

| Command | What it does |
|---|---|
| `python -m src.memory status` | the backend, the outbox size and the last sync |
| `python -m src.memory status --check` | the same, and connects to the hosted database |
| `python -m src.memory sync` | sends the outbox and run metrics to the hosted database |
| `python -m src.memory migrate` | applies new migrations (needs `MEMORY_ADMIN_DATABASE_URL` in the terminal) |
| `python -m src.memory check-role` | checks that `MEMORY_DATABASE_URL` can read and cannot delete |
| `python -m src.memory list` | lists memories |

Other settings (`MEMORY_MODE`, `MEMORY_REUSE`, ...) are explained in
`.env.example`. `MEMORY_BACKEND=off` turns memory off.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Set MEMORY_ADMIN_DATABASE_URL ...` | set the admin string in the terminal before `migrate` (part B, step 2) |
| connection timeout or refused | check the host and port; hosted databases need `?sslmode=require` (Neon) or the pooler host (Supabase); check your network or VPN |
| `password authentication failed` | wrong password, or a special character that is not URL-encoded; on the Supabase pooler the user name must be `memory_agent.<project-ref>` |
| `role "memory_agent" does not exist` / `permission denied for schema agent_memory` | `migrate` was not run on this database (part B, step 2) |
| `role ... is not permitted to log in` | the password step was not done (part B, step 3) |
| `check-role` prints `PROBLEM` | `MEMORY_DATABASE_URL` is the admin string; use the `memory_agent` string |
| `... changed after it was applied to the hosted store` | a migration file that was already applied was edited; restore it with `git checkout` and add a new migration file instead |
| `MEMORY_BACKEND must be one of off, sqlite, postgres` | fix the spelling in `.env` |
