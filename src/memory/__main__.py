"""
python -m src.memory <command>

  status [--check]   backend, outbox, last sync (--check: reach the hosted store)
  sync               send the outbox and run metrics to the hosted store
  migrate            apply the hosted store's migrations; needs the admin
                     connection string in MEMORY_ADMIN_DATABASE_URL (set it
                     only in the shell for this command, never in .env)
  check-role         connect with MEMORY_DATABASE_URL: reading must work and
                     DELETE must be refused (nothing is changed)
  list [--agent A] [--kind K] [--limit N]
  show <memory_id>
"""
import argparse
import json
import os
import sys

from src import memory


def _main(argv: list) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.memory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").add_argument("--check", action="store_true")
    sub.add_parser("sync")
    sub.add_parser("migrate")
    sub.add_parser("check-role")
    listing = sub.add_parser("list")
    listing.add_argument("--agent")
    listing.add_argument("--kind")
    listing.add_argument("--limit", type=int, default=20)
    sub.add_parser("show").add_argument("memory_id")
    args = parser.parse_args(argv)

    if args.command == "status":
        print(json.dumps(memory.status(check_remote=args.check), indent=2))
        return 0
    if args.command == "sync":
        report = memory.sync()
        print(json.dumps(report, indent=2))
        return 1 if report.get("error") else 0
    if args.command in ("migrate", "check-role"):
        from src.config import MEMORY_DATABASE_URL
        from src.memory.backends.postgres import PostgresStore

        if args.command == "migrate":
            url = os.getenv("MEMORY_ADMIN_DATABASE_URL", "")
            if not url:
                print("Set MEMORY_ADMIN_DATABASE_URL (the admin connection string) in this shell only.")
                return 1
            applied = PostgresStore(url).migrate()
            print("Applied: " + (", ".join(applied) if applied else "nothing (the store is up to date)"))
            return 0
        result = PostgresStore(MEMORY_DATABASE_URL).check_role()
        print(json.dumps(result, indent=2))
        ok = result["select_ok"] and result["delete_refused"]
        print("OK: the role can read and cannot delete." if ok else "PROBLEM: the role must read and must not delete.")
        return 0 if ok else 1
    if args.command == "list":
        for m in memory.list_memories(args.agent, args.kind, args.limit):
            print(f"{m['id']}  {m['state']:<11} {m['kind']:<9} {m['agent']:<20} {m['title'][:70]}")
        return 0
    found = memory.get_memory(args.memory_id)
    if not found:
        print(f"No memory {args.memory_id}")
        return 1
    print(json.dumps(found, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
