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
  retain <run_id> [--dry-run]       make the memories of one finished run
  backfill [--dry-run] [--report F]  retain every recorded run not yet retained;
                                     --dry-run writes nothing, only the report
  review [--agent A] [--n 10]        best and worst memories per agent (plan 10 gate)
  approve <memory_id> <reason>       candidate -> active
  quarantine <memory_id> <reason>    stop a memory from being recalled (never deleted)
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
    retain_cmd = sub.add_parser("retain")
    retain_cmd.add_argument("run_id")
    retain_cmd.add_argument("--dry-run", action="store_true")
    backfill_cmd = sub.add_parser("backfill")
    backfill_cmd.add_argument("--dry-run", action="store_true")
    backfill_cmd.add_argument("--report", help="write the report to this file (default: reports/memory_backfill_<time>.md)")
    review = sub.add_parser("review")
    review.add_argument("--agent")
    review.add_argument("--n", type=int, default=10)
    for name in ("approve", "quarantine"):
        cmd = sub.add_parser(name)
        cmd.add_argument("memory_id")
        cmd.add_argument("reason")
    args = parser.parse_args(argv)

    if args.command in ("retain", "backfill"):
        from datetime import datetime
        from pathlib import Path
        from src.memory import retain

        reports = ([retain.retain_run(args.run_id, dry_run=args.dry_run)] if args.command == "retain"
                   else retain.backfill(dry_run=args.dry_run))
        text = retain.format_report(reports, args.dry_run)
        path = Path(getattr(args, "report", None) or
                    memory.REPO_ROOT / "reports" / f"memory_backfill_{datetime.now():%Y%m%d_%H%M%S}.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        for r in reports:
            print(f"  {r['run_id']}: " + (r["skipped"] or f"{len(r['written'])} memories, "
                                          f"{len(r['duplicates'])} duplicate(s) ({r['lineage']})"))
        print(("Dry run: nothing written. " if args.dry_run else "") + f"Report: {path}")
        return 0
    if args.command == "review":
        found = memory.find_memories(agent=args.agent, states=None, limit=100_000)
        for agent in sorted({m["agent"] for m in found}):
            mine = sorted((m for m in found if m["agent"] == agent), key=lambda m: -(m["quality"] or 0))
            print(f"\n== {agent} ({len(mine)} memories)")
            shown = mine if len(mine) <= 2 * args.n else mine[:args.n] + [None] + mine[-args.n:]
            for m in shown:
                if m is None:
                    print("   ...")
                    continue
                print(f"   {m['quality'] or 0:.2f}  {m['state']:<11} {m['kind']}/{m['polarity']:<8} "
                      f"{m['lineage']:<10} {m['title'][:80]}  [{m['id'][:8]}]")
        return 0
    if args.command in ("approve", "quarantine"):
        state = "active" if args.command == "approve" else "quarantined"
        ok = memory.set_state(args.memory_id, state, args.reason)
        print(f"{args.memory_id}: {'now ' + state if ok else 'not changed (see the warning above)'}")
        return 0 if ok else 1

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
