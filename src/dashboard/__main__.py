"""python -m src.dashboard [--port 8050] -- start the local dashboard."""
import argparse

from dotenv import load_dotenv

load_dotenv()

from src.dashboard.freellmapi import FreeLLMAPIManager
from src.dashboard.runner import PipelineRunner
from src.dashboard.server import create_app, run_in_progress


def stop_router_on_exit(router: FreeLLMAPIManager, runner: PipelineRunner) -> None:
    """`docker compose down` when the dashboard closes, unless a run still needs the router."""
    if not router.status()["enabled"]:
        return
    if run_in_progress(runner):
        print("[Dashboard] FreeLLMAPI: a pipeline run is still active, so the router stays running.")
        return
    router.stop_now()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dashboard for the AI research pipeline.")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-freellmapi", action="store_true",
                        help="do not start or stop the FreeLLMAPI Docker router")
    parser.add_argument("--keep-freellmapi", action="store_true",
                        help="do not stop the FreeLLMAPI Docker router when the dashboard closes")
    args = parser.parse_args()

    router, runner = FreeLLMAPIManager(), PipelineRunner()
    if router.status()["enabled"] and not args.no_freellmapi:
        # LLM_PROVIDER=freellmapi: start Docker Desktop and the router in the
        # background, so the dashboard is usable while they come up.
        router.start_async()

    print(f"Dashboard running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    # 127.0.0.1 only: the dashboard can start pipeline runs and serve local
    # files, so it must not be reachable from other machines.
    try:
        create_app(runner=runner, router=router).run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    finally:
        if not (args.no_freellmapi or args.keep_freellmapi):
            stop_router_on_exit(router, runner)


if __name__ == "__main__":
    main()
