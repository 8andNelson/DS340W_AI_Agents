"""python -m src.dashboard [--port 8050] -- start the local dashboard."""
import argparse

from dotenv import load_dotenv

load_dotenv()

from src.dashboard.freellmapi import FreeLLMAPIManager
from src.dashboard.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dashboard for the AI research pipeline.")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-freellmapi", action="store_true",
                        help="do not start the FreeLLMAPI Docker router on launch")
    args = parser.parse_args()

    router = FreeLLMAPIManager()
    if router.status()["enabled"] and not args.no_freellmapi:
        # LLM_PROVIDER=freellmapi: start Docker Desktop and the router in the
        # background, so the dashboard is usable while they come up.
        router.start_async()

    print(f"Dashboard running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    # 127.0.0.1 only: the dashboard can start pipeline runs and serve local
    # files, so it must not be reachable from other machines.
    create_app(router=router).run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
