"""One-command launcher for API + SQLite ingest engine."""

import argparse
import os

import uvicorn


DEFAULT_CAPTURE_SOURCE = r"\Device\NPF_{94B4B764-8E09-485A-9EF1-10C26641DF79}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start NetGuard full backend chain")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="ingest existing realtime_predictions.jsonl from beginning (not tail-only)",
    )
    parser.set_defaults(auto_capture=True)
    parser.add_argument(
        "--auto-capture",
        dest="auto_capture",
        action="store_true",
        help="auto-start live capture/inference subprocess (default: on)",
    )
    parser.add_argument(
        "--no-auto-capture",
        dest="auto_capture",
        action="store_false",
        help="disable auto-start capture/inference subprocess",
    )
    parser.add_argument(
        "--capture-source",
        default=DEFAULT_CAPTURE_SOURCE,
        help="network interface for live capture (e.g. \\Device\\NPF_{...})",
    )
    parser.add_argument(
        "--capture-device",
        default=None,
        help="capture inference device, cpu or cuda",
    )
    parser.add_argument(
        "--no-engine",
        action="store_true",
        help="only run API, do not autostart backend_engine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Bridge CLI flags to api_server startup behavior.
    os.environ["NETGUARD_AUTOSTART_ENGINE"] = "0" if args.no_engine else "1"
    os.environ["NETGUARD_ENGINE_FROM_START"] = "1" if args.from_start else "0"
    os.environ["NETGUARD_ENGINE_AUTOCAPTURE"] = "1" if args.auto_capture else "0"
    if args.capture_source:
        os.environ["NETGUARD_CAPTURE_SOURCE"] = args.capture_source
    if args.capture_device:
        os.environ["NETGUARD_CAPTURE_DEVICE"] = args.capture_device

    print(
        "[NetGuard Launcher] "
        f"engine={'off' if args.no_engine else 'on'}, "
        f"from_start={'on' if args.from_start else 'off'}, "
        f"auto_capture={'on' if args.auto_capture else 'off'}, "
        f"capture_source={args.capture_source}"
    )

    uvicorn.run(
        "services.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
