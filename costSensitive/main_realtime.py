import argparse
import os

from realtime.config import RealtimeConfig
from realtime.single_pipeline import run_single_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Byte-session realtime inference (pcap mode)"
    )
    parser.add_argument("--mode", choices=["pcap"], default="pcap")
    parser.add_argument("--source", required=True, help="pcap file or directory")
    parser.add_argument(
        "--model-path",
        default=os.path.join("pytorch_model", "byte_session_classifier.pth"),
    )
    parser.add_argument(
        "--label-map",
        default=os.path.join("processed_full", "sessions", "label_map.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("pytorch_model", "realtime_sessions"),
    )
    parser.add_argument("--num-packets", type=int, default=12)
    parser.add_argument("--packet-len", type=int, default=256)
    parser.add_argument("--flow-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-active-flows", type=int, default=50000)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=int, default=50)
    parser.add_argument("--device", default=None, help="cuda/cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model not found: {args.model_path}")

    cfg = RealtimeConfig(
        num_packets=args.num_packets,
        packet_len=args.packet_len,
        flow_timeout_s=args.flow_timeout_s,
        max_active_flows=args.max_active_flows,
        infer_batch_size=args.infer_batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )

    run_single_pipeline(
        mode=args.mode,
        source=args.source,
        cfg=cfg,
        model_path=args.model_path,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
