import argparse
import os

from realtime.config import RealtimeConfig
from realtime.single_pipeline import run_single_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="单机实时流量推理系统（纯单线程模式）")
    parser.add_argument(
        "--mode",
        choices=["pcap", "live"],
        default="pcap",
        help="pcap=离线回放；live=实时网卡抓包",
    )
    parser.add_argument(
        "--source", required=True, help="pcap模式填文件/目录；live模式填网卡名"
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join("pytorch_model", "convnet.pth"),
        help="模型权重路径",
    )
    parser.add_argument(
        "--label-map",
        default=os.path.join("processed_full", "mnist", "label_map.json"),
        help="label_map.json路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("pytorch_model", "realtime"),
        help="推理结果输出目录",
    )
    parser.add_argument(
        "--unknown-enable",
        action="store_true",
        help="启用未知流量检测（MSP + 可选PyOD）",
    )
    parser.add_argument(
        "--unknown-detector-path",
        default=os.path.join("pytorch_model", "unknown_detector_best.pkl"),
        help="unknown检测器路径（inference.py 导出的 pkl）",
    )
    parser.add_argument(
        "--unknown-threshold-conf",
        type=float,
        default=None,
        help="MSP阈值tau1，confidence < tau1 判定为未知",
    )
    parser.add_argument(
        "--unknown-threshold-anom",
        type=float,
        default=None,
        help="异常分数阈值tau2，anomaly_score > tau2 判定为未知",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-wait-ms", type=int, default=50)
    parser.add_argument("--flow-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-active-flows", type=int, default=50000)
    parser.add_argument(
        "--live-duration-seconds",
        type=float,
        default=0.0,
        help="live模式运行时长（秒），0表示持续运行直到Ctrl+C",
    )
    parser.add_argument("--device", default=None, help="cuda/cpu，默认自动")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型不存在: {args.model_path}")

    cfg = RealtimeConfig(
        flow_timeout_s=args.flow_timeout_s,
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
        max_active_flows=args.max_active_flows,
    )

    print("=" * 88)
    print(f"Realtime(single-thread) mode={args.mode} source={args.source}")
    print(f"model={args.model_path} output_dir={args.output_dir}")
    print(
        f"batch={cfg.batch_size} wait={cfg.batch_wait_ms}ms timeout={cfg.flow_timeout_s}s"
    )
    print("=" * 88)
    run_single_pipeline(
        mode=args.mode,
        source=args.source,
        cfg=cfg,
        model_path=args.model_path,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        device=args.device,
        live_duration_seconds=args.live_duration_seconds,
        unknown_enable=args.unknown_enable,
        unknown_detector_path=args.unknown_detector_path,
        unknown_threshold_conf=args.unknown_threshold_conf,
        unknown_threshold_anom=args.unknown_threshold_anom,
    )


if __name__ == "__main__":
    main()
