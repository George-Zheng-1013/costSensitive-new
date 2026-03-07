import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .config import RealtimeConfig
from .flow_assembler import FlowAssembler, FlowSample, PacketEvent
from .unknown_detector import CentroidUnknownDetector


def _setup_windows_dll_paths() -> None:
    if sys.platform != "win32":
        return

    npcaps = [r"C:\Windows\System32\Npcap"]
    for path in npcaps:
        if not os.path.isdir(path):
            continue
        try:
            os.add_dll_directory(path)
        except Exception:
            pass
        if path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


_setup_windows_dll_paths()


class SinglePipelineRunner:
    def __init__(
        self,
        cfg: RealtimeConfig,
        model_path: str,
        label_map_path: Optional[str],
        detector_json_path: Optional[str],
        output_csv: str,
        output_jsonl: str,
        device: Optional[str] = None,
        unknown_label: str = "unknown_proxy_ood",
    ):
        import torch

        from .model_def import ByteSessionClassifier

        self.cfg = cfg
        self.torch = torch
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.label_map = self._load_label_map(label_map_path)
        num_classes = max(self.label_map.keys(), default=11) + 1
        self.unknown_label = unknown_label

        self.model = ByteSessionClassifier(num_classes=num_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.detector = CentroidUnknownDetector()
        if detector_json_path and os.path.exists(detector_json_path):
            self.detector.load(detector_json_path)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        self.output_csv = output_csv
        self.output_jsonl = output_jsonl

        self.batch: List[Dict] = []
        self.last_flush = time.time()

    @staticmethod
    def _load_label_map(path: Optional[str]) -> Dict[int, str]:
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(v): k for k, v in raw.items()}

    def _flush(self, writer, jsonf):
        if len(self.batch) == 0:
            return

        bytes_batch = np.stack([item["session_bytes"] for item in self.batch], axis=0)
        mask_batch = np.stack([item["packet_mask"] for item in self.batch], axis=0)

        with self.torch.no_grad():
            bytes_tensor = self.torch.from_numpy(bytes_batch.astype(np.int64)).to(
                self.device
            )
            mask_tensor = self.torch.from_numpy(mask_batch.astype(np.bool_)).to(
                self.device
            )

            pred, conf = self.model.predict(bytes_tensor, mask_tensor)
            emb = self.model.extract_embedding(bytes_tensor, mask_tensor)

        now = time.time()
        for i, item in enumerate(self.batch):
            pred_id = int(pred[i].item())
            confidence = float(conf[i].item())
            embedding_norm = float(self.torch.norm(emb[i], p=2).item())

            decision = None
            if self.detector.enabled:
                emb_i = emb[i].detach().cpu().numpy()
                decision = self.detector.decide(emb_i, pred_id)

            is_unknown = bool(decision.is_unknown) if decision is not None else False
            final_pred = -1 if is_unknown else pred_id
            final_pred_name = (
                self.unknown_label
                if is_unknown
                else self.label_map.get(pred_id, str(pred_id))
            )
            centroid_distance = (
                float(decision.centroid_distance)
                if decision is not None
                else float("nan")
            )
            centroid_threshold = (
                float(decision.centroid_threshold)
                if decision is not None
                else float("nan")
            )
            anomaly_score = (
                float(decision.anomaly_score) if decision is not None else 0.0
            )

            if anomaly_score >= 1.5:
                alert_level = "high"
            elif anomaly_score >= 1.0:
                alert_level = "medium"
            else:
                alert_level = "low"

            record = {
                "sample_id": item["sample_id"],
                "flow_key": str(item["flow_key"]),
                "trigger": item["trigger"],
                "capture_ts": item["ts"],
                "infer_ts": now,
                "end2end_ms": (now - item["enqueue_ts"]) * 1000.0,
                "pred": pred_id,
                "pred_name": self.label_map.get(pred_id, str(pred_id)),
                "confidence": confidence,
                "embedding_norm": embedding_norm,
                "centroid_distance": centroid_distance,
                "centroid_threshold": centroid_threshold,
                "anomaly_score": anomaly_score,
                "is_unknown": int(is_unknown),
                "final_pred": final_pred,
                "final_pred_name": final_pred_name,
                "alert_level": alert_level,
            }

            writer.writerow(
                [
                    record["sample_id"],
                    record["flow_key"],
                    record["trigger"],
                    f"{record['capture_ts']:.6f}",
                    f"{record['infer_ts']:.6f}",
                    f"{record['end2end_ms']:.3f}",
                    record["pred"],
                    record["pred_name"],
                    f"{record['confidence']:.6f}",
                    f"{record['embedding_norm']:.6f}",
                    f"{record['centroid_distance']:.6f}",
                    f"{record['centroid_threshold']:.6f}",
                    f"{record['anomaly_score']:.6f}",
                    record["is_unknown"],
                    record["final_pred"],
                    record["final_pred_name"],
                    record["alert_level"],
                ]
            )
            jsonf.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.batch.clear()
        self.last_flush = time.time()

    def _accept(self, sample: FlowSample, writer, jsonf):
        self.batch.append(
            {
                "sample_id": sample.sample_id,
                "flow_key": sample.flow_key,
                "trigger": sample.trigger,
                "ts": sample.ts,
                "enqueue_ts": time.time(),
                "session_bytes": sample.session_bytes,
                "packet_mask": sample.packet_mask,
            }
        )

        now = time.time()
        if (
            len(self.batch) >= self.cfg.infer_batch_size
            or (now - self.last_flush) * 1000.0 >= self.cfg.batch_wait_ms
        ):
            self._flush(writer, jsonf)

    def run_live(self, source: str, live_duration_seconds: float = 0.0):
        try:
            from scapy.all import sniff
            from scapy.layers.inet import IP, TCP, UDP
        except Exception as exc:
            raise RuntimeError(
                "Scapy is required for live capture. Install it with: pip install scapy"
            ) from exc

        assembler = FlowAssembler(self.cfg)
        start_ts = time.time()

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f_csv, open(
            self.output_jsonl, "w", encoding="utf-8"
        ) as f_jsonl:
            writer = csv.writer(f_csv)
            writer.writerow(
                [
                    "sample_id",
                    "flow_key",
                    "trigger",
                    "capture_ts",
                    "infer_ts",
                    "end2end_ms",
                    "pred",
                    "pred_name",
                    "confidence",
                    "embedding_norm",
                    "centroid_distance",
                    "centroid_threshold",
                    "anomaly_score",
                    "is_unknown",
                    "final_pred",
                    "final_pred_name",
                    "alert_level",
                ]
            )

            last_sweep = time.time()

            def _canonical_flow_key(
                src_ip: str,
                src_port: int,
                dst_ip: str,
                dst_port: int,
                proto: int,
            ):
                left = (src_ip, src_port)
                right = (dst_ip, dst_port)
                if left <= right:
                    return src_ip, dst_ip, src_port, dst_port, proto
                return dst_ip, src_ip, dst_port, src_port, proto

            def _packet_to_event(packet) -> Optional[PacketEvent]:
                if not packet.haslayer(IP):
                    return None
                ip_layer = packet[IP]

                if packet.haslayer(TCP):
                    trans = packet[TCP]
                    payload = bytes(trans.payload)
                    flags = int(getattr(trans, "flags", 0))
                    fin = bool(flags & 0x01)
                    rst = bool(flags & 0x04)
                    if flags == 0x10 and len(payload) == 0:
                        return None
                    flow_key = _canonical_flow_key(
                        str(ip_layer.src),
                        int(getattr(trans, "sport", 0)),
                        str(ip_layer.dst),
                        int(getattr(trans, "dport", 0)),
                        6,
                    )
                elif packet.haslayer(UDP):
                    trans = packet[UDP]
                    payload = bytes(trans.payload)
                    fin = False
                    rst = False
                    flow_key = _canonical_flow_key(
                        str(ip_layer.src),
                        int(getattr(trans, "sport", 0)),
                        str(ip_layer.dst),
                        int(getattr(trans, "dport", 0)),
                        17,
                    )
                else:
                    return None

                return PacketEvent(
                    ts=time.time(),
                    flow_key=flow_key,
                    payload=payload,
                    fin=fin,
                    rst=rst,
                )

            def _callback(packet):
                evt = _packet_to_event(packet)
                if evt is None:
                    return
                for sample in assembler.feed(evt):
                    self._accept(sample, writer, f_jsonl)

            try:
                while True:
                    now = time.time()
                    if (
                        live_duration_seconds > 0
                        and (now - start_ts) >= live_duration_seconds
                    ):
                        break

                    sniff(
                        iface=source,
                        prn=_callback,
                        store=False,
                        filter="ip and (tcp or udp)",
                        timeout=1,
                    )

                    now = time.time()
                    if now - last_sweep >= self.cfg.sweep_interval_s:
                        for sample in assembler.sweep_timeouts(now):
                            self._accept(sample, writer, f_jsonl)
                        last_sweep = now
            except KeyboardInterrupt:
                pass

            for sample in assembler.sweep_timeouts(
                time.time() + self.cfg.flow_timeout_s + 1.0
            ):
                self._accept(sample, writer, f_jsonl)
            self._flush(writer, f_jsonl)

        print(f"[DONE] output_csv={self.output_csv}")


def run_single_pipeline(
    mode: str,
    source: str,
    cfg: RealtimeConfig,
    model_path: str,
    label_map_path: Optional[str],
    detector_json_path: Optional[str],
    output_dir: str,
    device: Optional[str] = None,
    live_duration_seconds: float = 0.0,
    unknown_label: str = "unknown_proxy_ood",
):
    if mode != "live":
        raise ValueError("this architecture currently supports live mode only")

    output_csv = os.path.join(output_dir, "realtime_predictions.csv")
    output_jsonl = os.path.join(output_dir, "realtime_predictions.jsonl")

    runner = SinglePipelineRunner(
        cfg=cfg,
        model_path=model_path,
        label_map_path=label_map_path,
        detector_json_path=detector_json_path,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        device=device,
        unknown_label=unknown_label,
    )
    runner.run_live(source, live_duration_seconds=live_duration_seconds)
