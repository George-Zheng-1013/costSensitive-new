import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import dpkt
import numpy as np

from .config import RealtimeConfig
from .flow_assembler import (
    FlowAssembler,
    FlowSample,
    PacketEvent,
    parse_packet_to_event,
    sanitize_ip_packet,
)


def _setup_windows_dll_paths() -> None:
    if sys.platform != "win32":
        return

    candidates = [
        r"C:\Windows\System32\Npcap",
    ]

    for path in candidates:
        if not os.path.isdir(path):
            continue
        try:
            os.add_dll_directory(path)
        except Exception:
            pass
        if path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


_setup_windows_dll_paths()


def _packet_to_event_from_scapy(
    packet, ip_cls, tcp_cls, udp_cls
) -> Optional[PacketEvent]:
    if not packet.haslayer(ip_cls):
        return None

    ip_layer = packet[ip_cls]
    if packet.haslayer(tcp_cls):
        trans = packet[tcp_cls]
        proto = 6
    elif packet.haslayer(udp_cls):
        trans = packet[udp_cls]
        proto = 17
    else:
        return None

    parsed = sanitize_ip_packet(bytes(ip_layer))
    if parsed is None:
        return None

    payload, fin, rst = parsed
    return PacketEvent(
        ts=time.time(),
        src_ip=str(ip_layer.src),
        dst_ip=str(ip_layer.dst),
        src_port=int(trans.sport),
        dst_port=int(trans.dport),
        protocol=proto,
        payload=payload,
        fin=fin,
        rst=rst,
    )


class SinglePipelineRunner:
    def __init__(
        self,
        cfg: RealtimeConfig,
        model_path: str,
        label_map_path: Optional[str],
        output_csv: str,
        output_jsonl: str,
        device: Optional[str] = None,
        unknown_enable: bool = False,
        unknown_detector_path: Optional[str] = None,
        unknown_threshold_conf: Optional[float] = None,
        unknown_threshold_anom: Optional[float] = None,
    ):
        import torch
        import torch.nn.functional as F
        from .model_def import ConvNet

        self._torch = torch
        self._F = F
        self.cfg = cfg
        self.device = (
            self._torch.device(device)
            if device
            else self._torch.device(
                "cuda" if self._torch.cuda.is_available() else "cpu"
            )
        )
        self.model = ConvNet().to(self.device)
        self.model.load_state_dict(
            self._torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        self.unknown_enable = bool(unknown_enable)
        self.unknown_detector_path = unknown_detector_path
        self.unknown_tau1 = unknown_threshold_conf
        self.unknown_tau2 = unknown_threshold_anom
        self.unknown_detector_name = None
        self.unknown_per_class_detectors = None
        self.unknown_global_detector = None
        self._load_unknown_detector()

        self.label_map = self._load_label_map(label_map_path)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        self.output_csv = output_csv
        self.output_jsonl = output_jsonl

        self.batch: List[dict] = []
        self.last_flush = time.time()

        self.total_packets = 0
        self.total_parsed_packets = 0
        self.total_dropped_packets = 0
        self.total_samples = 0
        self.total_results = 0
        self.samples_by_trigger: Dict[str, int] = {
            "length": 0,
            "timeout": 0,
            "fin_rst": 0,
            "other": 0,
        }
        self.last_heartbeat = time.time()

    def _load_unknown_detector(self):
        if not self.unknown_enable:
            return

        if self.unknown_detector_path and os.path.exists(self.unknown_detector_path):
            try:
                import joblib

                payload = joblib.load(self.unknown_detector_path)
                self.unknown_detector_name = payload.get("detector_name")
                self.unknown_per_class_detectors = payload.get("per_class_detectors")
                self.unknown_global_detector = payload.get("global_detector")

                thresholds = payload.get("thresholds", {})
                if self.unknown_tau1 is None:
                    self.unknown_tau1 = thresholds.get("tau1")
                if self.unknown_tau2 is None:
                    self.unknown_tau2 = thresholds.get("tau2")
            except Exception as exc:
                print(f"[WARN] unknown检测器加载失败: {exc}")

        if self.unknown_tau1 is None:
            print("[WARN] unknown_enable已开启，但tau1缺失，降级为仅分类输出")
            self.unknown_enable = False
            return

        print(
            "[UNKNOWN] enabled=1 detector={} tau1={} tau2={}".format(
                self.unknown_detector_name,
                self.unknown_tau1,
                self.unknown_tau2,
            )
        )

    def _score_anomaly_batch(self, features, pred):
        if (
            self.unknown_per_class_detectors is None
            or self.unknown_global_detector is None
            or self.unknown_tau2 is None
        ):
            return np.full((features.shape[0],), np.nan, dtype=np.float32)

        feat_np = features.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        scores = np.zeros((feat_np.shape[0],), dtype=np.float32)
        for i in range(feat_np.shape[0]):
            class_id = int(pred_np[i])
            detector = self.unknown_per_class_detectors.get(
                class_id, self.unknown_global_detector
            )
            scores[i] = float(detector.decision_function(feat_np[i : i + 1])[0])
        return scores

    def _load_label_map(self, path: Optional[str]) -> Dict[int, str]:
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(v): k for k, v in raw.items()}

    def _heartbeat(self, assembler: FlowAssembler):
        now = time.time()
        if now - self.last_heartbeat < 5.0:
            return
        print(
            "[HEARTBEAT] packets_total={} parsed={} dropped={} samples_total={} results_total={} trigger(length/timeout/fin)={}/{}/{} active_flows={} batch_buffer={}".format(
                self.total_packets,
                self.total_parsed_packets,
                self.total_dropped_packets,
                self.total_samples,
                self.total_results,
                self.samples_by_trigger["length"],
                self.samples_by_trigger["timeout"],
                self.samples_by_trigger["fin_rst"],
                len(assembler.active),
                len(self.batch),
            )
        )
        self.last_heartbeat = now

    def _batch_to_tensor(self):
        arr_list = []
        for item in self.batch:
            img = (
                np.frombuffer(item["raw784"], dtype=np.uint8)
                .reshape(28, 28)
                .astype(np.float32)
                / 255.0
            )
            arr_list.append(img)
        batch_np = np.stack(arr_list, axis=0)
        batch_np = np.expand_dims(batch_np, axis=1)
        return self._torch.from_numpy(batch_np).to(self.device)

    def _flush_batch(self, writer, jsonf):
        if len(self.batch) == 0:
            return

        with self._torch.no_grad():
            x = self._batch_to_tensor()
            logits, features = self.model(x, return_features=True)
            probs = self._F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        anomaly_scores = np.full((len(self.batch),), np.nan, dtype=np.float32)
        if self.unknown_enable:
            anomaly_scores = self._score_anomaly_batch(features, pred)

        infer_ts = time.time()
        for i, item in enumerate(self.batch):
            pred_id = int(pred[i].item())
            confidence = float(conf[i].item())

            low_conf = bool(
                self.unknown_enable and confidence < float(self.unknown_tau1)
            )
            high_anom = bool(
                self.unknown_enable
                and self.unknown_tau2 is not None
                and np.isfinite(anomaly_scores[i])
                and anomaly_scores[i] > float(self.unknown_tau2)
            )

            is_unknown = bool(low_conf or high_anom)
            if not self.unknown_enable:
                unknown_reason = "unknown_disabled"
            elif low_conf and high_anom:
                unknown_reason = "low_confidence_and_high_anomaly"
            elif low_conf:
                unknown_reason = "low_confidence"
            elif high_anom:
                unknown_reason = "high_anomaly"
            else:
                unknown_reason = "known"

            record = {
                "sample_id": item["sample_id"],
                "flow_key": item["flow_key"],
                "trigger": item["trigger"],
                "capture_ts": item["ts"],
                "enqueue_ts": item["enqueue_ts"],
                "infer_ts": infer_ts,
                "end2end_ms": (infer_ts - item["enqueue_ts"]) * 1000.0,
                "pred": pred_id,
                "pred_name": self.label_map.get(pred_id, str(pred_id)),
                "confidence": confidence,
                "anomaly_score": (
                    float(anomaly_scores[i]) if np.isfinite(anomaly_scores[i]) else None
                ),
                "is_unknown": int(is_unknown),
                "unknown_reason": unknown_reason,
            }

            writer.writerow(
                [
                    record["sample_id"],
                    str(record["flow_key"]),
                    record["trigger"],
                    f"{record['capture_ts']:.6f}",
                    f"{record['enqueue_ts']:.6f}",
                    f"{record['infer_ts']:.6f}",
                    f"{record['end2end_ms']:.3f}",
                    record["pred"],
                    record["pred_name"],
                    f"{record['confidence']:.6f}",
                    (
                        f"{record['anomaly_score']:.6f}"
                        if record["anomaly_score"] is not None
                        else ""
                    ),
                    record["is_unknown"],
                    record["unknown_reason"],
                ]
            )
            jsonf.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.total_results += 1

        self.batch.clear()
        self.last_flush = time.time()

    def _accept_sample(self, sample: FlowSample, writer, jsonf):
        self.total_samples += 1
        trigger = (
            sample.trigger if sample.trigger in self.samples_by_trigger else "other"
        )
        self.samples_by_trigger[trigger] += 1
        self.batch.append(
            {
                "sample_id": sample.sample_id,
                "flow_key": sample.flow_key,
                "ts": sample.ts,
                "trigger": sample.trigger,
                "enqueue_ts": time.time(),
                "raw784": sample.raw784,
            }
        )

        now = time.time()
        if (
            len(self.batch) >= self.cfg.batch_size
            or (now - self.last_flush) * 1000.0 >= self.cfg.batch_wait_ms
        ):
            self._flush_batch(writer, jsonf)

    def _detect_reader(self, file_obj):
        magic = file_obj.read(4)
        file_obj.seek(0)
        if magic == b"\n\r\r\n":
            return dpkt.pcapng.Reader(file_obj)
        if magic == b"\xd4\xc3\xb2\xa1":
            return dpkt.pcap.Reader(file_obj)
        raise ValueError("unsupported pcap format")

    def _iter_pcap_files(self, source: str) -> List[str]:
        if os.path.isfile(source):
            return [source]
        targets = []
        for root, _, files in os.walk(source):
            for name in files:
                if name.lower().endswith((".pcap", ".pcapng")):
                    targets.append(os.path.join(root, name))
        return sorted(targets)

    def run_pcap(self, source: str):
        assembler = FlowAssembler(self.cfg)
        files = self._iter_pcap_files(source)
        if len(files) == 0:
            raise ValueError(f"未发现pcap文件: {source}")

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
                    "enqueue_ts",
                    "infer_ts",
                    "end2end_ms",
                    "pred",
                    "pred_name",
                    "confidence",
                    "anomaly_score",
                    "is_unknown",
                    "unknown_reason",
                ]
            )

            last_sweep = time.time()
            for pcap_path in files:
                with open(pcap_path, "rb") as fp:
                    reader = self._detect_reader(fp)
                    datalink = reader.datalink() if hasattr(reader, "datalink") else 1
                    for ts, pkt in reader:
                        self.total_packets += 1
                        evt = parse_packet_to_event(float(ts), pkt, datalink)
                        if evt is None:
                            self.total_dropped_packets += 1
                            self._heartbeat(assembler)
                            continue
                        self.total_parsed_packets += 1

                        for sample in assembler.feed(evt):
                            self._accept_sample(sample, writer, f_jsonl)

                        now = time.time()
                        if now - last_sweep >= self.cfg.sweep_interval_s:
                            for sample in assembler.sweep_timeouts(now):
                                self._accept_sample(sample, writer, f_jsonl)
                            last_sweep = now

                        self._heartbeat(assembler)

            for sample in assembler.sweep_timeouts(
                time.time() + self.cfg.flow_timeout_s + 1
            ):
                self._accept_sample(sample, writer, f_jsonl)
            self._flush_batch(writer, f_jsonl)

        print(
            "[DONE] packets={} parsed={} dropped={} samples={} results={} trigger(length/timeout/fin)={}/{}/{} output_csv={}".format(
                self.total_packets,
                self.total_parsed_packets,
                self.total_dropped_packets,
                self.total_samples,
                self.total_results,
                self.samples_by_trigger["length"],
                self.samples_by_trigger["timeout"],
                self.samples_by_trigger["fin_rst"],
                self.output_csv,
            )
        )

    def run_live(self, source: str, duration_seconds: float = 0.0):
        try:
            from scapy.all import sniff
            from scapy.layers.inet import IP, TCP, UDP
        except Exception as exc:
            raise RuntimeError(
                f"Scapy不可用，请先安装: pip install scapy。detail={exc}"
            )

        assembler = FlowAssembler(self.cfg)
        start_ts = time.time()
        last_sweep = time.time()

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
                    "enqueue_ts",
                    "infer_ts",
                    "end2end_ms",
                    "pred",
                    "pred_name",
                    "confidence",
                    "anomaly_score",
                    "is_unknown",
                    "unknown_reason",
                ]
            )

            def _callback(packet):
                self.total_packets += 1
                evt = _packet_to_event_from_scapy(packet, IP, TCP, UDP)
                if evt is None:
                    self.total_dropped_packets += 1
                    return
                self.total_parsed_packets += 1

                for sample in assembler.feed(evt):
                    self._accept_sample(sample, writer, f_jsonl)

            try:
                while True:
                    now = time.time()
                    if duration_seconds > 0 and (now - start_ts) >= duration_seconds:
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
                            self._accept_sample(sample, writer, f_jsonl)
                        last_sweep = now

                    self._heartbeat(assembler)
            except KeyboardInterrupt:
                pass
            finally:
                for sample in assembler.sweep_timeouts(
                    time.time() + self.cfg.flow_timeout_s + 1
                ):
                    self._accept_sample(sample, writer, f_jsonl)
                self._flush_batch(writer, f_jsonl)

        print(
            "[DONE-LIVE] packets={} parsed={} dropped={} samples={} results={} trigger(length/timeout/fin)={}/{}/{} output_csv={}".format(
                self.total_packets,
                self.total_parsed_packets,
                self.total_dropped_packets,
                self.total_samples,
                self.total_results,
                self.samples_by_trigger["length"],
                self.samples_by_trigger["timeout"],
                self.samples_by_trigger["fin_rst"],
                self.output_csv,
            )
        )


def run_single_pipeline(
    mode: str,
    source: str,
    cfg: RealtimeConfig,
    model_path: str,
    label_map_path: Optional[str],
    output_dir: str,
    device: Optional[str] = None,
    live_duration_seconds: float = 0.0,
    unknown_enable: bool = False,
    unknown_detector_path: Optional[str] = None,
    unknown_threshold_conf: Optional[float] = None,
    unknown_threshold_anom: Optional[float] = None,
):
    output_csv = os.path.join(output_dir, "realtime_predictions.csv")
    output_jsonl = os.path.join(output_dir, "realtime_predictions.jsonl")

    runner = SinglePipelineRunner(
        cfg=cfg,
        model_path=model_path,
        label_map_path=label_map_path,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        device=device,
        unknown_enable=unknown_enable,
        unknown_detector_path=unknown_detector_path,
        unknown_threshold_conf=unknown_threshold_conf,
        unknown_threshold_anom=unknown_threshold_anom,
    )

    if mode == "pcap":
        runner.run_pcap(source)
    else:
        runner.run_live(source, duration_seconds=live_duration_seconds)
