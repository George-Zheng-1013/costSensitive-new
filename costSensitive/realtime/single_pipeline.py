import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import dpkt
import numpy as np

from .config import RealtimeConfig
from .flow_assembler import FlowAssembler, FlowSample, parse_packet_to_event


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
        output_csv: str,
        output_jsonl: str,
        device: Optional[str] = None,
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

        self.model = ByteSessionClassifier(num_classes=num_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

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

    def _iter_pcap_files(self, source: str) -> List[str]:
        if os.path.isfile(source):
            return [source]
        out = []
        for root, _, files in os.walk(source):
            for name in files:
                if name.lower().endswith((".pcap", ".pcapng")):
                    out.append(os.path.join(root, name))
        return sorted(out)

    def _detect_reader(self, file_obj):
        magic = file_obj.read(4)
        file_obj.seek(0)
        if magic == b"\n\r\r\n":
            return dpkt.pcapng.Reader(file_obj)
        if magic == b"\xd4\xc3\xb2\xa1":
            return dpkt.pcap.Reader(file_obj)
        raise ValueError("unsupported pcap format")

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

    def run_pcap(self, source: str):
        assembler = FlowAssembler(self.cfg)
        files = self._iter_pcap_files(source)
        if len(files) == 0:
            raise ValueError(f"no pcap files found: {source}")

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
                ]
            )

            last_sweep = time.time()
            for pcap_path in files:
                with open(pcap_path, "rb") as fp:
                    reader = self._detect_reader(fp)
                    datalink = reader.datalink() if hasattr(reader, "datalink") else 1
                    for ts, pkt in reader:
                        evt = parse_packet_to_event(float(ts), pkt, datalink)
                        if evt is None:
                            continue
                        for sample in assembler.feed(evt):
                            self._accept(sample, writer, f_jsonl)

                        now = time.time()
                        if now - last_sweep >= self.cfg.sweep_interval_s:
                            for sample in assembler.sweep_timeouts(now):
                                self._accept(sample, writer, f_jsonl)
                            last_sweep = now

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
    output_dir: str,
    device: Optional[str] = None,
    live_duration_seconds: float = 0.0,
):
    if mode != "pcap":
        raise ValueError("new architecture currently supports pcap mode only")

    output_csv = os.path.join(output_dir, "realtime_predictions.csv")
    output_jsonl = os.path.join(output_dir, "realtime_predictions.jsonl")

    runner = SinglePipelineRunner(
        cfg=cfg,
        model_path=model_path,
        label_map_path=label_map_path,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        device=device,
    )
    runner.run_pcap(source)
