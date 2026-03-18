import argparse
import csv
import json
import os
import random
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import dpkt
import numpy as np
import torch
from torch.utils.data import Dataset


BYTE_PAD_TOKEN = 256
DEFAULT_NUM_PACKETS = 12
DEFAULT_PACKET_LEN = 256

CLASS_NAMES = [
    "nonvpn_chat",
    "nonvpn_email",
    "nonvpn_file_transfer",
    "nonvpn_p2p",
    "nonvpn_streaming",
    "nonvpn_voip",
    "vpn_chat",
    "vpn_email",
    "vpn_file_transfer",
    "vpn_p2p",
    "vpn_streaming",
    "vpn_voip",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

NONVPN_PREFIX_TO_CLASS = {
    "aim_chat": "nonvpn_chat",
    "aimchat": "nonvpn_chat",
    "facebook_chat": "nonvpn_chat",
    "facebookchat": "nonvpn_chat",
    "hangout_chat": "nonvpn_chat",
    "hangouts_chat": "nonvpn_chat",
    "icq_chat": "nonvpn_chat",
    "icqchat": "nonvpn_chat",
    "skype_chat": "nonvpn_chat",
    "gmailchat": "nonvpn_chat",
    "email": "nonvpn_email",
    "ftps_down": "nonvpn_file_transfer",
    "ftps_up": "nonvpn_file_transfer",
    "scp": "nonvpn_file_transfer",
    "scpdown": "nonvpn_file_transfer",
    "scpup": "nonvpn_file_transfer",
    "sftp": "nonvpn_file_transfer",
    "sftp_down": "nonvpn_file_transfer",
    "sftp_up": "nonvpn_file_transfer",
    "sftpdown": "nonvpn_file_transfer",
    "sftpup": "nonvpn_file_transfer",
    "skype_file": "nonvpn_file_transfer",
    "bittorrent": "nonvpn_p2p",
    "torrent": "nonvpn_p2p",
    "netflix": "nonvpn_streaming",
    "spotify": "nonvpn_streaming",
    "vimeo": "nonvpn_streaming",
    "youtube": "nonvpn_streaming",
    "youtubehtml": "nonvpn_streaming",
    "facebook_video": "nonvpn_streaming",
    "hangouts_video": "nonvpn_streaming",
    "skype_video": "nonvpn_streaming",
    "facebook_audio": "nonvpn_voip",
    "hangouts_audio": "nonvpn_voip",
    "skype_audio": "nonvpn_voip",
    "voipbuster": "nonvpn_voip",
}

VPN_PREFIX_TO_CLASS = {
    "aim_chat": "vpn_chat",
    "facebook_chat": "vpn_chat",
    "hangouts_chat": "vpn_chat",
    "icq_chat": "vpn_chat",
    "skype_chat": "vpn_chat",
    "email": "vpn_email",
    "ftps_a": "vpn_file_transfer",
    "ftps_b": "vpn_file_transfer",
    "sftp_a": "vpn_file_transfer",
    "sftp_b": "vpn_file_transfer",
    "skype_files": "vpn_file_transfer",
    "bittorrent": "vpn_p2p",
    "netflix_a": "vpn_streaming",
    "spotify_a": "vpn_streaming",
    "vimeo_a": "vpn_streaming",
    "vimeo_b": "vpn_streaming",
    "youtube_a": "vpn_streaming",
    "facebook_video": "vpn_streaming",
    "hangouts_video": "vpn_streaming",
    "skype_video": "vpn_streaming",
    "facebook_audio": "vpn_voip",
    "hangouts_audio": "vpn_voip",
    "skype_audio": "vpn_voip",
    "voipbuster": "vpn_voip",
}


FlowKey = Tuple[str, str, int, int, int]


@dataclass
class FlowState:
    packets: List[np.ndarray] = field(default_factory=list)
    last_ts: float = 0.0
    emitted: bool = False


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_pcap_reader(file_obj):
    magic = file_obj.read(4)
    file_obj.seek(0)
    if magic == b"\n\r\r\n":
        return dpkt.pcapng.Reader(file_obj)
    if magic == b"\xd4\xc3\xb2\xa1":
        return dpkt.pcap.Reader(file_obj)
    raise ValueError("unsupported pcap format")


def _safe_ip(ip_bytes: bytes) -> str:
    try:
        return socket.inet_ntoa(ip_bytes)
    except OSError:
        return "0.0.0.0"


def _canonical_flow_key(
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
    proto: int,
) -> FlowKey:
    left = (src_ip, src_port)
    right = (dst_ip, dst_port)
    if left <= right:
        return src_ip, dst_ip, src_port, dst_port, proto
    return dst_ip, src_ip, dst_port, src_port, proto


def _packet_to_vector(payload: bytes, packet_len: int, pad_token: int) -> np.ndarray:
    arr = np.full((packet_len,), pad_token, dtype=np.uint16)
    if len(payload) == 0:
        return arr
    payload_np = np.frombuffer(payload[:packet_len], dtype=np.uint8).astype(np.uint16)
    arr[: payload_np.shape[0]] = payload_np
    return arr


def _packet_from_raw(raw_pkt: bytes, datalink: int):
    ip_pkt = None
    try:
        if datalink == 1:
            eth = dpkt.ethernet.Ethernet(raw_pkt)
            if isinstance(eth.data, dpkt.ip.IP):
                ip_pkt = eth.data
        elif datalink == 101:
            ip_pkt = dpkt.ip.IP(raw_pkt)
        elif datalink == 113:
            sll = dpkt.sll.SLL(raw_pkt)
            if isinstance(sll.data, dpkt.ip.IP):
                ip_pkt = sll.data
    except Exception:
        return None

    if ip_pkt is None:
        return None

    src_ip = _safe_ip(getattr(ip_pkt, "src", b"\x00\x00\x00\x00"))
    dst_ip = _safe_ip(getattr(ip_pkt, "dst", b"\x00\x00\x00\x00"))

    fin = False
    rst = False
    payload = None
    src_port = 0
    dst_port = 0

    if isinstance(ip_pkt.data, dpkt.tcp.TCP):
        tcp = ip_pkt.data
        src_port = int(getattr(tcp, "sport", 0))
        dst_port = int(getattr(tcp, "dport", 0))
        flags = int(getattr(tcp, "flags", 0))
        fin = bool(flags & dpkt.tcp.TH_FIN)
        rst = bool(flags & dpkt.tcp.TH_RST)

        # Keep payload bytes only.
        payload = bytes(tcp.data)
        if flags == dpkt.tcp.TH_ACK and len(payload) == 0:
            return None
        proto = 6
    elif isinstance(ip_pkt.data, dpkt.udp.UDP):
        udp = ip_pkt.data
        src_port = int(getattr(udp, "sport", 0))
        dst_port = int(getattr(udp, "dport", 0))
        payload = bytes(udp.data)
        proto = 17
    else:
        return None

    key = _canonical_flow_key(src_ip, src_port, dst_ip, dst_port, proto)
    return key, payload, fin, rst


def infer_label_from_filename(file_name: str, domain: str) -> int:
    stem = os.path.splitext(os.path.basename(file_name).lower())[0]
    if domain == "vpn" and stem.startswith("vpn_"):
        stem = stem[4:]

    cleaned = stem
    while len(cleaned) > 0 and cleaned[-1].isdigit():
        cleaned = cleaned[:-1]
    cleaned = cleaned.rstrip("_")

    if domain == "vpn":
        class_name = VPN_PREFIX_TO_CLASS.get(cleaned)
    else:
        class_name = NONVPN_PREFIX_TO_CLASS.get(cleaned)

    if class_name is None:
        raise ValueError(
            f"unknown prefix: {cleaned} (domain={domain}, file={file_name})"
        )
    return CLASS_TO_ID[class_name]


def scan_capture_files(dataset_root: str) -> List[Tuple[str, str, str, int]]:
    records: List[Tuple[str, str, str, int]] = []
    for top in os.listdir(dataset_root):
        full = os.path.join(dataset_root, top)
        if not os.path.isdir(full):
            continue
        name = top.lower()
        if name.startswith("nonvpn"):
            domain = "nonvpn"
        elif name.startswith("vpn"):
            domain = "vpn"
        else:
            continue

        for root, _, files in os.walk(full):
            for f in files:
                if not f.lower().endswith((".pcap", ".pcapng")):
                    continue
                file_path = os.path.join(root, f)
                rel = os.path.relpath(file_path, dataset_root)
                try:
                    label = infer_label_from_filename(f, domain)
                except ValueError:
                    continue
                records.append((file_path, rel, domain, label))
    return sorted(records)


def _finalize_flow(
    flow_key: FlowKey,
    packets: List[np.ndarray],
    num_packets: int,
    packet_len: int,
    pad_token: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    bytes_arr = np.full((num_packets, packet_len), pad_token, dtype=np.uint16)
    mask = np.zeros((num_packets,), dtype=np.bool_)

    real_count = min(len(packets), num_packets)
    if real_count > 0:
        bytes_arr[:real_count] = np.stack(packets[:real_count], axis=0)
        mask[:real_count] = True

    flow_id = f"{flow_key[0]}:{flow_key[2]}-{flow_key[1]}:{flow_key[3]}-p{flow_key[4]}"
    return bytes_arr, mask, flow_id


def extract_sessions_from_capture(
    capture_path: str,
    num_packets: int,
    packet_len: int,
    pad_token: int,
    flow_timeout_s: float,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    states: Dict[FlowKey, FlowState] = {}
    sessions: List[Tuple[np.ndarray, np.ndarray, str]] = []

    def sweep(now_ts: float) -> None:
        timeout_keys: List[FlowKey] = []
        for key, st in states.items():
            if st.emitted:
                timeout_keys.append(key)
                continue
            if now_ts - st.last_ts >= flow_timeout_s and len(st.packets) > 0:
                timeout_keys.append(key)

        for key in timeout_keys:
            st = states.get(key)
            if st is None:
                continue
            if not st.emitted and len(st.packets) > 0:
                sessions.append(
                    _finalize_flow(
                        key,
                        st.packets,
                        num_packets,
                        packet_len,
                        pad_token,
                    )
                )
            states.pop(key, None)

    with open(capture_path, "rb") as fp:
        reader = detect_pcap_reader(fp)
        datalink = reader.datalink() if hasattr(reader, "datalink") else 1

        for ts, raw_pkt in reader:
            parsed = _packet_from_raw(raw_pkt, datalink)
            if parsed is None:
                continue
            key, payload, fin, rst = parsed

            state = states.get(key)
            if state is None:
                state = FlowState(last_ts=float(ts))
                states[key] = state

            state.last_ts = float(ts)
            if not state.emitted:
                state.packets.append(_packet_to_vector(payload, packet_len, pad_token))

            if len(state.packets) >= num_packets:
                sessions.append(
                    _finalize_flow(
                        key,
                        state.packets,
                        num_packets,
                        packet_len,
                        pad_token,
                    )
                )
                state.emitted = True

            if fin or rst:
                if not state.emitted and len(state.packets) > 0:
                    sessions.append(
                        _finalize_flow(
                            key,
                            state.packets,
                            num_packets,
                            packet_len,
                            pad_token,
                        )
                    )
                states.pop(key, None)

            sweep(float(ts))

    # flush all remaining incomplete flows
    sweep(float("inf"))
    return sessions


def _stratified_split_indices(
    labels: List[int], train_ratio: float, seed: int
) -> List[str]:
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = {}
    for idx, lb in enumerate(labels):
        by_label.setdefault(lb, []).append(idx)

    split = ["test"] * len(labels)
    for _, idx_list in by_label.items():
        rng.shuffle(idx_list)
        train_num = max(1, int(round(len(idx_list) * train_ratio)))
        train_num = min(train_num, len(idx_list))
        for idx in idx_list[:train_num]:
            split[idx] = "train"
    return split


def build_session_manifest(
    dataset_root: str,
    output_root: str,
    num_packets: int = DEFAULT_NUM_PACKETS,
    packet_len: int = DEFAULT_PACKET_LEN,
    pad_token: int = BYTE_PAD_TOKEN,
    train_ratio: float = 0.8,
    flow_timeout_s: float = 10.0,
    max_flows_per_class: int = 0,
    seed: int = 42,
) -> str:
    _mkdir(output_root)
    sample_dir = os.path.join(output_root, "samples")
    _mkdir(sample_dir)

    captures = scan_capture_files(dataset_root)
    if len(captures) == 0:
        raise ValueError(f"no pcap files found under {dataset_root}")

    rows = []
    class_counter: Dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}
    sample_id = 0

    for file_path, rel_path, _domain, label in captures:
        if max_flows_per_class > 0 and class_counter[label] >= max_flows_per_class:
            continue

        sessions = extract_sessions_from_capture(
            file_path,
            num_packets=num_packets,
            packet_len=packet_len,
            pad_token=pad_token,
            flow_timeout_s=flow_timeout_s,
        )

        for bytes_arr, packet_mask, flow_id in sessions:
            if max_flows_per_class > 0 and class_counter[label] >= max_flows_per_class:
                break

            sample_id += 1
            class_counter[label] += 1
            sample_path = os.path.join(sample_dir, f"sample_{sample_id:08d}.npz")
            np.savez_compressed(
                sample_path,
                bytes=bytes_arr,
                packet_mask=packet_mask,
                label=np.array([label], dtype=np.int64),
                flow_id=np.array([flow_id], dtype=object),
            )

            rows.append(
                {
                    "sample_path": sample_path,
                    "label": label,
                    "flow_id": flow_id,
                    "source_rel": rel_path,
                    "class_name": CLASS_NAMES[label],
                }
            )

    if len(rows) == 0:
        raise ValueError("no valid flow sessions extracted")

    split = _stratified_split_indices(
        [r["label"] for r in rows], train_ratio=train_ratio, seed=seed
    )
    for idx, part in enumerate(split):
        rows[idx]["split"] = part

    manifest_path = os.path.join(output_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample_path", "split", "label", "class_name", "flow_id", "source_rel"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["sample_path"],
                    row["split"],
                    row["label"],
                    row["class_name"],
                    row["flow_id"],
                    row["source_rel"],
                ]
            )

    label_map_path = os.path.join(output_root, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: int(v) for k, v in CLASS_TO_ID.items()}, f, ensure_ascii=False, indent=2
        )

    stats_path = os.path.join(output_root, "summary.json")
    summary = {
        "dataset_root": dataset_root,
        "num_packets": num_packets,
        "packet_len": packet_len,
        "pad_token": pad_token,
        "train_ratio": train_ratio,
        "total_samples": len(rows),
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in class_counter.items()},
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] manifest={manifest_path} total_samples={len(rows)}")
    return manifest_path


class ByteSessionDataset(Dataset):
    def __init__(self, manifest_path: str, split: Optional[str] = None):
        self.rows: List[Dict[str, str]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is not None and row["split"] != split:
                    continue
                self.rows.append(row)

        if len(self.rows) == 0:
            raise ValueError(
                f"empty dataset for split={split}, manifest={manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        sample = np.load(row["sample_path"], allow_pickle=True)
        bytes_arr = torch.from_numpy(sample["bytes"].astype(np.int64))
        packet_mask = torch.from_numpy(sample["packet_mask"].astype(np.bool_))
        label = int(row["label"])
        flow_id = str(row["flow_id"])
        return {
            "bytes": bytes_arr,
            "packet_mask": packet_mask,
            "label": label,
            "flow_id": flow_id,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build byte-session manifest from pcap files"
    )
    parser.add_argument("--dataset-root", default=os.path.join("dataset", "iscx"))
    parser.add_argument(
        "--output-root", default=os.path.join("processed_full", "sessions")
    )
    parser.add_argument("--num-packets", type=int, default=DEFAULT_NUM_PACKETS)
    parser.add_argument("--packet-len", type=int, default=DEFAULT_PACKET_LEN)
    parser.add_argument("--pad-token", type=int, default=BYTE_PAD_TOKEN)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--flow-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-flows-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_session_manifest(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        num_packets=args.num_packets,
        packet_len=args.packet_len,
        pad_token=args.pad_token,
        train_ratio=args.train_ratio,
        flow_timeout_s=args.flow_timeout_s,
        max_flows_per_class=args.max_flows_per_class,
        seed=args.seed,
    )
