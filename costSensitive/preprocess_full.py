import os
import csv
import json
import math
import errno
import random
import shutil
import argparse
import time
import re
from collections import defaultdict

import dpkt
import numpy as np
from PIL import Image

from mnist import CLASS_NAMES, CLASS_TO_ID, build_idx_dataset


PACKET_LEN = 784
PNG_SIZE = 28
PAD_LEN = 50
VALID_MAX_LEN = 1520


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


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def detect_pcap_reader(file_obj):
    magic_head = file_obj.read(4)
    file_obj.seek(0, 0)
    if magic_head == b"\n\r\r\n":
        return dpkt.pcapng.Reader(file_obj)
    if magic_head == b"\xd4\xc3\xb2\xa1":
        return dpkt.pcap.Reader(file_obj)
    raise ValueError("unsupported pcap format")


def infer_label_from_filename(filename, domain):
    name = os.path.splitext(os.path.basename(filename).lower())[0]
    if domain == "vpn" and name.startswith("vpn_"):
        name = name[4:]

    normalized_prefix = re.sub(r"(_?[0-9]+[a-z]?)$", "", name)
    normalized_prefix = re.sub(r"\d+$", "", normalized_prefix).rstrip("_")

    if domain == "vpn":
        class_name = VPN_PREFIX_TO_CLASS.get(normalized_prefix)
        if class_name is not None:
            return CLASS_TO_ID[class_name]
        raise ValueError(
            "[vpn] unknown prefix: {} (file={})".format(normalized_prefix, name)
        )

    class_name = NONVPN_PREFIX_TO_CLASS.get(normalized_prefix)
    if class_name is not None:
        return CLASS_TO_ID[class_name]
    raise ValueError(
        "[nonvpn] unknown prefix: {} (file={})".format(normalized_prefix, name)
    )


def iter_capture_samples(capture_path):
    def extract_ip_from_packet(pkt_bytes, datalink):
        if datalink == 1:
            eth = dpkt.ethernet.Ethernet(pkt_bytes)
            if isinstance(eth.data, dpkt.ip.IP):
                return eth.data
            return None

        if datalink == 101:
            return dpkt.ip.IP(pkt_bytes)

        if datalink == 113:
            sll = dpkt.sll.SLL(pkt_bytes)
            if isinstance(sll.data, dpkt.ip.IP):
                return sll.data
            return None

        return None

    def sanitize_and_payload(ip_pkt):
        ip_pkt.src = b"\x00\x00\x00\x00"
        ip_pkt.dst = b"\x00\x00\x00\x00"

        if isinstance(ip_pkt.data, dpkt.tcp.TCP):
            trans = ip_pkt.data
            trans.sport = 0
            trans.dport = 0
            if trans.flags == 16 and len(trans.data) == 0:
                return None
        elif isinstance(ip_pkt.data, dpkt.udp.UDP):
            trans = ip_pkt.data
            trans.sport = 0
            trans.dport = 0
        else:
            return None

        payload = bytes(ip_pkt)
        if 0 < len(payload) < VALID_MAX_LEN:
            return payload
        return None

    payload_all = bytearray()
    with open(capture_path, "rb") as f:
        reader = detect_pcap_reader(f)
        datalink = reader.datalink() if hasattr(reader, "datalink") else None

        for ts, pkt in reader:
            try:
                ip = extract_ip_from_packet(pkt, datalink)
                if ip is None:
                    continue

                payload = sanitize_and_payload(ip)
                if payload is None:
                    continue

                payload_all.extend(payload)
                payload_all.extend(b"\x00" * PAD_LEN)

                while len(payload_all) >= PACKET_LEN:
                    sample = bytes(payload_all[:PACKET_LEN])
                    del payload_all[:PACKET_LEN]
                    yield ts, sample
            except Exception:
                continue


def scan_capture_files(dataset_root):
    targets = []
    for name in os.listdir(dataset_root):
        full = os.path.join(dataset_root, name)
        if not os.path.isdir(full):
            continue
        lname = name.lower()
        if lname.startswith("nonvpn"):
            domain = "nonvpn"
        elif lname.startswith("vpn"):
            domain = "vpn"
        else:
            continue

        for root, _, files in os.walk(full):
            for filename in files:
                if filename.lower().endswith((".pcap", ".pcapng")):
                    path = os.path.join(root, filename)
                    rel = os.path.relpath(path, dataset_root)
                    targets.append((path, rel, domain))
    return sorted(targets)


def write_png(sample_bytes, out_path):
    arr = np.frombuffer(sample_bytes, dtype=np.uint8)
    if arr.shape[0] != PACKET_LEN:
        return False
    arr = arr.reshape(PNG_SIZE, PNG_SIZE)
    im = Image.fromarray(arr)
    im.save(out_path)
    return True


def stratified_group_split(records, train_ratio, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in records:
        by_class[item["class_name"]].append(item)

    train_ids = set()
    for class_name, items in by_class.items():
        by_source = defaultdict(list)
        for item in items:
            by_source[item["source_rel"]].append(item)

        groups = list(by_source.items())
        rng.shuffle(groups)
        total = len(items)
        target_train = int(math.floor(total * train_ratio))

        if len(groups) <= 1:
            for idx, item in enumerate(items):
                if idx < target_train:
                    train_ids.add(item["sample_id"])
            continue

        count_train = 0
        for source_rel, group_items in groups:
            if count_train < target_train:
                for item in group_items:
                    train_ids.add(item["sample_id"])
                count_train += len(group_items)

        # 防止某类全部进 train 或全部进 test
        class_train = [it for it in items if it["sample_id"] in train_ids]
        if len(class_train) == 0:
            first = groups[0][1][0]
            train_ids.add(first["sample_id"])
        if len(class_train) == len(items):
            last = groups[-1][1][0]
            if last["sample_id"] in train_ids:
                train_ids.remove(last["sample_id"])

    for item in records:
        item["split"] = "train" if item["sample_id"] in train_ids else "test"


def save_reports(output_root, records, failed_files, scanned_files):
    report_dir = os.path.join(output_root, "reports")
    mkdir_p(report_dir)

    manifest_path = os.path.join(report_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "split",
                "class_name",
                "class_id",
                "domain",
                "source_rel",
                "png_path",
            ]
        )
        for item in records:
            writer.writerow(
                [
                    item["sample_id"],
                    item["split"],
                    item["class_name"],
                    item["class_id"],
                    item["domain"],
                    item["source_rel"],
                    item["png_path"],
                ]
            )

    class_dist_path = os.path.join(report_dir, "class_distribution.csv")
    split_dist_path = os.path.join(report_dir, "split_distribution.csv")

    class_total = {name: 0 for name in CLASS_NAMES}
    split_total = {
        "train": {name: 0 for name in CLASS_NAMES},
        "test": {name: 0 for name in CLASS_NAMES},
    }
    for item in records:
        class_total[item["class_name"]] += 1
        split_total[item["split"]][item["class_name"]] += 1

    with open(class_dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "class_id", "count"])
        for class_name in CLASS_NAMES:
            writer.writerow(
                [class_name, CLASS_TO_ID[class_name], class_total[class_name]]
            )

    with open(split_dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_name", "class_id", "count"])
        for split_name in ["train", "test"]:
            for class_name in CLASS_NAMES:
                writer.writerow(
                    [
                        split_name,
                        class_name,
                        CLASS_TO_ID[class_name],
                        split_total[split_name][class_name],
                    ]
                )

    failed_path = os.path.join(report_dir, "failed_files.csv")
    with open(failed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_rel", "domain", "reason"])
        writer.writerows(failed_files)

    summary = {
        "scanned_capture_files": scanned_files,
        "failed_capture_files": len(failed_files),
        "total_samples": len(records),
        "train_samples": sum(1 for r in records if r["split"] == "train"),
        "test_samples": sum(1 for r in records if r["split"] == "test"),
        "class_names": CLASS_NAMES,
        "label_map": CLASS_TO_ID,
    }
    summary_path = os.path.join(report_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def clean_output(output_root):
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)


def preprocess_full(dataset_root, output_root, train_ratio, seed, max_per_class):
    captures = scan_capture_files(dataset_root)
    if len(captures) == 0:
        raise ValueError("No pcap/pcapng files found under: {}".format(dataset_root))

    total_files = len(captures)
    start_time = time.time()
    print(
        "[PROGRESS] stage=scan status=done total_capture_files={}".format(total_files),
        flush=True,
    )

    png_all_root = os.path.join(output_root, "png_all")
    png_split_root = os.path.join(output_root, "png")
    mnist_root = os.path.join(output_root, "mnist")

    mkdir_p(png_all_root)
    failed_files = []
    records = []
    sample_seq = 0
    progress_interval = 200
    class_counts = {name: 0 for name in CLASS_NAMES}

    class_file_map = {name: [] for name in CLASS_NAMES}
    for capture_path, source_rel, domain in captures:
        try:
            class_id = infer_label_from_filename(capture_path, domain)
            class_name = CLASS_NAMES[class_id]
            class_file_map[class_name].append((capture_path, source_rel, domain))
        except Exception as exc:
            failed_files.append((source_rel, domain, str(exc)))

    rng = random.Random(seed)
    file_counts = {}

    for class_name in CLASS_NAMES:
        class_id = CLASS_TO_ID[class_name]
        class_tmp_dir = os.path.join(png_all_root, class_name)
        mkdir_p(class_tmp_dir)

        class_files = class_file_map.get(class_name, [])
        rng.shuffle(class_files)
        derived_max_per_file = (
            max(1, int(math.ceil(max_per_class / len(class_files))))
            if len(class_files) > 0
            else 0
        )
        print(
            "[PROGRESS] stage=extract class={} files={} target={} max_per_file={}".format(
                class_name, len(class_files), max_per_class, derived_max_per_file
            ),
            flush=True,
        )

        active_states = []
        for capture_path, source_rel, domain in class_files:
            source_tag = (
                source_rel.replace("\\", "__").replace("/", "__").replace(".", "_")
            )
            active_states.append(
                {
                    "capture_path": capture_path,
                    "source_rel": source_rel,
                    "domain": domain,
                    "source_tag": source_tag,
                    "generator": iter_capture_samples(capture_path),
                    "file_samples": 0,
                }
            )
            file_counts[source_rel] = 0

        round_idx = 0
        while class_counts[class_name] < max_per_class and len(active_states) > 0:
            round_idx += 1
            produced_in_round = 0
            next_active_states = []

            for state in active_states:
                if class_counts[class_name] >= max_per_class:
                    break

                if state["file_samples"] >= derived_max_per_file:
                    print(
                        "[PROGRESS] stage=extract class={} source={} status=skip reason=file_full file_samples={}".format(
                            class_name, state["source_rel"], state["file_samples"]
                        ),
                        flush=True,
                    )
                    continue

                try:
                    _, sample_bytes = next(state["generator"])
                except StopIteration:
                    continue
                except Exception as exc:
                    failed_files.append(
                        (state["source_rel"], state["domain"], str(exc))
                    )
                    continue

                png_name = "{}__{:06d}.png".format(
                    state["source_tag"], state["file_samples"]
                )
                png_tmp_path = os.path.join(class_tmp_dir, png_name)

                if write_png(sample_bytes, png_tmp_path):
                    records.append(
                        {
                            "sample_id": sample_seq,
                            "domain": state["domain"],
                            "class_id": class_id,
                            "class_name": class_name,
                            "source_rel": state["source_rel"],
                            "tmp_png_path": png_tmp_path,
                        }
                    )
                    sample_seq += 1
                    state["file_samples"] += 1
                    class_counts[class_name] += 1
                    file_counts[state["source_rel"]] = state["file_samples"]
                    produced_in_round += 1

                    if (
                        state["file_samples"] % progress_interval == 0
                        or class_counts[class_name] == max_per_class
                    ):
                        print(
                            "[PROGRESS] stage=extract class={} source={} file_samples={} class_samples={} total_samples={}".format(
                                class_name,
                                state["source_rel"],
                                state["file_samples"],
                                class_counts[class_name],
                                sample_seq,
                            ),
                            flush=True,
                        )

                if state["file_samples"] < derived_max_per_file:
                    next_active_states.append(state)

            active_states = next_active_states

            if produced_in_round == 0:
                break

            if round_idx % 20 == 0:
                print(
                    "[PROGRESS] stage=extract class={} round={} class_samples={} active_files={}".format(
                        class_name,
                        round_idx,
                        class_counts[class_name],
                        len(active_states),
                    ),
                    flush=True,
                )

        print(
            "[PROGRESS] stage=extract class={} status=done class_samples={} total_samples={} active_files_left={}".format(
                class_name,
                class_counts[class_name],
                sample_seq,
                len(active_states),
            ),
            flush=True,
        )

    if len(records) == 0:
        raise ValueError(
            "No samples extracted. Please check dataset and filename rules."
        )

    print(
        "[PROGRESS] stage=extract status=done total_samples={} failed_files={} elapsed={:.2f}s".format(
            len(records), len(failed_files), time.time() - start_time
        ),
        flush=True,
    )
    print(
        "[PROGRESS] stage=extract class_counts={} max_per_class={}".format(
            class_counts, max_per_class
        ),
        flush=True,
    )
    print(
        "[PROGRESS] stage=extract file_count_stats min={} max={} files_with_samples={}".format(
            min(file_counts.values()) if len(file_counts) > 0 else 0,
            max(file_counts.values()) if len(file_counts) > 0 else 0,
            sum(1 for v in file_counts.values() if v > 0),
        ),
        flush=True,
    )

    stratified_group_split(records, train_ratio=train_ratio, seed=seed)
    print("[PROGRESS] stage=split status=done", flush=True)

    for split_name in ["train", "test"]:
        for class_name in CLASS_NAMES:
            mkdir_p(os.path.join(png_split_root, split_name, class_name))

    move_start = time.time()
    for move_idx, item in enumerate(records, start=1):
        src = item["tmp_png_path"]
        dst = os.path.join(
            png_split_root,
            item["split"],
            item["class_name"],
            os.path.basename(src),
        )
        shutil.move(src, dst)
        item["png_path"] = dst
        item.pop("tmp_png_path", None)
        if move_idx % 5000 == 0 or move_idx == len(records):
            print(
                "[PROGRESS] stage=organize moved={}/{} elapsed={:.2f}s".format(
                    move_idx, len(records), time.time() - move_start
                ),
                flush=True,
            )

    idx_start = time.time()
    print("[PROGRESS] stage=idx status=start", flush=True)
    build_idx_dataset(
        png_root=png_split_root,
        output_dir=mnist_root,
        report_dir=os.path.join(output_root, "reports"),
        seed=seed,
    )
    print(
        "[PROGRESS] stage=idx status=done elapsed={:.2f}s".format(
            time.time() - idx_start
        ),
        flush=True,
    )

    save_reports(output_root, records, failed_files, scanned_files=len(captures))
    print("[PROGRESS] stage=report status=done", flush=True)

    label_map_path = os.path.join(output_root, "mnist", "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_ID, f, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("预处理完成")
    print("dataset_root:", dataset_root)
    print("output_root :", output_root)
    print("total samples:", len(records))
    print(
        "train/test  :",
        sum(1 for r in records if r["split"] == "train"),
        "/",
        sum(1 for r in records if r["split"] == "test"),
    )
    print("reports dir :", os.path.join(output_root, "reports"))
    print("mnist dir   :", os.path.join(output_root, "mnist"))
    print("elapsed     : {:.2f}s".format(time.time() - start_time))
    print("=" * 88)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full preprocessing pipeline for ISCX dataset"
    )
    parser.add_argument(
        "--dataset-root",
        default=os.path.join("dataset", "iscx"),
        help="Raw ISCX dataset root",
    )
    parser.add_argument(
        "--output-root",
        default="processed_full",
        help="Output root for processed artifacts",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio, default=0.8",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=10000,
        help="Maximum extracted samples per class, default=10000",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output root before processing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.clean:
        clean_output(args.output_root)
    preprocess_full(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
