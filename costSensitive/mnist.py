# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
import errno
import gzip
import shutil
import json
import csv
import random
import argparse
from PIL import Image
from array import array


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


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def gzip_file(src_path):
    gz_path = src_path + ".gz"
    with open(src_path, "rb") as fin, gzip.open(gz_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return gz_path


def collect_split_files(png_root, split_name):
    split_dir = os.path.join(png_root, split_name)
    if not os.path.isdir(split_dir):
        raise ValueError("missing split dir: {}".format(split_dir))

    records = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = CLASS_TO_ID[class_name]
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(".png"):
                records.append((os.path.join(class_dir, filename), label, class_name))
    return records


def write_idx(records, output_prefix):
    data_image = array("B")
    data_label = array("B")

    if len(records) == 0:
        raise ValueError(
            "No PNG files found for output prefix: {}".format(output_prefix)
        )

    width = 0
    height = 0
    for filepath, label, _ in records:
        im = Image.open(filepath).convert("L")
        pixel = im.load()
        width, height = im.size
        for x in range(0, width):
            for y in range(0, height):
                data_image.append(pixel[y, x])
        data_label.append(label)

    hexval = "{0:#0{1}x}".format(len(records), 6)
    hexval = "0x" + hexval[2:].zfill(8)

    header = array("B")
    header.extend([0, 0, 8, 1])
    header.append(int("0x" + hexval[2:][0:2], 16))
    header.append(int("0x" + hexval[2:][2:4], 16))
    header.append(int("0x" + hexval[2:][4:6], 16))
    header.append(int("0x" + hexval[2:][6:8], 16))
    data_label = header + data_label

    if max([width, height]) <= 256:
        header.extend([0, 0, 0, width, 0, 0, 0, height])
    else:
        raise ValueError("Image exceeds maximum size: 256x256 pixels")

    header[3] = 3
    data_image = header + data_image

    img_raw = output_prefix + "-images-idx3-ubyte"
    lbl_raw = output_prefix + "-labels-idx1-ubyte"

    with open(img_raw, "wb") as output_file:
        data_image.tofile(output_file)
    with open(lbl_raw, "wb") as output_file:
        data_label.tofile(output_file)

    img_gz = gzip_file(img_raw)
    lbl_gz = gzip_file(lbl_raw)
    print("[INFO] gzip done:", img_gz)
    print("[INFO] gzip done:", lbl_gz)


def build_idx_dataset(png_root, output_dir, report_dir=None, seed=42):
    random.seed(seed)
    mkdir_p(output_dir)
    if report_dir:
        mkdir_p(report_dir)

    split_to_prefix = {
        "train": os.path.join(output_dir, "train"),
        "test": os.path.join(output_dir, "t10k"),
    }

    split_records = {}
    for split_name in ["train", "test"]:
        records = collect_split_files(png_root, split_name)
        random.shuffle(records)
        split_records[split_name] = records
        print("[INFO] split={} total_png={}".format(split_name, len(records)))
        write_idx(records, split_to_prefix[split_name])

    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_ID, f, ensure_ascii=False, indent=2)
    print("[INFO] label map saved:", label_map_path)

    if report_dir:
        split_dist_path = os.path.join(report_dir, "idx_split_distribution.csv")
        with open(split_dist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "class_name", "class_id", "count"])
            for split_name, records in split_records.items():
                class_count = {name: 0 for name in CLASS_NAMES}
                for _, _, class_name in records:
                    class_count[class_name] += 1
                for class_name in CLASS_NAMES:
                    writer.writerow(
                        [
                            split_name,
                            class_name,
                            CLASS_TO_ID[class_name],
                            class_count[class_name],
                        ]
                    )
        print("[INFO] report saved:", split_dist_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MNIST IDX from split PNG directories"
    )
    parser.add_argument(
        "--png-root",
        default="processed_full/png",
        help="PNG root dir containing train/ and test/ subdirs",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_full/mnist",
        help="Output dir for IDX files",
    )
    parser.add_argument(
        "--report-dir",
        default="processed_full/reports",
        help="Output dir for build reports",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_idx_dataset(
        png_root=args.png_root,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        seed=args.seed,
    )
