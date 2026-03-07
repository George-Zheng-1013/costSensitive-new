import argparse
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from realtime.model_def import ByteSessionClassifier
from realtime.unknown_detector import build_centroid_detector_dict
from session_data import ByteSessionDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build centroid unknown detector from train embeddings"
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join("processed_full", "sessions", "manifest.csv"),
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join("pytorch_model", "byte_session_classifier.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--distance-quantile", type=float, default=0.95)
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        default=True,
        help="l2-normalize embeddings before centroid/distance computation",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_false",
        dest="l2_normalize",
        help="disable l2-normalization",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join("pytorch_model", "centroid_detector.json"),
    )
    return parser.parse_args()


def get_device(device_arg: str):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model not found: {args.model_path}")

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ds = ByteSessionDataset(args.manifest, split="train")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    max_label = max(int(r["label"]) for r in ds.rows)
    num_classes = max_label + 1

    device = get_device(args.device)
    model = ByteSessionClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["bytes"].to(device)
            mask = batch["packet_mask"].to(device)
            y = batch["label"].cpu().numpy().astype(np.int64)

            _, emb = model(x, mask)
            all_embeddings.append(emb.cpu().numpy().astype(np.float32))
            all_labels.append(y)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    detector = build_centroid_detector_dict(
        embeddings=embeddings,
        labels=labels,
        distance_quantile=args.distance_quantile,
        l2_normalize=args.l2_normalize,
    )

    detector["manifest"] = args.manifest
    detector["model_path"] = args.model_path
    detector["num_samples"] = int(embeddings.shape[0])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(detector, f, ensure_ascii=False, indent=2)

    print(
        "[DONE] detector={} num_classes={} num_samples={} quantile={:.3f} l2_normalize={}".format(
            args.output_json,
            detector["num_classes"],
            detector["num_samples"],
            args.distance_quantile,
            args.l2_normalize,
        )
    )


if __name__ == "__main__":
    main()
