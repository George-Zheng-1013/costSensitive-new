import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from realtime.model_def import ByteSessionClassifier
from session_data import ByteSessionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Byte-session offline inference")
    parser.add_argument(
        "--manifest",
        default=os.path.join("processed_full", "sessions", "manifest.csv"),
    )
    parser.add_argument(
        "--label-map",
        default=os.path.join("processed_full", "sessions", "label_map.json"),
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join("pytorch_model", "byte_session_classifier.pth"),
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-csv",
        default=os.path.join("pytorch_model", "session_predictions.csv"),
    )
    parser.add_argument(
        "--embedding-npy",
        default=os.path.join("pytorch_model", "session_embeddings.npy"),
    )
    parser.add_argument(
        "--report-json",
        default=os.path.join("pytorch_model", "session_eval_report.json"),
    )
    return parser.parse_args()


def get_device(device_arg: str):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_map(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(v): k for k, v in raw.items()}


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model not found: {args.model_path}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.embedding_npy), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)

    ds = ByteSessionDataset(args.manifest, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    max_label = max(int(r["label"]) for r in ds.rows)
    num_classes = max_label + 1

    device = get_device(args.device)
    model = ByteSessionClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    label_map = load_label_map(args.label_map)

    all_embeddings = []
    all_rows = []
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["bytes"].to(device)
            mask = batch["packet_mask"].to(device)
            y = batch["label"].to(device)
            flow_ids = batch["flow_id"]

            logits, emb = model(x, mask)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            total += y.size(0)
            correct += pred.eq(y).sum().item()

            all_embeddings.append(emb.cpu().numpy())
            for i in range(y.size(0)):
                label_id = int(y[i].item())
                pred_id = int(pred[i].item())
                all_rows.append(
                    {
                        "flow_id": str(flow_ids[i]),
                        "label": label_id,
                        "label_name": label_map.get(label_id, str(label_id)),
                        "pred": pred_id,
                        "pred_name": label_map.get(pred_id, str(pred_id)),
                        "confidence": float(conf[i].item()),
                    }
                )

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(args.embedding_npy, embeddings)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["flow_id", "label", "label_name", "pred", "pred_name", "confidence"]
        )
        for row in all_rows:
            writer.writerow(
                [
                    row["flow_id"],
                    row["label"],
                    row["label_name"],
                    row["pred"],
                    row["pred_name"],
                    f"{row['confidence']:.6f}",
                ]
            )

    acc = correct / total if total > 0 else 0.0
    report = {
        "split": args.split,
        "num_samples": total,
        "accuracy": acc,
        "embedding_shape": list(embeddings.shape),
        "output_csv": args.output_csv,
        "embedding_npy": args.embedding_npy,
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] split={args.split} samples={total} accuracy={acc:.4f}")
    print(f"predictions={args.output_csv}")
    print(f"embeddings={args.embedding_npy}")


if __name__ == "__main__":
    main()
