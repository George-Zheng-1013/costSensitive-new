import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from realtime.model_def import ByteSessionClassifier
from realtime.unknown_detector import CentroidUnknownDetector
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
    parser.add_argument(
        "--detector-json",
        default=os.path.join("pytorch_model", "centroid_detector.json"),
    )
    parser.add_argument("--unknown-label", default="unknown_proxy_ood")
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
    detector = CentroidUnknownDetector()
    if args.detector_json and os.path.exists(args.detector_json):
        detector.load(args.detector_json)

    all_embeddings = []
    all_rows = []
    total = 0
    correct = 0
    unknown_count = 0

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
                emb_i = emb[i].detach().cpu().numpy()
                decision = detector.decide(emb_i, pred_id) if detector.enabled else None

                is_unknown = (
                    bool(decision.is_unknown) if decision is not None else False
                )
                if is_unknown:
                    unknown_count += 1

                final_pred = -1 if is_unknown else pred_id
                final_pred_name = (
                    args.unknown_label
                    if is_unknown
                    else label_map.get(pred_id, str(pred_id))
                )

                all_rows.append(
                    {
                        "flow_id": str(flow_ids[i]),
                        "label": label_id,
                        "label_name": label_map.get(label_id, str(label_id)),
                        "pred": pred_id,
                        "pred_name": label_map.get(pred_id, str(pred_id)),
                        "confidence": float(conf[i].item()),
                        "centroid_distance": (
                            float(decision.centroid_distance)
                            if decision is not None
                            else float("nan")
                        ),
                        "centroid_threshold": (
                            float(decision.centroid_threshold)
                            if decision is not None
                            else float("nan")
                        ),
                        "anomaly_score": (
                            float(decision.anomaly_score)
                            if decision is not None
                            else 0.0
                        ),
                        "is_unknown": int(is_unknown),
                        "final_pred": final_pred,
                        "final_pred_name": final_pred_name,
                    }
                )

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(args.embedding_npy, embeddings)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "flow_id",
                "label",
                "label_name",
                "pred",
                "pred_name",
                "confidence",
                "centroid_distance",
                "centroid_threshold",
                "anomaly_score",
                "is_unknown",
                "final_pred",
                "final_pred_name",
            ]
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
                    f"{row['centroid_distance']:.6f}",
                    f"{row['centroid_threshold']:.6f}",
                    f"{row['anomaly_score']:.6f}",
                    row["is_unknown"],
                    row["final_pred"],
                    row["final_pred_name"],
                ]
            )

    acc = correct / total if total > 0 else 0.0
    report = {
        "split": args.split,
        "num_samples": total,
        "accuracy": acc,
        "unknown_count": int(unknown_count),
        "unknown_rate": (unknown_count / total if total > 0 else 0.0),
        "detector_enabled": bool(detector.enabled),
        "detector_json": args.detector_json,
        "unknown_label": args.unknown_label,
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
