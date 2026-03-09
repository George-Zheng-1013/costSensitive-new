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


def iter_with_progress(iterable, total, desc: str, enable: bool):
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
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


def compute_per_class_metrics(conf_mat: np.ndarray, label_map: dict):
    out = []
    precisions = []
    recalls = []
    f1s = []

    for i in range(conf_mat.shape[0]):
        tp = float(conf_mat[i, i])
        fp = float(conf_mat[:, i].sum() - tp)
        fn = float(conf_mat[i, :].sum() - tp)
        support = int(conf_mat[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        out.append(
            {
                "class_id": i,
                "class_name": label_map.get(i, str(i)),
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }
    return out, macro


def main():
    args = parse_args()
    show_progress = not args.no_progress

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
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        infer_loop = iter_with_progress(
            loader,
            total=len(loader),
            desc="inference",
            enable=show_progress,
        )
        for batch in infer_loop:
            x = batch["bytes"].to(device)
            mask = batch["packet_mask"].to(device)
            y = batch["label"].to(device)
            flow_ids = batch["flow_id"]

            logits, emb = model(x, mask)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            total += y.size(0)
            correct += pred.eq(y).sum().item()

            y_np = y.detach().cpu().numpy().astype(np.int64)
            p_np = pred.detach().cpu().numpy().astype(np.int64)
            for yi, pi in zip(y_np, p_np):
                if 0 <= yi < num_classes and 0 <= pi < num_classes:
                    conf_mat[yi, pi] += 1

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

            if show_progress:
                set_postfix = getattr(infer_loop, "set_postfix", None)
                if callable(set_postfix):
                    set_postfix(
                        acc=f"{(100.0 * correct / max(total, 1)):.2f}%",
                        unknown=f"{unknown_count}",
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
        row_loop = iter_with_progress(
            all_rows,
            total=len(all_rows),
            desc="write-csv",
            enable=show_progress,
        )
        for row in row_loop:
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
    per_class, macro = compute_per_class_metrics(conf_mat, label_map)
    report = {
        "split": args.split,
        "num_samples": total,
        "num_classes": num_classes,
        "accuracy": acc,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
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
