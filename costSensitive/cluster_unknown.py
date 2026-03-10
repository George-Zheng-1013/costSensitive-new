import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster confirmed unknown embeddings")
    parser.add_argument(
        "--embeddings",
        default=os.path.join("pytorch_model", "session_embeddings.npy"),
    )
    parser.add_argument(
        "--pred-csv",
        default=os.path.join("pytorch_model", "session_predictions.csv"),
    )
    parser.add_argument(
        "--out-json",
        default=os.path.join("pytorch_model", "unknown_clusters.json"),
    )
    parser.add_argument(
        "--history-json",
        default=os.path.join("pytorch_model", "unknown_cluster_history.json"),
    )
    parser.add_argument(
        "--assignment-csv",
        default=os.path.join("pytorch_model", "unknown_cluster_assignments.csv"),
    )
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--spike-ratio", type=float, default=2.0)
    parser.add_argument("--spike-min-increase", type=int, default=8)
    parser.add_argument("--history-max", type=int, default=300)
    return parser.parse_args()


def _load_predictions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"prediction csv not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"flow_id", "unknown_level", "pred_name", "final_pred_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "prediction csv missing required columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            rows.append(row)
    return rows


def _l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _read_history(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return raw
        return []
    except Exception:
        return []


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    if not os.path.exists(args.embeddings):
        raise FileNotFoundError(f"embeddings not found: {args.embeddings}")

    embeddings = np.load(args.embeddings)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2 [N, D]")

    rows = _load_predictions(args.pred_csv)
    if embeddings.shape[0] != len(rows):
        raise ValueError(
            f"embedding length mismatch: embeddings={embeddings.shape[0]} rows={len(rows)}"
        )

    unknown_indices = []
    for i, row in enumerate(rows):
        if _safe_int(row.get("unknown_level"), 0) == 2:
            unknown_indices.append(i)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if len(unknown_indices) == 0:
        payload = {
            "generated_at": ts,
            "total_samples": int(len(rows)),
            "total_unknown": 0,
            "noise_count": 0,
            "config": {
                "eps": float(args.eps),
                "min_samples": int(args.min_samples),
                "metric": str(args.metric),
                "l2_normalize": bool(args.l2_normalize),
            },
            "clusters": [],
            "assignments": [],
        }
        _write_json(args.out_json, payload)
        print("[DONE] no confirmed unknown samples")
        return

    x = embeddings[np.asarray(unknown_indices, dtype=np.int64)]
    if args.l2_normalize:
        x = _l2_normalize(x)

    model = DBSCAN(
        eps=float(args.eps),
        min_samples=int(args.min_samples),
        metric=str(args.metric),
    )
    labels = model.fit_predict(x)

    non_noise = sorted(set(int(v) for v in labels.tolist() if int(v) >= 0))
    counts = {label: int(np.sum(labels == label)) for label in non_noise}
    sorted_labels = sorted(non_noise, key=lambda lb: counts[lb], reverse=True)
    label_to_name = {
        lb: f"unknown_cluster_{idx + 1}" for idx, lb in enumerate(sorted_labels)
    }

    clusters = []
    for lb in sorted_labels:
        idxs = np.where(labels == lb)[0]
        global_idxs = [unknown_indices[int(i)] for i in idxs.tolist()]
        center = embeddings[np.asarray(global_idxs, dtype=np.int64)].mean(axis=0)

        top_pred = {}
        for gi in global_idxs:
            name = str(rows[gi].get("pred_name") or "unknown")
            top_pred[name] = top_pred.get(name, 0) + 1
        top_pred_list = [
            {"pred_name": k, "count": int(v)}
            for k, v in sorted(top_pred.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        clusters.append(
            {
                "cluster_id": label_to_name[lb],
                "size": int(len(global_idxs)),
                "top_pred": top_pred_list,
                "center": [float(v) for v in center.astype(np.float32).tolist()],
            }
        )

    assignments = []
    for local_i, gi in enumerate(unknown_indices):
        label = int(labels[local_i])
        if label < 0:
            cluster_id = "unknown_cluster_noise"
        else:
            cluster_id = label_to_name[label]
        row = rows[gi]
        assignments.append(
            {
                "row_index": int(gi),
                "flow_id": str(row.get("flow_id") or ""),
                "pred_name": str(row.get("pred_name") or ""),
                "final_pred_name": str(row.get("final_pred_name") or ""),
                "anomaly_score": float(row.get("anomaly_score") or 0.0),
                "cluster_id": cluster_id,
            }
        )

    payload = {
        "generated_at": ts,
        "total_samples": int(len(rows)),
        "total_unknown": int(len(unknown_indices)),
        "noise_count": int(np.sum(labels == -1)),
        "config": {
            "eps": float(args.eps),
            "min_samples": int(args.min_samples),
            "metric": str(args.metric),
            "l2_normalize": bool(args.l2_normalize),
        },
        "clusters": clusters,
        "assignments": assignments,
    }
    _write_json(args.out_json, payload)

    os.makedirs(os.path.dirname(args.assignment_csv), exist_ok=True)
    with open(args.assignment_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_index",
                "flow_id",
                "pred_name",
                "final_pred_name",
                "anomaly_score",
                "cluster_id",
            ]
        )
        for row in assignments:
            writer.writerow(
                [
                    row["row_index"],
                    row["flow_id"],
                    row["pred_name"],
                    row["final_pred_name"],
                    f"{row['anomaly_score']:.6f}",
                    row["cluster_id"],
                ]
            )

    history = _read_history(args.history_json)
    prev_sizes = {}
    if len(history) > 0 and isinstance(history[-1], dict):
        prev_sizes = dict(history[-1].get("sizes") or {})

    sizes = {item["cluster_id"]: int(item["size"]) for item in clusters}
    spikes = []
    for cluster_id, cur_size in sizes.items():
        prev_size = int(prev_sizes.get(cluster_id, 0))
        growth = cur_size - prev_size
        ratio = (cur_size / max(prev_size, 1)) if cur_size > 0 else 1.0
        is_spike = bool(growth >= args.spike_min_increase and ratio >= args.spike_ratio)
        if is_spike:
            spikes.append(
                {
                    "cluster_id": cluster_id,
                    "prev_size": prev_size,
                    "cur_size": cur_size,
                    "growth": growth,
                    "growth_ratio": round(float(ratio), 4),
                }
            )

    history.append(
        {
            "timestamp": ts,
            "total_unknown": int(len(unknown_indices)),
            "sizes": sizes,
            "noise_count": int(np.sum(labels == -1)),
            "spikes": spikes,
        }
    )
    if len(history) > args.history_max:
        history = history[-args.history_max :]
    _write_json(args.history_json, history)

    print(
        f"[DONE] unknown={len(unknown_indices)} clusters={len(clusters)} noise={int(np.sum(labels == -1))}"
    )


if __name__ == "__main__":
    main()
