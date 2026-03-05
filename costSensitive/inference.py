import argparse
import csv
import gzip
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from realtime.model_def import ConvNet


@dataclass
class InferResult:
    labels: np.ndarray
    preds: np.ndarray
    confidence: np.ndarray
    features: np.ndarray


class IdxDataset(Dataset):
    def __init__(self, data_dir: str, image_stem: str, label_stem: str):
        image_path = self._resolve_path(data_dir, image_stem)
        label_path = self._resolve_path(data_dir, label_stem)
        self.images = self._load_images(image_path)
        self.labels = self._load_labels(label_path)

        if len(self.images) != len(self.labels):
            raise ValueError("images 与 labels 数量不一致")

    @staticmethod
    def _resolve_path(data_dir: str, stem: str) -> str:
        candidates = [
            os.path.join(data_dir, stem),
            os.path.join(data_dir, stem + ".gz"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"找不到 IDX 文件: {stem} 或 {stem}.gz")

    @staticmethod
    def _open(path: str):
        if path.endswith(".gz"):
            return gzip.open(path, "rb")
        return open(path, "rb")

    @classmethod
    def _load_images(cls, path: str) -> np.ndarray:
        with cls._open(path) as f:
            raw = f.read()
        images = np.frombuffer(raw, dtype=np.uint8, offset=16)
        images = images.reshape(-1, 28, 28)
        return images.copy()

    @classmethod
    def _load_labels(cls, path: str) -> np.ndarray:
        with cls._open(path) as f:
            raw = f.read()
        labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
        return labels.copy()

    def __getitem__(self, index: int):
        img = self.images[index].astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        label = int(self.labels[index])
        return torch.from_numpy(img), label

    def __len__(self) -> int:
        return len(self.labels)


def parse_args():
    parser = argparse.ArgumentParser(description="离线推理 + MSP + PyOD复核")
    parser.add_argument("--data-dir", default=os.path.join("processed_full", "mnist"))
    parser.add_argument(
        "--model-path", default=os.path.join("pytorch_model", "convnet.pth")
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join("pytorch_model", "predictions_with_unknown.csv"),
    )
    parser.add_argument(
        "--report-json",
        default=os.path.join("pytorch_model", "unknown_eval_report.json"),
    )
    parser.add_argument(
        "--detector-path",
        default=os.path.join("pytorch_model", "unknown_detector_best.pkl"),
    )
    parser.add_argument(
        "--detectors",
        default="iforest,ocsvm,copod",
        help="逗号分隔的PyOD检测器列表，可选: iforest,ocsvm,copod",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--max-train-detector-samples",
        type=int,
        default=0,
        help="检测器训练样本上限，0表示使用全部",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="验证样本上限，0表示使用全部",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="测试样本上限，0表示使用全部",
    )
    parser.add_argument("--ood-mode", choices=["proxy", "none"], default="proxy")
    parser.add_argument("--device", default=None, help="cuda/cpu，默认自动")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条显示",
    )
    return parser.parse_args()


def iter_with_progress(
    iterable: Iterable,
    total: Optional[int],
    desc: str,
    enable: bool,
):
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(
    dataset: Dataset, batch_size: int, shuffle: bool = False
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def extract_outputs(
    model: ConvNet,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    show_progress: bool,
) -> InferResult:
    labels_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    conf_all: List[np.ndarray] = []
    feat_all: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        loop = iter_with_progress(
            loader,
            total=len(loader),
            desc=desc,
            enable=show_progress,
        )
        for data, target in loop:
            data = data.to(device)
            target = target.to(device)

            logits, features = model(data, return_features=True)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            labels_all.append(target.cpu().numpy())
            preds_all.append(pred.cpu().numpy())
            conf_all.append(conf.cpu().numpy())
            feat_all.append(features.cpu().numpy())

    return InferResult(
        labels=np.concatenate(labels_all, axis=0),
        preds=np.concatenate(preds_all, axis=0),
        confidence=np.concatenate(conf_all, axis=0),
        features=np.concatenate(feat_all, axis=0),
    )


def build_proxy_ood_images(images: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = images.shape[0]
    flat = images.reshape(n, -1)

    perm = rng.permutation(flat.shape[1])
    shuffled = flat[:, perm]
    noise = rng.integers(0, 256, size=flat.shape, dtype=np.uint8)
    selector = rng.integers(0, 2, size=(n, 1), dtype=np.uint8)
    mixed = np.where(selector == 1, shuffled, noise)
    return mixed.reshape(n, 28, 28)


class ArrayImageDataset(Dataset):
    def __init__(self, images: np.ndarray, label: int = -1):
        self.images = images
        self.label = label

    def __getitem__(self, index: int):
        img = self.images[index].astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), self.label

    def __len__(self):
        return self.images.shape[0]


def fit_pyod_detectors(
    detector_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    contamination: float,
    random_seed: int,
    show_progress: bool,
):
    detector_name = detector_name.lower().strip()
    build_detector: Any
    if detector_name == "iforest":
        from pyod.models.iforest import IForest

        build_detector = lambda: IForest(
            contamination=contamination, random_state=random_seed
        )

    elif detector_name == "ocsvm":
        from pyod.models.ocsvm import OCSVM

        build_detector = lambda: OCSVM(
            contamination=contamination, kernel="rbf", gamma="scale"
        )

    elif detector_name == "copod":
        from pyod.models.copod import COPOD

        build_detector = lambda: COPOD(contamination=contamination)

    else:
        raise ValueError(f"不支持的检测器: {detector_name}")

    per_class: Dict[int, object] = {}
    unique_classes = sorted(np.unique(train_labels).tolist())
    class_loop = iter_with_progress(
        unique_classes,
        total=len(unique_classes),
        desc=f"fit-{detector_name}-per-class",
        enable=show_progress,
    )
    for class_id in class_loop:
        cls_feat = train_features[train_labels == class_id]
        if cls_feat.shape[0] < 10:
            continue
        detector = build_detector()
        detector.fit(cls_feat)
        per_class[int(class_id)] = detector

    global_detector = build_detector()
    global_detector.fit(train_features)

    return per_class, global_detector


def parse_detectors(raw: str) -> List[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if len(parts) == 0:
        return ["iforest"]
    return parts


def maybe_limit_subset(
    ds: Dataset,
    max_samples: int,
    seed: int,
    stratify_labels: Optional[np.ndarray] = None,
) -> Dataset:
    ds_any: Any = ds
    ds_size = len(ds_any)
    if max_samples <= 0 or ds_size <= max_samples:
        return ds

    rng = np.random.default_rng(seed)
    indices = np.arange(ds_size)
    if stratify_labels is None or len(np.unique(stratify_labels)) < 2:
        selected = rng.choice(indices, size=max_samples, replace=False)
        return Subset(ds, selected.tolist())

    selected_all = []
    for label in np.unique(stratify_labels):
        label_idx = indices[stratify_labels == label]
        take = max(1, int(round(max_samples * (len(label_idx) / len(indices)))))
        take = min(take, len(label_idx))
        selected_all.extend(rng.choice(label_idx, size=take, replace=False).tolist())

    if len(selected_all) > max_samples:
        selected_all = rng.choice(
            np.array(selected_all), size=max_samples, replace=False
        ).tolist()

    return Subset(ds, selected_all)


def score_anomaly(
    per_class_detectors,
    global_detector,
    features: np.ndarray,
    preds: np.ndarray,
    desc: str,
    show_progress: bool,
) -> np.ndarray:
    scores = np.zeros(features.shape[0], dtype=np.float32)
    loop = iter_with_progress(
        range(features.shape[0]),
        total=features.shape[0],
        desc=desc,
        enable=show_progress,
    )
    for i in loop:
        pred = int(preds[i])
        detector = per_class_detectors.get(pred, global_detector)
        scores[i] = float(detector.decision_function(features[i : i + 1])[0])
    return scores


def fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(np.min(fpr[idx]))


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.detector_path), exist_ok=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    device = get_device(args.device)
    print(f"[INFO] device={device}")

    train_ds = IdxDataset(
        args.data_dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte"
    )
    test_ds = IdxDataset(
        args.data_dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    )

    train_indices = np.arange(len(train_ds))
    train_labels_np = train_ds.labels.copy()
    pyod_train_idx, val_idx = train_test_split(
        train_indices,
        test_size=args.val_ratio,
        random_state=args.random_seed,
        stratify=train_labels_np,
    )

    pyod_train_ds = Subset(train_ds, pyod_train_idx.tolist())
    val_known_ds = Subset(train_ds, val_idx.tolist())

    pyod_train_ds = maybe_limit_subset(
        pyod_train_ds,
        args.max_train_detector_samples,
        seed=args.random_seed,
        stratify_labels=train_ds.labels[pyod_train_idx],
    )
    val_known_ds = maybe_limit_subset(
        val_known_ds,
        args.max_val_samples,
        seed=args.random_seed + 1,
        stratify_labels=train_ds.labels[val_idx],
    )
    test_ds = maybe_limit_subset(
        test_ds,
        args.max_test_samples,
        seed=args.random_seed + 2,
        stratify_labels=test_ds.labels,
    )

    pyod_train_loader = build_loader(pyod_train_ds, args.batch_size)
    val_known_loader = build_loader(val_known_ds, args.batch_size)
    test_loader = build_loader(test_ds, args.batch_size)

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    show_progress = not args.no_progress

    print("[INFO] 提取训练/验证/测试特征...")
    train_out = extract_outputs(
        model,
        pyod_train_loader,
        device,
        desc="extract-train",
        show_progress=show_progress,
    )
    val_out = extract_outputs(
        model,
        val_known_loader,
        device,
        desc="extract-val",
        show_progress=show_progress,
    )
    test_out = extract_outputs(
        model,
        test_loader,
        device,
        desc="extract-test",
        show_progress=show_progress,
    )
    tau1 = float(np.quantile(val_out.confidence, args.target_fpr))

    proxy_out = None
    if args.ood_mode == "proxy":
        val_known_indices = np.array(val_idx)
        val_known_images = train_ds.images[val_known_indices]
        proxy_images = build_proxy_ood_images(val_known_images, seed=args.random_seed)
        proxy_ds = ArrayImageDataset(proxy_images, label=-1)
        proxy_loader = build_loader(proxy_ds, args.batch_size)
        proxy_out = extract_outputs(
            model,
            proxy_loader,
            device,
            desc="extract-proxy-ood",
            show_progress=show_progress,
        )

    detector_names = parse_detectors(args.detectors)
    detector_reports = []
    best_state = None

    detector_loop = iter_with_progress(
        detector_names,
        total=len(detector_names),
        desc="detector-benchmark",
        enable=show_progress,
    )
    for detector_name in detector_loop:
        print(f"[INFO] 训练检测器: {detector_name}")
        try:
            per_class_detectors, global_detector = fit_pyod_detectors(
                detector_name=detector_name,
                train_features=train_out.features,
                train_labels=train_out.labels,
                contamination=args.contamination,
                random_seed=args.random_seed,
                show_progress=show_progress,
            )
        except Exception as exc:
            detector_reports.append(
                {"detector": detector_name, "status": "failed", "error": str(exc)}
            )
            continue

        val_anom = score_anomaly(
            per_class_detectors,
            global_detector,
            val_out.features,
            val_out.preds,
            desc=f"score-val-{detector_name}",
            show_progress=show_progress,
        )
        tau2 = float(np.quantile(val_anom, 1.0 - args.target_fpr))

        test_anom = score_anomaly(
            per_class_detectors,
            global_detector,
            test_out.features,
            test_out.preds,
            desc=f"score-test-{detector_name}",
            show_progress=show_progress,
        )
        test_unknown = (test_out.confidence < tau1) | (test_anom > tau2)

        report = {
            "detector": detector_name,
            "status": "ok",
            "thresholds": {"tau1_confidence": tau1, "tau2_anomaly": tau2},
            "validation": {
                "known_count": int(val_out.labels.shape[0]),
                "known_fpr_conf": float((val_out.confidence < tau1).mean()),
                "known_fpr_anomaly": float((val_anom > tau2).mean()),
                "known_fpr_combined": float(
                    ((val_out.confidence < tau1) | (val_anom > tau2)).mean()
                ),
            },
            "test_unknown_rate": float(test_unknown.mean()),
        }

        ranking_score = -report["validation"]["known_fpr_combined"]
        if proxy_out is not None:
            proxy_anom = score_anomaly(
                per_class_detectors,
                global_detector,
                proxy_out.features,
                proxy_out.preds,
                desc=f"score-proxy-{detector_name}",
                show_progress=show_progress,
            )
            y_true = np.concatenate(
                [
                    np.zeros(val_out.confidence.shape[0], dtype=np.int64),
                    np.ones(proxy_out.confidence.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            score_anom = np.concatenate([val_anom, proxy_anom], axis=0)
            rule_pred_known = (val_out.confidence < tau1) | (val_anom > tau2)
            rule_pred_proxy = (proxy_out.confidence < tau1) | (proxy_anom > tau2)
            report["ood_eval"] = {
                "ood_type": "proxy",
                "known_count": int(val_out.confidence.shape[0]),
                "ood_count": int(proxy_out.confidence.shape[0]),
                "auroc_pyod": safe_auroc(y_true, score_anom),
                "fpr95_pyod": fpr95(y_true, score_anom),
                "combined_rule_fpr": float(rule_pred_known.mean()),
                "combined_rule_tpr": float(rule_pred_proxy.mean()),
            }
            ranking_score = (
                report["ood_eval"]["combined_rule_tpr"]
                - report["ood_eval"]["combined_rule_fpr"]
            )

        report["ranking_score"] = float(ranking_score)
        detector_reports.append(report)

        if best_state is None or ranking_score > best_state["ranking_score"]:
            best_state = {
                "ranking_score": float(ranking_score),
                "detector": detector_name,
                "per_class_detectors": per_class_detectors,
                "global_detector": global_detector,
                "tau2": tau2,
                "test_anom": test_anom,
                "test_unknown": test_unknown,
                "report": report,
            }

    if best_state is None:
        raise RuntimeError("所有检测器训练均失败，请检查 pyod 依赖或参数设置")

    print(
        f"[BEST] detector={best_state['detector']} score={best_state['ranking_score']:.6f} tau1={tau1:.6f} tau2={best_state['tau2']:.6f}"
    )

    test_anom = best_state["test_anom"]
    test_unknown = best_state["test_unknown"]
    tau2_best = best_state["tau2"]

    total = int(test_out.labels.shape[0])
    cls_correct = int((test_out.preds == test_out.labels).sum())
    cls_acc = cls_correct / total if total > 0 else 0.0
    unknown_rate = float(test_unknown.mean()) if total > 0 else 0.0

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "pred",
                "label",
                "confidence",
                "anomaly_score",
                "is_unknown",
                "unknown_reason",
            ]
        )
        for i in range(total):
            low_conf = bool(test_out.confidence[i] < tau1)
            high_anom = bool(test_anom[i] > tau2_best)
            if low_conf and high_anom:
                reason = "low_confidence_and_high_anomaly"
            elif low_conf:
                reason = "low_confidence"
            elif high_anom:
                reason = "high_anomaly"
            else:
                reason = "known"

            writer.writerow(
                [
                    i,
                    int(test_out.preds[i]),
                    int(test_out.labels[i]),
                    f"{float(test_out.confidence[i]):.6f}",
                    f"{float(test_anom[i]):.6f}",
                    int(test_unknown[i]),
                    reason,
                ]
            )

    metrics = {
        "classification_accuracy": cls_acc,
        "test_total": total,
        "test_unknown_rate": unknown_rate,
        "selected_detector": best_state["detector"],
        "detector_candidates": detector_reports,
        "thresholds": {
            "tau1_confidence": tau1,
            "tau2_anomaly": best_state["tau2"],
            "target_fpr": args.target_fpr,
        },
        "validation": best_state["report"]["validation"],
    }

    if args.ood_mode == "proxy" and proxy_out is not None:
        val_anom_best = score_anomaly(
            best_state["per_class_detectors"],
            best_state["global_detector"],
            val_out.features,
            val_out.preds,
            desc="score-val-best",
            show_progress=show_progress,
        )
        proxy_anom = score_anomaly(
            best_state["per_class_detectors"],
            best_state["global_detector"],
            proxy_out.features,
            proxy_out.preds,
            desc="score-proxy-best",
            show_progress=show_progress,
        )
        y_true = np.concatenate(
            [
                np.zeros(val_out.confidence.shape[0], dtype=np.int64),
                np.ones(proxy_out.confidence.shape[0], dtype=np.int64),
            ],
            axis=0,
        )
        score_conf = np.concatenate(
            [-val_out.confidence, -proxy_out.confidence], axis=0
        )
        score_anom = np.concatenate([val_anom_best, proxy_anom], axis=0)
        rule_pred_known = (val_out.confidence < tau1) | (
            val_anom_best > best_state["tau2"]
        )
        rule_pred_proxy = (proxy_out.confidence < tau1) | (
            proxy_anom > best_state["tau2"]
        )

        metrics["ood_eval"] = {
            "ood_type": "proxy",
            "selected_detector": best_state["detector"],
            "known_count": int(val_out.confidence.shape[0]),
            "ood_count": int(proxy_out.confidence.shape[0]),
            "auroc_msp": safe_auroc(y_true, score_conf),
            "fpr95_msp": fpr95(y_true, score_conf),
            "auroc_pyod": safe_auroc(y_true, score_anom),
            "fpr95_pyod": fpr95(y_true, score_anom),
            "combined_rule_fpr": float(rule_pred_known.mean()),
            "combined_rule_tpr": float(rule_pred_proxy.mean()),
        }

    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    try:
        import joblib

        payload = {
            "detector_name": best_state["detector"],
            "per_class_detectors": best_state["per_class_detectors"],
            "global_detector": best_state["global_detector"],
            "thresholds": {"tau1": tau1, "tau2": best_state["tau2"]},
            "meta": {
                "contamination": args.contamination,
                "target_fpr": args.target_fpr,
                "ood_mode": args.ood_mode,
                "detectors": detector_names,
            },
        }
        joblib.dump(payload, args.detector_path)
        print(f"[SAVE] detector={args.detector_path}")
    except Exception as exc:
        print(f"[WARN] detector 保存失败: {exc}")

    print("=" * 88)
    print(f"[DONE] cls_acc={cls_acc:.6f} ({cls_correct}/{total})")
    print(f"[DONE] unknown_rate={unknown_rate:.6f}")
    print(f"[DONE] output_csv={args.output_csv}")
    print(f"[DONE] report_json={args.report_json}")
    print("=" * 88)


if __name__ == "__main__":
    main()
