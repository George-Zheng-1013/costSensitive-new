import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DetectorDecision:
    pred_id: int
    centroid_distance: float
    centroid_threshold: float
    anomaly_score: float
    unknown_level: int
    unknown_state: str
    is_suspected: bool
    is_unknown: bool


class CentroidUnknownDetector:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.enabled = False
        self.l2_normalize = True
        self.quantile = 0.95
        self.suspected_score_threshold = 1.0
        self.confirmed_score_threshold = 1.5
        self.centroids: Dict[int, np.ndarray] = {}
        self.thresholds: Dict[int, float] = {}

        if path:
            self.load(path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"detector file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.l2_normalize = bool(raw.get("l2_normalize", True))
        self.quantile = float(raw.get("distance_quantile", 0.95))
        self.suspected_score_threshold = float(
            raw.get("suspected_score_threshold", 1.0)
        )
        self.confirmed_score_threshold = float(
            raw.get("confirmed_score_threshold", 1.5)
        )

        classes = raw.get("classes", {})
        self.centroids = {}
        self.thresholds = {}
        for key, item in classes.items():
            class_id = int(key)
            self.centroids[class_id] = np.asarray(item["centroid"], dtype=np.float32)
            self.thresholds[class_id] = float(item["threshold"])

        self.path = path
        self.enabled = len(self.centroids) > 0

    def _maybe_norm(self, vec: np.ndarray) -> np.ndarray:
        if not self.l2_normalize:
            return vec
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    def distance_to_centroid(
        self, embedding: np.ndarray, class_id: int
    ) -> Optional[float]:
        centroid = self.centroids.get(class_id)
        if centroid is None:
            return None

        emb = np.asarray(embedding, dtype=np.float32)
        emb = self._maybe_norm(emb)
        ctr = self._maybe_norm(centroid)
        return float(np.linalg.norm(emb - ctr))

    def decide(self, embedding: np.ndarray, pred_id: int) -> DetectorDecision:
        distance = self.distance_to_centroid(embedding, pred_id)
        threshold = float(self.thresholds.get(pred_id, float("inf")))

        if distance is None:
            return DetectorDecision(
                pred_id=pred_id,
                centroid_distance=float("nan"),
                centroid_threshold=threshold,
                anomaly_score=0.0,
                unknown_level=0,
                unknown_state="known",
                is_suspected=False,
                is_unknown=False,
            )

        denom = threshold if threshold > 1e-12 else 1.0
        score = float(distance / denom)
        if score >= self.confirmed_score_threshold:
            unknown_level = 2
            unknown_state = "confirmed_unknown"
        elif score >= self.suspected_score_threshold:
            unknown_level = 1
            unknown_state = "suspected"
        else:
            unknown_level = 0
            unknown_state = "known"

        is_unknown = bool(unknown_level == 2)
        return DetectorDecision(
            pred_id=pred_id,
            centroid_distance=float(distance),
            centroid_threshold=float(threshold),
            anomaly_score=score,
            unknown_level=unknown_level,
            unknown_state=unknown_state,
            is_suspected=bool(unknown_level == 1),
            is_unknown=is_unknown,
        )


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def build_centroid_detector_dict(
    embeddings: np.ndarray,
    labels: np.ndarray,
    distance_quantile: float = 0.95,
    l2_normalize: bool = True,
    suspected_score_threshold: float = 1.0,
    confirmed_score_threshold: float = 1.5,
) -> dict:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2 [N, D]")
    if labels.ndim != 1:
        raise ValueError("labels must be rank-1 [N]")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same length")
    if not (0.5 < distance_quantile < 1.0):
        raise ValueError("distance_quantile must be in (0.5, 1.0)")

    x = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if l2_normalize:
        x = _l2_normalize_rows(x)

    classes = {}
    for class_id in sorted(np.unique(y).tolist()):
        mask = y == class_id
        cls_x = x[mask]
        centroid = cls_x.mean(axis=0)

        if l2_normalize:
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
                centroid = centroid / centroid_norm

        distances = np.linalg.norm(cls_x - centroid[None, :], axis=1)
        threshold = float(np.quantile(distances, distance_quantile))

        classes[str(int(class_id))] = {
            "count": int(cls_x.shape[0]),
            "threshold": threshold,
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
            "centroid": centroid.astype(np.float32).tolist(),
        }

    return {
        "detector_type": "centroid_distance",
        "distance_metric": "l2",
        "distance_quantile": float(distance_quantile),
        "l2_normalize": bool(l2_normalize),
        "suspected_score_threshold": float(suspected_score_threshold),
        "confirmed_score_threshold": float(confirmed_score_threshold),
        "embedding_dim": int(x.shape[1]),
        "num_classes": int(len(classes)),
        "classes": classes,
    }
