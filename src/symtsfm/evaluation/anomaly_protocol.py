"""Explicit anomaly-evaluation protocol.

Best-F1 uses test labels, matching the common paper/TSB-UAD oracle convention.
It is saved and labelled as an oracle metric, not as a deployable threshold.
FPR is calculated from raw point labels at the selected unadjusted threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

ANOMALY_THRESHOLD_PROTOCOL = "oracle Best-F1 threshold selected on test scores, per entity; FPR uses raw point labels"
VUS_PROTOCOL = "per-entity min-max score normalization; vus.metrics.get_metrics(metric='vus', slidingWindow=512); equal entity mean"


def point_adjust(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mark a whole labelled event detected when any point in that event is hit."""
    output = np.asarray(prediction, dtype=bool).copy()
    truth = np.asarray(labels, dtype=bool)
    edges = np.diff(np.r_[0, truth.astype(np.int8), 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if output[start:end].any():
            output[start:end] = True
    return output


def _prf(prediction: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(labels, dtype=bool)
    tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return f1, precision, recall


def _raw_fpr(prediction: np.ndarray, labels: np.ndarray) -> float:
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(labels, dtype=bool)
    negatives = int((~truth).sum())
    return float((pred & ~truth).sum() / negatives) if negatives else float("nan")


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    """Exact unadjusted maximum F1 across score thresholds from sklearn's PR curve."""
    score, truth = np.asarray(scores, dtype=float), np.asarray(labels, dtype=np.int8)
    precision, recall, thresholds = precision_recall_curve(truth, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    best = int(np.nanargmax(f1))
    threshold = (float(thresholds[best]) if best < len(thresholds)
                 else float(np.nextafter(score.max(), np.inf)))
    prediction = score >= threshold
    return (*_prf(prediction, truth), threshold)


def adjusted_best_f1(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 512) -> float:
    """Range-adjusted oracle F1 on a fixed, documented score-quantile grid."""
    score, truth = np.asarray(scores, dtype=float), np.asarray(labels, dtype=np.int8)
    thresholds = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_thresholds, endpoint=True)))
    return max((_prf(point_adjust(score >= threshold, truth), truth)[0] for threshold in thresholds), default=0.0)


def vus_roc(scores: np.ndarray, labels: np.ndarray, max_buffer: int = 512) -> float:
    """VUS-ROC with a fixed reference-package protocol; fail rather than omit it."""
    try:
        from vus.metrics import get_metrics
    except ImportError as exc:
        raise RuntimeError("VUS-ROC requested but package vus is not installed") from exc
    score, truth = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if len(score) != len(truth) or len(score) < 2 or truth.min() == truth.max():
        return float("nan")
    minimum, maximum = score.min(), score.max()
    normalized = (score - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(score)
    result = get_metrics(normalized, truth, metric="vus", slidingWindow=min(int(max_buffer), len(score) - 1))
    if "VUS_ROC" not in result:
        raise RuntimeError(f"installed vus package returned no VUS_ROC key: {sorted(result)}")
    return float(result["VUS_ROC"])


def anomaly_metrics(scores_by_entity: dict, labels_by_entity: dict, vus_max_buffer: int = 512) -> dict:
    """Compute each entity independently, then take an unweighted entity mean."""
    rows = []
    for entity, scores in scores_by_entity.items():
        labels = np.asarray(labels_by_entity[entity], dtype=np.int8)
        if labels.min() == labels.max():
            continue
        f1, precision, recall, threshold = best_f1_threshold(scores, labels)
        raw_prediction = np.asarray(scores) >= threshold
        rows.append({"f1": f1, "precision": precision, "recall": recall,
                     "pr_auc": float(average_precision_score(labels, scores)),
                     "fpr": _raw_fpr(raw_prediction, labels),
                     "adjusted_best_f1": adjusted_best_f1(scores, labels),
                     "vus_roc": vus_roc(scores, labels, vus_max_buffer)})
    metric_names = ("f1", "precision", "recall", "pr_auc", "fpr", "adjusted_best_f1", "vus_roc")
    if not rows:
        return {name: float("nan") for name in metric_names} | {"n_entities_scored": 0}
    return {name: float(np.mean([row[name] for row in rows])) for name in metric_names} | {"n_entities_scored": len(rows)}
