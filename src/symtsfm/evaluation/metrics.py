"""Task metrics.

classification  accuracy, macro-F1, macro precision, macro recall, macro PR-AUC
forecasting     MSE, MAE on the standardised scale (the ETT/Weather/Electricity convention)
anomaly         per entity, then averaged over entities with >= 1 anomaly:
                PR-AUC; F1 / precision / recall / false-positive rate at the best-F1 threshold;
                point-adjusted best F1; VUS-ROC with the configured maximum buffer.
                Best-threshold metrics use test labels and are therefore optimistic
                upper bounds; this is the MOMENT/TSB-UAD convention and is stated in
                every report under ``anomaly_threshold_protocol``.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support

ANOMALY_THRESHOLD_PROTOCOL = "best-F1 threshold over test scores, per entity (oracle upper bound)"
VUS_PROTOCOL = "VUS-ROC from the `vus` package; per entity with maximum buffer = 512 timestamps"


# =========================================================================== classification
def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    try:
        if prob.shape[1] == 2:
            pr_auc = average_precision_score(y_true, prob[:, 1])
        else:
            one_hot = np.eye(prob.shape[1], dtype=int)[y_true]
            pr_auc = average_precision_score(one_hot, prob, average="macro")
    except ValueError:
        pr_auc = float("nan")
    return {"accuracy": accuracy_score(y_true, pred), "precision": p, "recall": r,
            "macro_f1": f, "pr_auc": pr_auc}


# =========================================================================== forecasting
def forecasting_metrics(pred, true, *_unused, **_) -> dict:
    """Predictions and targets [n, C, H], evaluated on the standardised benchmark scale."""
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    return {"mse": float(((pred - true) ** 2).mean()), "mae": float(np.abs(pred - true).mean())}


# =========================================================================== anomaly
def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """If any point of a ground-truth anomaly segment is detected, the whole segment counts."""
    pred = pred.astype(bool).copy()
    lab = labels.astype(bool)
    edges = np.diff(np.concatenate([[0], lab.astype(int), [0]]))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if pred[s:e].any():
            pred[s:e] = True
    return pred


def _prf(pred, lab):
    tp = np.sum(pred & lab)
    fp = np.sum(pred & ~lab)
    fn = np.sum(~pred & lab)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def _fpr(pred, lab):
    """False-positive rate: false alarms among all truly normal timesteps."""
    negatives = np.sum(~lab)
    return float(np.sum(pred & ~lab) / negatives) if negatives else float("nan")


def best_f1(scores, labels, n_thresholds: int = 200, adjust: bool = False):
    """Best F1 over candidate thresholds (score quantiles). Returns (f1, precision, recall, thr)."""
    scores = np.asarray(scores, dtype=float)
    lab = np.asarray(labels).astype(bool)
    qs = np.unique(np.quantile(scores, np.linspace(0.5, 1.0, n_thresholds, endpoint=False)))
    best = (0.0, 0.0, 0.0, float(qs[-1]) if len(qs) else 0.0)
    for t in qs:
        pred = scores > t
        if adjust:
            pred = point_adjust(pred, lab)
        f, p, r = _prf(pred, lab)
        if f > best[0]:
            best = (f, p, r, float(t))
    return best


def vus_roc(scores, labels, max_buffer: int = 512) -> float:
    """Official range-aware VUS-ROC implementation, with a fixed reproducible buffer.

    The maintained ``vus`` package implements the VUS definition used by TSB-UAD.
    Scores are min-max scaled as prescribed by that package's reference example.
    Returning NaN only covers non-notebook environments where the optional package has
    not been installed; the Colab install cell installs it explicitly.
    """
    try:
        from vus.metrics import get_metrics
    except ImportError:
        return float("nan")
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) != len(labels) or len(scores) < 2 or labels.min() == labels.max():
        return float("nan")
    lo, hi = scores.min(), scores.max()
    scaled = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
    result = get_metrics(scaled, labels, metric="vus", slidingWindow=min(int(max_buffer), len(scores) - 1))
    value = result.get("VUS_ROC")
    return float(value) if value is not None else float("nan")


def anomaly_metrics(scores_by_entity: dict, labels_by_entity: dict, vus_max_buffer: int = 512) -> dict:
    rows = []
    for e, s in scores_by_entity.items():
        lab = np.asarray(labels_by_entity[e]).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue  # metrics undefined without both classes
        f, p, r, threshold = best_f1(s, lab)
        paf, _, _, _ = best_f1(s, lab, adjust=True)
        rows.append({"f1": f, "precision": p, "recall": r,
                     "pr_auc": average_precision_score(lab, s),
                     "fpr": _fpr(s > threshold, lab),
                     "adjusted_best_f1": paf,
                     "vus_roc": vus_roc(s, lab, vus_max_buffer)})
    if not rows:
        return {k: float("nan") for k in ("f1", "precision", "recall", "pr_auc", "fpr", "adjusted_best_f1", "vus_roc")} | {"n_entities_scored": 0}
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out["n_entities_scored"] = len(rows)
    return out


def assemble_scores(window_scores: np.ndarray, entity: np.ndarray, start: np.ndarray, entities: dict):
    """Per-window per-point scores [n, L] -> full-length per-entity score arrays."""
    L = window_scores.shape[1]
    out = {e: np.zeros(v["length"], dtype=np.float64) for e, v in entities.items()}
    for w, e, s in zip(window_scores, entity, start):
        if s < 0:  # left-padded short series: drop padding
            out[e][:] = w[-s : -s + len(out[e])]
        else:
            out[e][s : s + L] = w
    return out
