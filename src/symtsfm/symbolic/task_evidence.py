"""Task-conditioned symbolic evidence policies.

The module deliberately does *not* inject one generic symbolic vector into
every UniTS head.  It uses the same train-only FastShapelets/BOSS-ST event
features in three different, auditable policies:

* classification: class-conditional symbolic evidence;
* forecasting: symbolic-context to historical-continuation retrieval;
* anomaly detection: distance from the normal symbolic event memory.

Every policy is selected on validation data and includes an exact foundation
model fallback.  This is a pilot implementation intended to test the research
hypothesis, not to guarantee an improvement on every data set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _flat(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected symbolic features [n, channels, features], got {x.shape}")
    return x.reshape(len(x), -1)


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def _as_logits(prob: np.ndarray) -> np.ndarray:
    """Log probabilities are valid logits and preserve the selected mixture."""
    return np.log(np.clip(np.asarray(prob, dtype=np.float32), 1e-8, 1.0))


@dataclass
class EvidenceResult:
    output: np.ndarray
    selected_weight: float
    summary: dict
    preview: dict


class ClassificationEvidence:
    """Sparse class-conditional symbolic evidence with validation abstention."""

    def __init__(self, c: float = 0.1, weights=(0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0):
        self.c, self.weights, self.seed = float(c), tuple(float(w) for w in weights), int(seed)

    def fit_apply(self, train_features, train_labels, val_features, val_labels,
                  val_base_logits, test_features, test_base_logits) -> EvidenceResult:
        xtr, xva, xte = _flat(train_features), _flat(val_features), _flat(test_features)
        ytr, yva = np.asarray(train_labels), np.asarray(val_labels)
        base_val = _softmax(val_base_logits)
        base_acc = float(accuracy_score(yva, base_val.argmax(1)))
        # The classifier contains no foundation-model state; it is an explicit
        # symbolic evidence source.  Its contribution is calibrated separately.
        clf = LogisticRegression(C=self.c, max_iter=2000, class_weight="balanced", random_state=self.seed)
        clf.fit(xtr, ytr)
        sym_val, sym_test = clf.predict_proba(xva), clf.predict_proba(xte)
        # Align probabilities when a small training split omits a class.
        n_classes = base_val.shape[1]
        def aligned(p):
            out = np.full((len(p), n_classes), 1e-8, dtype=np.float32)
            out[:, clf.classes_.astype(int)] = p
            return out / out.sum(1, keepdims=True)
        sym_val, sym_test = aligned(sym_val), aligned(sym_test)
        candidates, chosen = [], 0.0
        best = (base_acc, 0.0)
        for w in sorted(set(self.weights + (0.0,))):
            p = (1.0 - w) * base_val + w * sym_val
            acc = float(accuracy_score(yva, p.argmax(1)))
            candidates.append({"weight": w, "validation_accuracy": acc,
                               "noninferior": bool(acc + 1e-12 >= base_acc)})
            if acc + 1e-12 >= base_acc and (acc, -w) > best:
                best, chosen = (acc, -w), w
        base_test = _softmax(test_base_logits)
        output = _as_logits((1.0 - chosen) * base_test + chosen * sym_test)
        coeff = np.abs(clf.coef_).mean(0)
        top = np.argsort(-coeff)[: min(20, len(coeff))]
        return EvidenceResult(output, chosen, {
            "policy": "class_conditional_symbolic_evidence",
            "selection_rule": "max_validation_accuracy_subject_to_noninferiority",
            "validation_baseline_accuracy": base_acc,
            "weight_candidates": candidates,
            "selected_weight": chosen,
        }, {"top_global_feature_indices": top.tolist(), "top_global_feature_weights": coeff[top].tolist()})


class ForecastEvidence:
    """Symbolic-context kNN continuation evidence with MAE-safe validation selection.

    This is not generic latent RAG.  Neighbourhoods are formed from the ordered
    symbolic event descriptor supplied by the event miner.  The known futures
    of causal training windows supply the continuation evidence.
    """

    def __init__(self, n_neighbors: int = 8, weights=(0.0, 0.1, 0.25, 0.5, 1.0),
                 mae_tolerance: float = 0.0, chunk_size: int = 256):
        self.k, self.weights = int(n_neighbors), tuple(float(w) for w in weights)
        self.mae_tolerance, self.chunk_size = float(mae_tolerance), int(chunk_size)

    def _fit_bank(self, features, futures):
        x = _flat(features)
        self.mean_ = x.mean(0, keepdims=True)
        self.scale_ = np.maximum(x.std(0, keepdims=True), 1e-5)
        self.bank_ = _l2((x - self.mean_) / self.scale_)
        self.future_ = np.asarray(futures, dtype=np.float32)

    def retrieve(self, features):
        q = _l2((_flat(features) - self.mean_) / self.scale_)
        k = min(self.k, len(self.bank_))
        all_ix, all_score, all_weight, all_future = [], [], [], []
        for start in range(0, len(q), self.chunk_size):
            score = q[start:start + self.chunk_size] @ self.bank_.T
            ix = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
            sc = np.take_along_axis(score, ix, axis=1)
            order = np.argsort(-sc, axis=1)
            ix, sc = np.take_along_axis(ix, order, axis=1), np.take_along_axis(sc, order, axis=1)
            w = np.exp(sc - sc.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            all_ix.append(ix); all_score.append(sc); all_weight.append(w)
            all_future.append((self.future_[ix] * w[:, :, None, None]).sum(1))
        return (np.concatenate(all_future), np.concatenate(all_ix), np.concatenate(all_score),
                np.concatenate(all_weight))

    def fit_apply(self, train_features, train_future, val_features, val_base, val_target,
                  test_features, test_base) -> EvidenceResult:
        self._fit_bank(train_features, train_future)
        val_ref, _, _, _ = self.retrieve(val_features)
        base_mse = float(np.mean((val_base - val_target) ** 2))
        base_mae = float(np.mean(np.abs(val_base - val_target)))
        candidates, selected = [], 0.0
        best = (base_mse, 0.0)
        for w in sorted(set(self.weights + (0.0,))):
            pred = val_base + w * (val_ref - val_base)
            mse, mae = float(np.mean((pred - val_target) ** 2)), float(np.mean(np.abs(pred - val_target)))
            ok = mae <= base_mae + self.mae_tolerance + 1e-12
            candidates.append({"weight": w, "validation_mse": mse, "validation_mae": mae,
                               "mae_noninferior": bool(ok)})
            if ok and (mse, w) < best:
                best, selected = (mse, w), w
        test_ref, ix, score, weight = self.retrieve(test_features)
        out = test_base + selected * (test_ref - test_base)
        preview = min(64, len(ix))
        return EvidenceResult(out, selected, {
            "policy": "symbolic_context_to_continuation",
            "selection_rule": "min_validation_mse_subject_to_mae_noninferiority",
            "validation_baseline_mse": base_mse, "validation_baseline_mae": base_mae,
            "mae_tolerance": self.mae_tolerance, "weight_candidates": candidates,
            "selected_weight": selected, "n_neighbors": self.k,
        }, {"query_split": "test", "n_preview": preview, "neighbor_train_indices": ix[:preview].tolist(),
             "symbolic_similarity": score[:preview].tolist(), "retrieval_weight": weight[:preview].tolist()})


class AnomalyEvidence:
    """Normal symbolic-event memory with a validation false-alarm budget.

    Many anomaly benchmarks provide all-normal training/validation histories.
    In that setting label-tuning a symbolic weight is impossible and test-tuning
    is invalid.  We therefore choose the largest symbolic novelty contribution
    that does not raise the 95th percentile normal validation score by more
    than a pre-declared fraction of its normal score spread.
    """

    def __init__(self, weights=(0.0, 0.1, 0.25, 0.5, 1.0), normal_score_tolerance: float = 0.05):
        self.weights = tuple(float(w) for w in weights)
        self.normal_score_tolerance = float(normal_score_tolerance)

    def fit_apply(self, train_features, val_base_score, val_features, val_label,
                  test_base_score, test_features) -> EvidenceResult:
        tr = _flat(train_features)
        self.mean_, self.scale_ = tr.mean(0, keepdims=True), np.maximum(tr.std(0, keepdims=True), 1e-5)
        def novelty(x):
            z = (_flat(x) - self.mean_) / self.scale_
            return np.mean(np.minimum(z * z, 25.0), axis=1)
        nv, nt = novelty(val_features), novelty(test_features)
        # Robust [0,1] scaling is fit on normal training events only.
        train_nv = novelty(train_features)
        lo, hi = np.quantile(train_nv, [0.05, 0.95])
        nv = np.clip((nv - lo) / max(hi - lo, 1e-6), 0, 1)
        nt = np.clip((nt - lo) / max(hi - lo, 1e-6), 0, 1)
        # A window-level symbolic novelty signal is broadcast across its points.
        base = np.asarray(val_base_score, dtype=np.float32)
        scale = np.quantile(base, 0.95) - np.quantile(base, 0.05)
        scale = max(float(scale), 1e-6)
        # Use known normal validation windows when labels exist; otherwise all
        # validation windows are treated as nominal.  This never uses test
        # labels and is valid for the usual all-normal training protocol.
        lab = np.asarray(val_label).reshape(len(base), -1).max(1).astype(int)
        normal = lab == 0
        if not normal.any():
            normal = np.ones(len(base), dtype=bool)
        base_window = base.mean(1)
        q05, q95 = np.quantile(base_window[normal], [0.05, 0.95])
        allowed_q95 = q95 + self.normal_score_tolerance * max(q95 - q05, 1e-6)
        candidates, selected = [], 0.0
        for w in sorted(set(self.weights + (0.0,))):
            score = base_window + w * scale * nv
            normal_q95 = float(np.quantile(score[normal], 0.95))
            allowed = normal_q95 <= allowed_q95 + 1e-12
            candidates.append({"weight": w, "validation_normal_q95": normal_q95,
                               "within_normal_false_alarm_budget": bool(allowed)})
            if allowed:
                selected = w
        out_score = np.asarray(test_base_score, dtype=np.float32) + selected * scale * nt[:, None]
        top = np.argsort(-nt)[: min(64, len(nt))]
        return EvidenceResult(out_score, selected, {
            "policy": "normal_symbolic_event_novelty",
            "selection_rule": "largest_weight_within_validation_normal_false_alarm_budget",
            "weight_candidates": candidates, "selected_weight": selected,
            "normal_train_rows": int(len(tr)), "normal_validation_rows": int(normal.sum()),
            "validation_normal_q95_baseline": float(q95), "validation_normal_q95_allowed": float(allowed_q95),
        }, {"query_split": "test", "top_novel_windows": top.tolist(),
             "symbolic_novelty": nt[top].tolist()})
