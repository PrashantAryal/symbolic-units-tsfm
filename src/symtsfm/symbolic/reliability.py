"""Sparse symbolic reliability heads and risk-coverage evaluation (Phase 10).

A reliability head predicts, for a frozen foundation-model output, how likely
that output is to be wrong (classification) or how large its error will be
(forecasting).  The symbolic head is an L1-regularised linear model over the
FastShapelets / BOSS-ST feature vector, so every risk score decomposes exactly
into ``coefficient x standardised feature`` terms that name a channel, a word
and a feature kind (presence / frequency / distance / rarity).

Ranking test outputs by predicted risk gives selective prediction: answer the
most reliable fraction (coverage) and defer the rest.  Quality is measured by
the area under the risk-coverage curve (AURC, lower is better).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

COVERAGE_GRID = np.round(np.linspace(0.05, 1.0, 20), 2)


# =========================================================================== risk-coverage
def _order(score: np.ndarray, seed: int) -> np.ndarray:
    """Accept the lowest predicted risk first; ties broken by a seeded random key."""
    tie = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tie, np.asarray(score, dtype=np.float64)))


def risk_coverage(score: np.ndarray, loss: np.ndarray, seed: int = 0) -> dict:
    loss = np.asarray(loss, dtype=np.float64)
    n = len(loss)
    cum = np.cumsum(loss[_order(score, seed)]) / np.arange(1, n + 1)
    oracle = np.cumsum(np.sort(loss)) / np.arange(1, n + 1)

    def at(cov):
        return float(cum[max(1, int(math.ceil(cov * n))) - 1])

    full = float(loss.mean())
    aurc = float(cum.mean())
    return {
        "aurc": aurc,
        "oracle_aurc": float(oracle.mean()),
        "e_aurc": aurc - float(oracle.mean()),
        "full_coverage_risk": full,
        # A random ranking has expected AURC equal to the full-coverage risk.
        "aurc_gain_vs_random": (1.0 - aurc / full) if full > 0 else 0.0,
        "risk_at_50": at(0.5), "risk_at_80": at(0.8), "risk_at_90": at(0.9),
        "curve": {"coverage": COVERAGE_GRID.tolist(), "risk": [at(c) for c in COVERAGE_GRID]},
    }


# =========================================================================== features
STAT_NAMES = ("mean", "std", "min", "max", "last", "slope", "mean_abs_diff", "recent_shift")


def summary_statistics(X: np.ndarray) -> np.ndarray:
    """Non-symbolic control features: X [n, C, T] -> [n, C * 8]."""
    X = np.asarray(X, dtype=np.float64)
    n, C, T = X.shape
    t = np.arange(T) - (T - 1) / 2.0
    tail = max(2, T // 4)
    feats = np.stack([
        X.mean(-1), X.std(-1), X.min(-1), X.max(-1), X[..., -1],
        (X * t).sum(-1) / max((t ** 2).sum(), 1e-12),
        np.abs(np.diff(X, axis=-1)).mean(-1) if T > 1 else np.zeros((n, C)),
        X[..., -tail:].mean(-1) - X.mean(-1),
    ], axis=-1)  # [n, C, 8]
    return np.nan_to_num(feats.reshape(n, -1).astype(np.float32))


def summary_statistic_names(n_channels: int) -> list[dict]:
    return [{"channel": c, "kind": k, "word": None} for c in range(n_channels) for k in STAT_NAMES]


def symbolic_feature_names(extractor) -> list[dict]:
    """Flat [C * (3K+1)] symbolic feature index -> channel / word / kind."""
    names = []
    for c in range(extractor.n_channels):
        words = extractor.words(c)
        for j in range(extractor.dim_per_channel):
            word_i, kind = extractor.word_of_feature(j)
            word = None if word_i is None else words[word_i]
            names.append({"channel": c, "kind": kind, "word_index": word_i,
                          "word": None if word is None else word.get("word"),
                          "word_class": None if word is None else word.get("class"),
                          "support": None if word is None else word.get("support")})
    return names


def context_volatility(X: np.ndarray) -> np.ndarray:
    """Heuristic forecasting risk: recent within-window volatility, averaged over channels."""
    X = np.asarray(X, dtype=np.float64)
    tail = max(2, X.shape[-1] // 4)
    return X[..., -tail:].std(-1).mean(-1)


# =========================================================================== sparse head
class SparseReliability:
    """L1 logistic (P[wrong]) or Lasso (log error) reliability readout."""

    def __init__(self, kind: str, n_folds: int = 5, n_c: int = 10, seed: int = 0,
                 time_ordered: bool = False, max_iter: int = 5000):
        if kind not in ("classification", "regression"):
            raise ValueError(kind)
        self.kind, self.n_folds, self.n_c, self.seed = kind, int(n_folds), int(n_c), int(seed)
        self.time_ordered, self.max_iter = bool(time_ordered), int(max_iter)

    def fit(self, F: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import StandardScaler

        F = np.nan_to_num(np.asarray(F, dtype=np.float64))
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(F)
        Z = self.scaler_.transform(F)
        self.constant_ = None
        self.n_fit_ = len(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.kind == "classification":
                from sklearn.linear_model import LogisticRegressionCV
                from sklearn.model_selection import StratifiedKFold

                y = y.astype(int)
                n_pos, n_neg = int(y.sum()), int((1 - y).sum())
                folds = min(self.n_folds, n_pos, n_neg)
                if folds < 2:
                    # Too few failures to learn from: abstain (constant risk), recorded honestly.
                    self.constant_ = float(y.mean())
                    self.coef_ = np.zeros(Z.shape[1])
                    return self
                # C <= 10: beyond that liblinear L1 stalls on near-separable rows
                # and the rule cards stop being sparse.
                self.model_ = LogisticRegressionCV(
                    Cs=np.logspace(-2.5, 1.0, self.n_c),
                    cv=StratifiedKFold(folds, shuffle=True, random_state=self.seed),
                    penalty="l1", solver="liblinear", scoring="neg_log_loss", class_weight="balanced",
                    max_iter=self.max_iter, random_state=self.seed).fit(Z, y)
                self.coef_ = self.model_.coef_[0].copy()
            else:
                from sklearn.linear_model import LassoCV
                from sklearn.model_selection import KFold

                cv = KFold(self.n_folds, shuffle=not self.time_ordered,
                           random_state=None if self.time_ordered else self.seed)
                self.model_ = LassoCV(n_alphas=30, cv=cv, max_iter=self.max_iter,
                                      random_state=self.seed).fit(Z, y.astype(np.float64))
                self.coef_ = self.model_.coef_.copy()
        return self

    def _z(self, F):
        return self.scaler_.transform(np.nan_to_num(np.asarray(F, dtype=np.float64)))

    def score(self, F: np.ndarray) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(F), self.constant_)
        Z = self._z(F)
        if self.kind == "classification":
            return self.model_.predict_proba(Z)[:, 1]
        return self.model_.predict(Z)

    def contributions(self, F: np.ndarray) -> np.ndarray:
        """Exact additive decomposition of the linear risk logit / prediction."""
        return self._z(F) * self.coef_[None, :]

    @property
    def n_nonzero(self) -> int:
        return int(np.count_nonzero(self.coef_))

    def top_terms(self, names: list[dict], k: int = 10) -> list[dict]:
        order = np.argsort(-np.abs(self.coef_), kind="stable")
        rows = []
        for i in order[:k]:
            if self.coef_[i] == 0:
                break
            rows.append({**names[i], "feature_index": int(i), "coefficient": float(self.coef_[i]),
                         "direction": "raises_risk" if self.coef_[i] > 0 else "lowers_risk"})
        return rows


def permutation_null(F_fit, y_fit, F_test, loss_test, kind: str, n_perm: int, seed: int,
                     time_ordered: bool = False, n_folds: int = 5) -> dict:
    """AURC of the same head trained on permuted reliability targets (null distribution)."""
    rng = np.random.default_rng(seed + 7919)
    null = []
    for p in range(int(n_perm)):
        head = SparseReliability(kind, n_folds=n_folds, seed=seed + p, time_ordered=time_ordered)
        head.fit(F_fit, rng.permutation(np.asarray(y_fit)))
        null.append(risk_coverage(head.score(F_test), loss_test, seed=seed)["aurc"])
    return {"n_perm": int(n_perm), "null_aurc": [float(v) for v in null],
            "null_mean": float(np.mean(null)) if null else float("nan"),
            "null_std": float(np.std(null)) if null else float("nan")}


def permutation_p_value(observed_aurc: float, null: dict) -> float:
    """One-sided p: fraction of permuted heads with AURC <= the observed head."""
    values = np.asarray(null["null_aurc"])
    return float((1 + (values <= observed_aurc).sum()) / (1 + len(values)))
