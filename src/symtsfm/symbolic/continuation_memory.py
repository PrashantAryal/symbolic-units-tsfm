"""Training-only symbolic target-memory for long-horizon forecasting.

This module intentionally is not a neural layer and does not depend on a
Transformer.  It turns the existing per-channel FastShapelets/BOSS-ST feature
vector into a small data-mining memory:

    symbolic context features -> nearest prototype -> mean observed target

Prototype *centres* are chosen by deterministic farthest-point selection from a
training candidate pool, while target values are averaged over every training
window assigned to them. The values may be full future trajectories (V6) or
foundation-model residual corrections (V7). Validation chooses a single convex
blending coefficient in [0, 1], with zero included. Therefore the foundation-
model forecast is always a valid validation-selected fallback and no test target
is used for fitting or blending.
"""
from __future__ import annotations

import time

import numpy as np


class SymbolicContinuationMemory:
    """Per-channel prototype memory from symbolic contexts to target trajectories."""

    def __init__(self, n_prototypes: int = 64, seed: int = 0, chunk_size: int = 2048,
                 candidate_limit: int = 4096, support_shrinkage: float = 0.0,
                 retrieval_mode: str = "prototype", n_neighbors: int = 8,
                 joint_channels: bool = False):
        if n_prototypes < 1 or chunk_size < 1 or candidate_limit < 1 or support_shrinkage < 0:
            raise ValueError("n_prototypes, chunk_size, candidate_limit must be positive and support_shrinkage non-negative")
        if retrieval_mode not in {"prototype", "knn"}:
            raise ValueError("retrieval_mode must be 'prototype' or 'knn'")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        self.n_prototypes, self.seed, self.chunk_size = int(n_prototypes), int(seed), int(chunk_size)
        self.candidate_limit = int(candidate_limit)
        self.support_shrinkage = float(support_shrinkage)
        self.retrieval_mode = retrieval_mode
        self.n_neighbors = int(n_neighbors)
        self.joint_channels = bool(joint_channels)
        self.centres_ = self.continuations_ = self.support_ = None
        self.keys_ = self.values_ = self.key_mean_ = self.key_scale_ = None
        self.candidate_indices_ = None
        self.distance_scale_ = None
        self.fit_seconds_ = None

    @staticmethod
    def _validate(features, future=None):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"features must be [n, channels, features], got {x.shape}")
        if future is None:
            return x
        y = np.asarray(future, dtype=np.float32)
        if y.ndim != 3 or y.shape[:2] != x.shape[:2]:
            raise ValueError(f"future must be [n, channels, horizon] matching {x.shape[:2]}, got {y.shape}")
        return x, y

    @staticmethod
    def _nearest(x: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest prototype index/distance without materialising [n,p,f]."""
        d2 = (x * x).sum(1, keepdims=True) + (centres * centres).sum(1)[None, :] - 2.0 * x @ centres.T
        d2 = np.maximum(d2, 0.0)
        index = d2.argmin(1)
        return index, np.sqrt(d2[np.arange(len(x)), index])

    @staticmethod
    def _farthest_centres(candidates: np.ndarray, n_centres: int, rng) -> np.ndarray:
        """Greedy k-centre anchors to avoid wasting all prototypes on one motif group."""
        first = int(rng.integers(len(candidates)))
        chosen = [first]
        min_d2 = ((candidates - candidates[first]) ** 2).sum(1)
        while len(chosen) < n_centres:
            nxt = int(min_d2.argmax())
            chosen.append(nxt)
            d2 = ((candidates - candidates[nxt]) ** 2).sum(1)
            min_d2 = np.minimum(min_d2, d2)
        return candidates[np.asarray(chosen)].copy()

    def fit(self, features, future):
        x, y = self._validate(features, future)
        n, channels, width = x.shape
        horizon = y.shape[-1]
        if self.retrieval_mode == "knn":
            return self._fit_knn(x, y)
        p = min(self.n_prototypes, n)
        rng = np.random.default_rng(self.seed)
        centres = np.empty((channels, p, width), dtype=np.float32)
        continuations = np.empty((channels, p, horizon), dtype=np.float32)
        support = np.empty((channels, p), dtype=np.int64)
        t0 = time.perf_counter()
        for channel in range(channels):
            # All training rows contribute to continuation means. Farthest-point
            # selection is performed only on a bounded deterministic candidate
            # pool so Weather-scale data remains practical on a shared server.
            pool_size = min(n, self.candidate_limit)
            pool = x[rng.choice(n, size=pool_size, replace=False), channel]
            c = self._farthest_centres(pool, p, rng)
            sums = np.zeros((p, horizon), dtype=np.float64)
            counts = np.zeros(p, dtype=np.int64)
            for start in range(0, n, self.chunk_size):
                end = min(n, start + self.chunk_size)
                assignment, _ = self._nearest(x[start:end, channel], c)
                np.add.at(sums, assignment, y[start:end, channel])
                np.add.at(counts, assignment, 1)
            default = y[:, channel].mean(0)
            cont = np.repeat(default[None, :], p, axis=0)
            nonempty = counts > 0
            raw_mean = sums[nonempty] / counts[nonempty, None]
            if self.support_shrinkage:
                # Rare retrieved motifs should not emit a large, unreliable
                # correction. Shrink them smoothly toward the global target.
                weight = counts[nonempty, None] / (counts[nonempty, None] + self.support_shrinkage)
                raw_mean = weight * raw_mean + (1.0 - weight) * default[None, :]
            cont[nonempty] = raw_mean
            centres[channel], continuations[channel], support[channel] = c, cont, counts
        self.centres_, self.continuations_, self.support_ = centres, continuations, support
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def _fit_knn(self, x: np.ndarray, y: np.ndarray):
        """Store individual training contexts and their observed continuations.

        Unlike the V6 prototype memory, this does not average unrelated motifs
        into one centroid.  A query retrieves actual train windows, with their
        immediately following targets.  For joint multivariate retrieval one
        context index is used for every channel, so cross-channel transitions
        are preserved in the retrieved trajectory.
        """
        n, channels, width = x.shape
        rng = np.random.default_rng(self.seed)
        limit = min(n, self.candidate_limit)
        # A deterministic sample bounds search cost on Weather/Electricity yet
        # keeps keys and observed continuations paired.  All candidates are
        # drawn only from the training split.
        chosen = np.sort(rng.choice(n, size=limit, replace=False))
        self.candidate_indices_ = chosen.astype(np.int64)
        t0 = time.perf_counter()
        if self.joint_channels:
            keys = x[chosen].reshape(limit, channels * width)
            mean = keys.mean(0, keepdims=True)
            scale = keys.std(0, keepdims=True)
            scale[scale < 1e-6] = 1.0
            self.keys_ = ((keys - mean) / scale).astype(np.float32)
            self.values_ = y[chosen].astype(np.float32)
            self.key_mean_, self.key_scale_ = mean.astype(np.float32), scale.astype(np.float32)
        else:
            # [channel, candidate, feature], with a separate symbolic search
            # for each channel.  This is less costly for very wide datasets.
            keys = np.swapaxes(x[chosen], 0, 1)
            mean = keys.mean(1, keepdims=True)
            scale = keys.std(1, keepdims=True)
            scale[scale < 1e-6] = 1.0
            self.keys_ = ((keys - mean) / scale).astype(np.float32)
            self.values_ = np.swapaxes(y[chosen], 0, 1).astype(np.float32)
            self.key_mean_, self.key_scale_ = mean.astype(np.float32), scale.astype(np.float32)
        # Used only for a transparent reliability diagnostic; the prediction
        # still has validation-selected alpha=0 as an exact fallback.
        self.distance_scale_ = float(np.sqrt(np.asarray(self.keys_, dtype=np.float64).var(axis=-1).mean()
                                             * self.keys_.shape[-1]))
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    @staticmethod
    def _topk_distances(query: np.ndarray, keys: np.ndarray, k: int):
        """Top-k Euclidean search in bounded chunks without a 3-D allocation."""
        d2 = ((query * query).sum(1, keepdims=True) +
              (keys * keys).sum(1)[None, :] - 2.0 * query @ keys.T)
        d2 = np.maximum(d2, 0.0)
        k = min(k, keys.shape[0])
        ids = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        local = np.take_along_axis(d2, ids, axis=1)
        order = np.argsort(local, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        return ids, np.sqrt(np.take_along_axis(local, order, axis=1))

    @staticmethod
    def _weighted_trajectories(values: np.ndarray, distances: np.ndarray):
        """Distance-weighted neighbours plus support and consistency signals."""
        base = distances[:, :1]
        # Relative distances avoid an arbitrary global temperature.  Exact
        # matches receive the dominant weight, but tied close matches share it.
        temperature = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-4)
        logits = -(distances - base) / temperature
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        while weights.ndim < values.ndim:
            weights = weights[..., None]
        pred = (weights * values).sum(axis=1)
        local_variance = ((values - pred[:, None]) ** 2).mean(axis=tuple(range(2, values.ndim)))
        flat_weights = weights.reshape(weights.shape[0], weights.shape[1])
        variance = (flat_weights * local_variance).sum(axis=1)
        effective_support = 1.0 / np.maximum((flat_weights ** 2).sum(axis=1), 1e-12)
        return pred.astype(np.float32), effective_support.astype(np.float32), variance.astype(np.float32)

    def predict(self, features, return_retrieval: bool = False):
        x = self._validate(features)
        if self.retrieval_mode == "knn":
            return self._predict_knn(x, return_retrieval)
        if self.centres_ is None:
            raise RuntimeError("fit must be called before predict")
        n, channels, _ = x.shape
        if channels != self.centres_.shape[0]:
            raise ValueError(f"memory has {self.centres_.shape[0]} channels but query has {channels}")
        horizon = self.continuations_.shape[-1]
        pred = np.empty((n, channels, horizon), dtype=np.float32)
        assignment = np.empty((n, channels), dtype=np.int64)
        distance = np.empty((n, channels), dtype=np.float32)
        for channel in range(channels):
            for start in range(0, n, self.chunk_size):
                end = min(n, start + self.chunk_size)
                ix, d = self._nearest(x[start:end, channel], self.centres_[channel])
                pred[start:end, channel] = self.continuations_[channel, ix]
                assignment[start:end, channel], distance[start:end, channel] = ix, d
        if return_retrieval:
            return pred, {"prototype_index": assignment, "prototype_distance": distance,
                          "prototype_support": self.support_[np.arange(channels)[None, :], assignment]}
        return pred

    def _predict_knn(self, x: np.ndarray, return_retrieval: bool):
        if self.keys_ is None:
            raise RuntimeError("fit must be called before predict")
        n, channels, width = x.shape
        chunks = []
        distances = []
        supports = []
        consistencies = []
        neighbor_indices = []
        if self.joint_channels:
            query = x.reshape(n, channels * width)
            query = (query - self.key_mean_) / self.key_scale_
            for start in range(0, n, self.chunk_size):
                end = min(n, start + self.chunk_size)
                ids, d = self._topk_distances(query[start:end], self.keys_, self.n_neighbors)
                value = self.values_[ids]
                p, s, v = self._weighted_trajectories(value, d)
                chunks.append(p); distances.append(d.mean(1)); supports.append(s); consistencies.append(v)
                neighbor_indices.append(self.candidate_indices_[ids])
            pred = np.concatenate(chunks, axis=0)
        else:
            pred = np.empty((n, channels, self.values_.shape[-1]), dtype=np.float32)
            distance_matrix = np.empty((n, channels), dtype=np.float32)
            support_matrix = np.empty((n, channels), dtype=np.float32)
            consistency_matrix = np.empty((n, channels), dtype=np.float32)
            neighbor_matrix = np.empty((n, channels, min(self.n_neighbors, self.keys_.shape[-2])), dtype=np.int64)
            for channel in range(channels):
                query = (x[:, channel] - self.key_mean_[channel, 0]) / self.key_scale_[channel, 0]
                for start in range(0, n, self.chunk_size):
                    end = min(n, start + self.chunk_size)
                    ids, d = self._topk_distances(query[start:end], self.keys_[channel], self.n_neighbors)
                    p, s, v = self._weighted_trajectories(self.values_[channel, ids], d)
                    pred[start:end, channel] = p
                    distance_matrix[start:end, channel] = d.mean(1)
                    support_matrix[start:end, channel] = s
                    consistency_matrix[start:end, channel] = v
                    neighbor_matrix[start:end, channel] = self.candidate_indices_[ids]
            distances, supports, consistencies = distance_matrix, support_matrix, consistency_matrix
        if self.joint_channels:
            distances = np.concatenate(distances, axis=0)
            supports = np.concatenate(supports, axis=0)
            consistencies = np.concatenate(consistencies, axis=0)
            neighbor_indices = np.concatenate(neighbor_indices, axis=0)
        else:
            neighbor_indices = neighbor_matrix
        # Reliability is deliberately evidence-based, not a learned test-time
        # score: close neighbours, multiple effective neighbours, and agreeing
        # future paths produce high confidence.  It is reported and used to
        # attenuate the retrieval correction before validation calibration.
        support_factor = np.minimum(supports / float(min(self.n_neighbors, self.keys_.shape[-2])), 1.0)
        distance_factor = np.exp(-distances / max(self.distance_scale_, 1e-6))
        consistency_factor = 1.0 / (1.0 + np.sqrt(np.maximum(consistencies, 0.0)))
        reliability = (support_factor * distance_factor * consistency_factor).astype(np.float32)
        if self.joint_channels:
            reliability = reliability[:, None, None]
        else:
            reliability = reliability[:, :, None]
        if return_retrieval:
            return pred, {
                "neighbor_distance": distances, "effective_support": supports,
                "future_disagreement": consistencies, "reliability": reliability.squeeze(-1),
                "neighbor_train_indices": neighbor_indices,
            }
        return pred

    @staticmethod
    def choose_alpha(baseline, memory, target, n_grid: int = 21,
                     mae_tolerance: float = 0.0):
        """Choose a validation-only blend without sacrificing validation MAE.

        Among blends whose MAE is no worse than the foundation-model baseline
        plus ``mae_tolerance``, select the one with the lowest MSE.  This avoids
        accepting a small MSE reduction that produces a material MAE increase.
        Alpha zero is always eligible (for a non-negative tolerance), so the
        original foundation-model prediction remains a valid fallback.
        """
        base, mem, y = (np.asarray(a, dtype=np.float64) for a in (baseline, memory, target))
        if base.shape != mem.shape or base.shape != y.shape:
            raise ValueError(f"baseline/memory/target shapes must agree, got {base.shape}, {mem.shape}, {y.shape}")
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2 so alpha=0 and alpha=1 are available")
        if mae_tolerance < 0:
            raise ValueError("mae_tolerance must be non-negative")
        alphas = np.linspace(0.0, 1.0, n_grid)
        errors = np.stack([base + a * (mem - base) - y for a in alphas])
        mse = np.mean(errors ** 2, axis=tuple(range(1, errors.ndim)))
        mae = np.mean(np.abs(errors), axis=tuple(range(1, errors.ndim)))
        eligible = mae <= mae[0] + float(mae_tolerance) + 1e-12
        # Alpha zero is eligible. Inf makes non-eligible blends impossible to
        # select; np.argmin keeps the lower alpha on an exact MSE tie.
        i = int(np.where(eligible, mse, np.inf).argmin())
        return float(alphas[i]), {
            "selection_rule": "min_validation_mse_subject_to_mae_noninferiority",
            "validation_mae_tolerance": float(mae_tolerance),
            "validation_baseline_mse": float(mse[0]),
            "validation_baseline_mae": float(mae[0]),
            "validation_memory_mse": float(mse[-1]),
            "validation_memory_mae": float(mae[-1]),
            "validation_selected_mse": float(mse[i]),
            "validation_selected_mae": float(mae[i]),
            "validation_eligible_alpha_count": int(eligible.sum()),
            "alpha_grid_size": int(n_grid),
        }

    @staticmethod
    def choose_selective_alpha(baseline, memory, target, reliability, n_grid: int = 21,
                               threshold_grid: int = 21, mae_tolerance: float = 0.0):
        """Validation-calibrate a correction only where retrieval is reliable.

        A single global blend can harm windows whose historical neighbours have
        weak support or disagreeing continuations. This procedure chooses both
        a blend alpha and a minimum reliability threshold on validation data.
        The threshold is based only on retrieval information available at test
        time; validation targets are used solely to select the fixed rule.
        """
        base, mem, y = (np.asarray(a, dtype=np.float64) for a in (baseline, memory, target))
        rel = np.asarray(reliability, dtype=np.float64)
        while rel.ndim < base.ndim:
            rel = rel[..., None]
        if base.shape != mem.shape or base.shape != y.shape:
            raise ValueError(f"baseline/memory/target shapes must agree, got {base.shape}, {mem.shape}, {y.shape}")
        if rel.shape[0] != base.shape[0]:
            raise ValueError(f"reliability has {rel.shape[0]} rows but baseline has {base.shape[0]}")
        if n_grid < 2 or threshold_grid < 2 or mae_tolerance < 0:
            raise ValueError("n_grid/threshold_grid must be at least 2 and mae_tolerance non-negative")
        # Include +inf as the explicit exact-baseline fallback, even when every
        # retrieval score is positive. Quantile thresholds make the procedure
        # robust to different reliability scales across datasets.
        thresholds = np.unique(np.concatenate([
            np.quantile(rel.reshape(len(rel), -1).mean(1), np.linspace(0.0, 1.0, threshold_grid)),
            [np.inf],
        ]))
        alphas = np.linspace(0.0, 1.0, n_grid)
        base_error = base - y
        base_mse, base_mae = float(np.mean(base_error ** 2)), float(np.mean(np.abs(base_error)))
        best = None
        for threshold in thresholds:
            mask = rel >= threshold
            for alpha in alphas:
                candidate = base + alpha * mask * (mem - base)
                err = candidate - y
                mse, mae = float(np.mean(err ** 2)), float(np.mean(np.abs(err)))
                if mae <= base_mae + float(mae_tolerance) + 1e-12:
                    item = (mse, float(alpha), float(threshold), float(mask.mean()), mae)
                    # Deterministic tie break: prefer less intervention.
                    if best is None or item[0] < best[0] - 1e-12 or (
                        abs(item[0] - best[0]) <= 1e-12 and item[3] < best[3]):
                        best = item
        assert best is not None  # alpha=0 is always eligible
        mse, alpha, threshold, coverage, mae = best
        return alpha, threshold, {
            "selection_rule": "min_validation_mse_subject_to_mae_noninferiority_selective_retrieval",
            "validation_mae_tolerance": float(mae_tolerance),
            "validation_baseline_mse": base_mse,
            "validation_baseline_mae": base_mae,
            "validation_selected_mse": mse,
            "validation_selected_mae": mae,
            "selected_alpha": alpha,
            "selected_min_reliability": threshold,
            "validation_selected_fraction": coverage,
            "alpha_grid_size": int(n_grid),
            "reliability_threshold_grid_size": int(threshold_grid),
        }

    def summary(self):
        if self.retrieval_mode == "knn" and self.keys_ is not None:
            return {"retrieval_mode": "individual_knn", "joint_channels": self.joint_channels,
                    "n_candidates": int(self.keys_.shape[-2]), "n_neighbors": self.n_neighbors,
                    "fit_seconds": float(self.fit_seconds_), "candidate_limit": self.candidate_limit,
                    "distance_scale": float(self.distance_scale_)}
        if self.centres_ is None:
            return {}
        return {"n_prototypes": int(self.centres_.shape[1]), "channels": int(self.centres_.shape[0]),
                "horizon": int(self.continuations_.shape[-1]), "fit_seconds": float(self.fit_seconds_),
                "mean_prototype_support": float(self.support_.mean()),
                "prototype_candidate_limit": self.candidate_limit,
                "support_shrinkage": self.support_shrinkage}
