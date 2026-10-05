"""Small, deterministic multi-output ridge model for symbolic forecast residuals.

The model is intentionally a transparent data-mining component.  For each
channel it maps a multi-scale symbolic feature vector to all forecast-horizon
residual values using one ridge regression.  It has no access to validation or
test targets during fitting.
"""
from __future__ import annotations

import numpy as np


class SymbolicResidualRidge:
    """Channel-wise standardised multi-output ridge residual predictor."""

    def __init__(self, ridge_alpha: float = 10.0):
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        self.ridge_alpha = float(ridge_alpha)
        self.x_mean_ = self.x_scale_ = self.coef_ = None

    @staticmethod
    def _validate(features, residual=None):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"features must be [n, channels, features], got {x.shape}")
        if residual is None:
            return x
        y = np.asarray(residual, dtype=np.float64)
        if y.ndim != 3 or y.shape[:2] != x.shape[:2]:
            raise ValueError(f"residual must be [n, channels, horizon] matching {x.shape[:2]}, got {y.shape}")
        return x, y

    def fit(self, features, residual):
        x, y = self._validate(features, residual)
        n, channels, width = x.shape
        horizon = y.shape[-1]
        means = x.mean(0)
        scales = x.std(0)
        scales[scales < 1e-8] = 1.0
        coef = np.empty((channels, width + 1, horizon), dtype=np.float64)
        eye = np.eye(width + 1, dtype=np.float64)
        eye[0, 0] = 0.0  # never penalise intercept
        for c in range(channels):
            z = (x[:, c] - means[c]) / scales[c]
            design = np.concatenate([np.ones((n, 1)), z], axis=1)
            lhs = design.T @ design + self.ridge_alpha * eye
            rhs = design.T @ y[:, c]
            coef[c] = np.linalg.solve(lhs, rhs)
        self.x_mean_, self.x_scale_, self.coef_ = means, scales, coef
        return self

    def predict(self, features):
        x = self._validate(features)
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        if x.shape[1:] != (self.coef_.shape[0], self.coef_.shape[1] - 1):
            raise ValueError("feature channel/width does not match fitted ridge")
        n, channels, _ = x.shape
        out = np.empty((n, channels, self.coef_.shape[-1]), dtype=np.float32)
        for c in range(channels):
            z = (x[:, c] - self.x_mean_[c]) / self.x_scale_[c]
            out[:, c] = (self.coef_[c, 0] + z @ self.coef_[c, 1:]).astype(np.float32)
        return out

    def summary(self, feature_labels=None, top_n: int = 10):
        if self.coef_ is None:
            return {}
        strength = np.abs(self.coef_[:, 1:]).mean((0, 2))
        order = np.argsort(-strength)[:top_n]
        labels = feature_labels or [f"feature_{i}" for i in range(len(strength))]
        return {
            "mode": "multiscale_ridge_residual",
            "ridge_alpha": self.ridge_alpha,
            "channels": int(self.coef_.shape[0]),
            "feature_dim": int(self.coef_.shape[1] - 1),
            "horizon": int(self.coef_.shape[-1]),
            "top_features_by_mean_absolute_coefficient": [
                {"feature": str(labels[i]), "mean_absolute_coefficient": float(strength[i])}
                for i in order
            ],
        }
