"""Embedding retrieval with symbolic reranking for forecasting.

This module deliberately keeps the two jobs separate:

* a foundation-model representation retrieves contexts that are predictive in
  the model's feature space;
* symbolic words rerank those candidates and provide inspectable provenance.

It is not a replacement for a TSFM.  It is a small, downstream adapter that
learns whether the observed continuations of retrieved historical contexts
should correct the TSFM forecast for a particular query.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def pooled_symbolic_features(features: np.ndarray) -> np.ndarray:
    """Turn per-channel symbolic features into a joint retrieval descriptor."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected [n, channels, features], got {x.shape}")
    return x.reshape(len(x), -1)


@dataclass
class RetrievalBatch:
    future: np.ndarray
    embedding_score: np.ndarray
    symbolic_score: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


class SymbolicEmbeddingRetriever:
    """Top-k historical context--future retrieval with transparent reranking."""

    def __init__(self, n_neighbors: int = 8, symbolic_weight: float = 0.20,
                 chunk_size: int = 256):
        if n_neighbors < 1 or chunk_size < 1 or symbolic_weight < 0:
            raise ValueError("n_neighbors/chunk_size must be positive and symbolic_weight non-negative")
        self.n_neighbors = int(n_neighbors)
        self.symbolic_weight = float(symbolic_weight)
        self.chunk_size = int(chunk_size)
        self.indices_ = self.embedding_ = self.symbolic_ = self.future_ = None

    def fit(self, embeddings, symbolic, future, indices=None):
        e = np.asarray(embeddings, dtype=np.float32)
        s = pooled_symbolic_features(symbolic)
        y = np.asarray(future, dtype=np.float32)
        if e.ndim != 2 or y.ndim != 3 or len(e) != len(s) or len(e) != len(y):
            raise ValueError("embedding/symbolic/future rows must agree: [n,d], [n,c,f], [n,c,h]")
        ix = np.arange(len(e), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if len(ix) != len(e):
            raise ValueError("indices must have one entry per bank row")
        self.indices_ = ix
        self.embedding_ = _l2(e)
        # Standardise dimensions before cosine similarity, preventing a few
        # high-variance symbolic counts from dominating a word match.
        mu, sd = s.mean(0, keepdims=True), s.std(0, keepdims=True)
        self.symbolic_ = _l2((s - mu) / np.maximum(sd, 1e-5))
        self.symbolic_mean_, self.symbolic_scale_ = mu, np.maximum(sd, 1e-5)
        self.future_ = y
        return self

    def retrieve(self, embeddings, symbolic, symbolic_weight=None) -> RetrievalBatch:
        if self.embedding_ is None:
            raise RuntimeError("fit must be called before retrieve")
        e = _l2(np.asarray(embeddings, dtype=np.float32))
        s = pooled_symbolic_features(symbolic)
        s = _l2((s - self.symbolic_mean_) / self.symbolic_scale_)
        if e.shape[1] != self.embedding_.shape[1] or s.shape[1] != self.symbolic_.shape[1]:
            raise ValueError("query feature dimensions do not match the retrieval bank")
        beta = self.symbolic_weight if symbolic_weight is None else float(symbolic_weight)
        k = min(self.n_neighbors, len(self.indices_))
        futures = []; escores = []; sscores = []; weights = []; indices = []
        for start in range(0, len(e), self.chunk_size):
            end = min(len(e), start + self.chunk_size)
            ec = e[start:end] @ self.embedding_.T
            sc = s[start:end] @ self.symbolic_.T
            score = ec + beta * sc
            local = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
            local_score = np.take_along_axis(score, local, axis=1)
            order = np.argsort(-local_score, axis=1)
            local = np.take_along_axis(local, order, axis=1)
            local_score = np.take_along_axis(local_score, order, axis=1)
            # A query-specific softmax is a retrieval confidence distribution,
            # not a learned forecast blend.  The learned mixer consumes it.
            w = np.exp(local_score - local_score.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            futures.append(self.future_[local])
            escores.append(np.take_along_axis(ec, local, axis=1))
            sscores.append(np.take_along_axis(sc, local, axis=1))
            weights.append(w.astype(np.float32)); indices.append(self.indices_[local])
        return RetrievalBatch(np.concatenate(futures), np.concatenate(escores), np.concatenate(sscores),
                              np.concatenate(weights), np.concatenate(indices))


class _AdaptiveMixer(nn.Module):
    """Small residual adapter over a base forecast and retrieved continuations."""

    def __init__(self, embedding_dim: int, channels: int, horizon: int, hidden_dim: int = 128):
        super().__init__()
        self.channels, self.horizon = int(channels), int(horizon)
        # mean/max embedding similarity, symbolic agreement, and retrieval
        # entropy tell the gate how much evidence is available for this query.
        in_dim = int(embedding_dim) + 4
        self.context = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.gate = nn.Linear(hidden_dim, channels)
        self.residual = nn.Linear(hidden_dim, channels * horizon)
        nn.init.constant_(self.gate.bias, -3.0)  # start near the exact UniTS path
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(self, base, embedding, ref, embedding_score, symbolic_score, weights):
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1, keepdim=True)
        aux = torch.cat([embedding, embedding_score.mean(1, keepdim=True), embedding_score.max(1, keepdim=True).values,
                         symbolic_score.mean(1, keepdim=True), entropy], dim=1)
        z = self.context(aux)
        gate = torch.sigmoid(self.gate(z))[:, :, None]
        learned_residual = self.residual(z).view(-1, self.channels, self.horizon)
        return base + gate * (ref - base + learned_residual), gate


class AdaptiveRetrievalMixer:
    """Train/evaluate the query-dependent retrieval adapter on held-out time blocks."""

    def __init__(self, embedding_dim, channels, horizon, *, hidden_dim=128, lr=1e-3,
                 weight_decay=1e-4, gate_l1=1e-3, epochs=40, batch_size=128,
                 patience=6, device="cpu", seed=0):
        self.cfg = {"embedding_dim": int(embedding_dim), "channels": int(channels), "horizon": int(horizon),
                    "hidden_dim": int(hidden_dim), "lr": float(lr), "weight_decay": float(weight_decay),
                    "gate_l1": float(gate_l1), "epochs": int(epochs), "batch_size": int(batch_size),
                    "patience": int(patience), "seed": int(seed)}
        self.device = torch.device(device)
        self.model = _AdaptiveMixer(embedding_dim, channels, horizon, hidden_dim).to(self.device)
        self.fit_seconds_ = 0.0

    def _tensor(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)

    @staticmethod
    def _reference(batch: RetrievalBatch) -> np.ndarray:
        return (batch.future * batch.weights[:, :, None, None]).sum(1).astype(np.float32)

    def _forward(self, base, embedding, retrieved: RetrievalBatch):
        ref = self._reference(retrieved)
        return self.model(self._tensor(base), self._tensor(embedding), self._tensor(ref),
                          self._tensor(retrieved.embedding_score), self._tensor(retrieved.symbolic_score),
                          self._tensor(retrieved.weights))

    def fit(self, train_base, train_embedding, train_retrieval, train_target,
            val_base, val_embedding, val_retrieval, val_target):
        n = len(train_base)
        if n < 2 or len(val_base) < 1:
            raise ValueError("retrieval mixer needs non-empty train and validation blocks")
        torch.manual_seed(self.cfg["seed"])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        best, bad, best_state = float("inf"), 0, None
        t0 = time.perf_counter()
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            order = np.random.default_rng(self.cfg["seed"] + epoch).permutation(n)
            for start in range(0, n, self.cfg["batch_size"]):
                idx = order[start:start + self.cfg["batch_size"]]
                sub = RetrievalBatch(*(getattr(train_retrieval, name)[idx] for name in
                                       ("future", "embedding_score", "symbolic_score", "weights", "indices")))
                pred, gate = self._forward(np.asarray(train_base)[idx], np.asarray(train_embedding)[idx], sub)
                loss = F.mse_loss(pred, self._tensor(np.asarray(train_target)[idx])) + self.cfg["gate_l1"] * gate.mean()
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0); opt.step()
            self.model.eval()
            with torch.no_grad():
                val_pred, _ = self._forward(val_base, val_embedding, val_retrieval)
                val_mse = F.mse_loss(val_pred, self._tensor(val_target)).item()
            if val_mse < best - 1e-10:
                best, bad = val_mse, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        self.model.load_state_dict(best_state)
        self.fit_seconds_ = time.perf_counter() - t0
        self.validation_mse_ = float(best)
        self.epochs_run_ = int(epoch + 1)
        return self

    def predict(self, base, embedding, retrieval: RetrievalBatch):
        self.model.eval()
        with torch.no_grad():
            out, gate = self._forward(base, embedding, retrieval)
        return out.cpu().numpy(), gate.cpu().numpy()

    def summary(self):
        return {**self.cfg, "fit_seconds": float(self.fit_seconds_), "validation_mse": self.validation_mse_,
                "epochs_run": self.epochs_run_, "mixer": "adaptive_query_conditioned_residual"}
