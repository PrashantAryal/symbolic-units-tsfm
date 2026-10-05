"""Turn top-K SAX/SFA words into fixed-length, standardised feature vectors.

Per channel and per selected word k the vector holds
``[presence_k, frequency_k, distance_k]`` for k = 1..K, followed by one
``rarity`` feature (max -log training-frequency of any word in the input),
so every channel contributes ``3K + 1`` features.

All statistics (selected words, MCB bins, feature mean/std) are fitted on the
training split only. Standardisation matters: the technical documentation's
"naive concatenation" rejection is exactly about un-scaled foreign features
distorting the representation the task head sees.
"""
from __future__ import annotations

import logging
import pickle
import time

import numpy as np

from symtsfm.symbolic.sax import FastShapeletSelector, paa, sax_symbols
from symtsfm.symbolic.sfa import BOSSSTSelector

log = logging.getLogger(__name__)
FEATURE_KINDS = ("presence", "frequency", "distance")
METHODS = {"fastshapelets": FastShapeletSelector, "bossst": BOSSSTSelector}


def ordered_sax_context_key(X: np.ndarray, n_words: int = 24,
                            alphabet_size: int = 4) -> np.ndarray:
    """Ordered SAX signature for retrieval, preserving the recent motif order.

    The usual feature vector tells whether selected words occurred and where
    their *best* match was. It can still map many different contexts to the
    same pooled key. This signature discretises the entire input context into
    an ordered PAA/SAX word sequence for every channel. It is a retrieval key,
    not a learned forecast feature, and uses no future values.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must be [n, channels, length], got {X.shape}")
    if n_words < 2 or n_words > X.shape[-1]:
        raise ValueError(f"n_words must be in [2, {X.shape[-1]}], got {n_words}")
    symbols = sax_symbols(X, word_len=int(n_words), alphabet_size=int(alphabet_size))
    # Integer symbols preserve their ordinal SAX state. The kNN retriever
    # standardises each feature from train data after concatenation.
    return symbols.astype(np.float32)


def forecast_state_key(X: np.ndarray, n_segments: int = 24,
                       recent: int = 24) -> np.ndarray:
    """Level/trend/volatility context retained alongside symbolic motif order.

    SAX normalises each context, intentionally removing its level and scale.
    That is helpful for motif discovery but harmful when a similar waveform has
    a different local operating state and therefore a different continuation.
    This key uses only observed, train-scaled context values: raw PAA levels,
    last value, mean, standard deviation, recent-minus-global mean, endpoint
    change, and least-squares slope.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must be [n, channels, length], got {X.shape}")
    if n_segments < 2 or n_segments > X.shape[-1]:
        raise ValueError(f"n_segments must be in [2, {X.shape[-1]}], got {n_segments}")
    recent = int(np.clip(recent, 2, X.shape[-1]))
    t = np.linspace(-1.0, 1.0, X.shape[-1], dtype=np.float32)
    slope = (X * t).mean(-1, keepdims=True) / max(float((t * t).mean()), 1e-8)
    global_mean = X.mean(-1, keepdims=True)
    summary = np.concatenate([
        X[..., -1:], global_mean, X.std(-1, keepdims=True),
        X[..., -1:] - X[..., :1],
        X[..., -recent:].mean(-1, keepdims=True) - global_mean,
        slope,
    ], axis=-1)
    return np.concatenate([paa(X, int(n_segments)).astype(np.float32), summary], axis=-1)


def resolve_window(window, length: int, min_window: int = 4) -> int:
    """``window`` may be an int (points) or a float in (0, 1] (fraction of length)."""
    w = int(round(window * length)) if isinstance(window, float) and window <= 1 else int(window)
    return int(np.clip(w, min_window, length))


def forecast_direction_labels(context: np.ndarray, future: np.ndarray, n_bins: int = 3,
                              tail: int = 24, edges: np.ndarray | None = None):
    """Pseudo-classes for forecasting windows so words can be IG-scored on the train split.

    Label = quantile bin of (mean(future) - mean(last ``tail`` context points)) / std(context),
    per channel. ``context`` [n, C, L], ``future`` [n, C, H] -> labels [n, C], edges.
    """
    sd = context.std(-1) + 1e-8
    delta = (future.mean(-1) - context[..., -tail:].mean(-1)) / sd
    if edges is None:
        edges = np.quantile(delta, np.arange(1, n_bins) / n_bins)
    return np.digitize(delta, edges), edges


class SymbolicFeatureExtractor:
    """Fits one selector per channel and produces ``[n, C, 3K+1]`` standardised features."""

    def __init__(self, method: str, window, word_len: int = 8, alphabet_size: int = 4,
                 top_k: int = 10, step: int = 1, max_fit_series: int | None = 500,
                 seed: int = 0, method_kwargs: dict | None = None, chunk: int = 256):
        if method not in METHODS:
            raise ValueError(f"unknown symbolic method {method!r}; choose from {list(METHODS)}")
        self.method = method
        self.window = window
        self.word_len = word_len
        self.alphabet_size = alphabet_size
        self.top_k = top_k
        self.step = step
        self.max_fit_series = max_fit_series
        self.seed = seed
        self.method_kwargs = dict(method_kwargs or {})
        self.chunk = chunk
        self.timings = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """X [n, C, T]; y None, [n] (shared labels) or [n, C] (per-channel labels)."""
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        n, C, T = X.shape
        self.n_channels, self.length = C, T
        self.window_ = resolve_window(self.window, T)
        self.selectors_ = []
        for c in range(C):
            yc = None if y is None else (y if np.ndim(y) == 1 else y[:, c])
            sel = METHODS[self.method](
                window=self.window_, word_len=self.word_len, alphabet_size=self.alphabet_size,
                top_k=self.top_k, step=self.step, max_fit_series=self.max_fit_series,
                seed=self.seed + c, **self.method_kwargs,
            )
            self.selectors_.append(sel.fit(X[:, c], yc))
        # Real multivariate data contains dead channels (constant sensors, one-hot command
        # flags). Those yield fewer words -- or none. Every channel's block is padded to the
        # same width K with zero features and "<none>" placeholder words, so the feature
        # tensor stays rectangular without inventing evidence that does not exist.
        self.k_ = [len(s.words_) for s in self.selectors_]
        self.K = max(self.k_) if self.k_ else 0
        if self.K == 0:
            raise RuntimeError(
                "no symbolic words could be selected on any channel: every channel looks "
                "constant. Check the input, or lower symbolic 'window'.")
        if len(set(self.k_)) != 1:
            thin = [c for c, k in enumerate(self.k_) if k < self.K]
            log.info("channels with fewer than K=%d words (padded): %s", self.K, thin)
        self.timings["fit_s"] = time.perf_counter() - t0
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None):
        """Fit selectors and standardisation statistics on the (training) split X."""
        self.fit(X, y)
        t0 = time.perf_counter()
        raw, meta = self._raw(np.asarray(X, dtype=float))
        flat = raw.reshape(-1, raw.shape[-1])
        self.mean_ = flat.mean(0)
        self.std_ = flat.std(0)
        self.std_[self.std_ < 1e-6] = 1.0
        self.timings["transform_s"] = time.perf_counter() - t0
        return ((raw - self.mean_) / self.std_).astype(np.float32), meta

    @property
    def dim_per_channel(self) -> int:
        return 3 * self.K + 1

    # ------------------------------------------------------------------ transform
    def _raw(self, X):
        n, C, _ = X.shape
        out = np.zeros((n, C, self.dim_per_channel), dtype=np.float32)
        meta = {
            "location": np.zeros((n, C, self.K), dtype=np.int64),
            "distance": np.zeros((n, C, self.K), dtype=np.float32),
            "presence": np.zeros((n, C, self.K), dtype=np.float32),
            "informative": np.zeros((n, C, self.K), dtype=bool),
            "rarity_location": np.zeros((n, C), dtype=np.int64),
        }
        for s in range(0, n, self.chunk):
            xb = X[s : s + self.chunk]
            for c, sel in enumerate(self.selectors_):
                m = sel.match(xb[:, c])
                k = self.k_[c]
                if k:  # leave the padding slots at zero for a channel with fewer words
                    block = np.stack([m[key] for key in FEATURE_KINDS], -1).reshape(len(xb), -1)
                    out[s : s + len(xb), c, : 3 * k] = block
                    for key in ("location", "distance", "presence", "informative"):
                        meta[key][s : s + len(xb), c, :k] = m[key]
                out[s : s + len(xb), c, -1] = m["rarity"]
                meta["rarity_location"][s : s + len(xb), c] = m["rarity_location"]
        return out, meta

    def transform(self, X: np.ndarray):
        """X [n, C, T] -> standardised features [n, C, 3K+1] and match metadata."""
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {X.shape[1]}")
        t0 = time.perf_counter()
        raw, meta = self._raw(X)
        feats = ((raw - self.mean_) / self.std_).astype(np.float32)
        self.timings["transform_s"] = self.timings.get("transform_s", 0.0) + time.perf_counter() - t0
        return feats, meta

    # ------------------------------------------------------------------ description
    def word_of_feature(self, j: int):
        """Feature index within a channel -> (word index or None for rarity, kind)."""
        if j == self.dim_per_channel - 1:
            return None, "rarity"
        return j // 3, FEATURE_KINDS[j % 3]

    PLACEHOLDER = {"word": "<none>", "info_gain": 0.0, "class": None, "support": 0.0}

    def words(self, channel: int) -> list[dict]:
        """Selected words for this channel, padded to K with placeholders."""
        got = self.selectors_[channel].describe()
        return got + [dict(self.PLACEHOLDER) for _ in range(self.K - len(got))]

    def summary(self) -> dict:
        return {
            "method": self.method,
            "window": self.window_,
            "word_len": self.word_len,
            "alphabet_size": self.alphabet_size,
            "top_k": self.K,
            "dim_per_channel": self.dim_per_channel,
            "channels": [self.words(c) for c in range(self.n_channels)],
            "timings": dict(self.timings),
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "SymbolicFeatureExtractor":
        with open(path, "rb") as f:
            return pickle.load(f)
