"""BOSS-ST-style symbolic path: sliding window -> (z-norm) -> DFT -> MCB
quantisation -> SFA words -> per-class word statistics -> Information Gain on
the symbolic features directly (no real-distance verification: BOSS's
"symbolic distance as a tabular feature").

References: Schaefer, "The BOSS is concerned with time series classification in
the presence of noise", DMKD 2015; Schaefer & Hoegqvist, "SFA: a symbolic
Fourier approximation and index for similarity search", EDBT 2012.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from symtsfm.symbolic.sax import (
    SymbolicSelector,
    _subsample,
    best_split_info_gain,
    encode_symbols,
    symbols_to_word,
    znormalize,
)


# =========================================================================== SFA
def dft_coefficients(windows: np.ndarray, word_len: int, drop_dc: bool = True) -> np.ndarray:
    """First ``word_len`` real values of the DFT, interleaved [Re1, Im1, Re2, Im2, ...].

    With ``drop_dc`` the DC term (constant 0 after normalisation) is skipped.
    """
    windows = np.asarray(windows, dtype=float)
    n_c = (word_len + 1) // 2
    start = 1 if drop_dc else 0
    F = np.fft.rfft(windows, axis=-1)[..., start : start + n_c]
    if F.shape[-1] < n_c:  # very short windows: pad with zero coefficients
        pad = np.zeros(F.shape[:-1] + (n_c - F.shape[-1],), dtype=F.dtype)
        F = np.concatenate([F, pad], axis=-1)
    out = np.empty(F.shape[:-1] + (2 * n_c,))
    out[..., 0::2] = F.real
    out[..., 1::2] = F.imag
    return out[..., :word_len]


def mcb_edges(coefs: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Multiple Coefficient Binning: equi-depth edges per coefficient. [L, alphabet-1]."""
    coefs = np.asarray(coefs).reshape(-1, coefs.shape[-1])
    qs = np.arange(1, alphabet_size) / alphabet_size
    return np.quantile(coefs, qs, axis=0).T


def sfa_symbols(coefs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Quantise coefficients [..., L] with per-position edges [L, a-1] -> ints [..., L]."""
    out = np.empty(coefs.shape, dtype=np.int64)
    for i in range(coefs.shape[-1]):
        out[..., i] = np.searchsorted(edges[i], coefs[..., i], side="right")
    return out


class SFA:
    """Symbolic Fourier Approximation with MCB bins learnt from training windows."""

    def __init__(self, word_len: int = 8, alphabet_size: int = 4, znorm: bool = True, drop_dc: bool = True):
        self.word_len = int(word_len)
        self.alphabet_size = int(alphabet_size)
        self.znorm = znorm
        self.drop_dc = drop_dc

    def _prep(self, windows):
        windows = np.asarray(windows, dtype=float)
        if self.znorm:
            return znormalize(windows)
        return windows - windows.mean(-1, keepdims=True)

    def coefficients(self, windows):
        return dft_coefficients(self._prep(windows), self.word_len, self.drop_dc)

    def fit(self, windows):
        self.edges_ = mcb_edges(self.coefficients(windows), self.alphabet_size)
        return self

    def set_edges(self, edges):
        self.edges_ = np.asarray(edges, dtype=float)
        return self

    def symbols(self, windows):
        return sfa_symbols(self.coefficients(windows), self.edges_)

    def words(self, windows) -> list[str]:
        s = self.symbols(windows)
        return [symbols_to_word(w) for w in s.reshape(-1, s.shape[-1])]


# =========================================================================== BOSS-ST
class BOSSSTSelector(SymbolicSelector):
    """Top-K SFA words by Information Gain of their per-series relative frequency.

    1. Fit MCB bins on training windows (training split only).
    2. Bag-of-SFA-words per training series (optionally with BOSS numerosity
       reduction: consecutive identical words are counted once).
    3. For every word with support >= ``min_support``, the best-split IG of its
       relative-frequency feature across training series; keep ``top_k``.
    4. ``match``: exact presence, relative frequency and *symbolic* distance
       (min over windows of the mean absolute symbol difference, scaled to [0,1])
       -- no raw-distance verification, which is what makes BOSS-ST cheap.

    Unsupervised (``y=None`` / one class): words ranked by document frequency.
    """

    kind = "bossst"

    def __init__(
        self,
        window: int,
        word_len: int = 8,
        alphabet_size: int = 4,
        top_k: int = 10,
        numerosity_reduction: bool = True,
        min_support: float = 0.02,
        znorm: bool = True,
        step: int = 1,
        max_fit_series: int | None = 500,
        exclude_flat: bool = True,
        seed: int = 0,
    ):
        self.window = int(window)
        self.word_len = int(word_len)
        self.alphabet_size = int(alphabet_size)
        self.top_k = int(top_k)
        self.numerosity_reduction = numerosity_reduction
        self.min_support = float(min_support)
        self.exclude_flat = exclude_flat
        self.step = int(step)
        self.max_fit_series = max_fit_series
        self.seed = seed
        self.sfa = SFA(word_len, alphabet_size, znorm=znorm)

    def _windows(self, X):
        return sliding_window_view(np.asarray(X, float), self.window, axis=1)[:, :: self.step, :]

    def _counts(self, codes: np.ndarray):
        """codes [n, nw] -> per-series (unique codes, counts) with numerosity reduction."""
        out = []
        for row in codes:
            if self.numerosity_reduction:
                keep = np.ones(len(row), dtype=bool)
                keep[1:] = row[1:] != row[:-1]
                row = row[keep]
            out.append(np.unique(row, return_counts=True))
        return out

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        if y is not None:
            y = np.asarray(y)
            if len(np.unique(y)) < 2:
                y = None
        idx = _subsample(len(X), self.max_fit_series, y, rng)
        Xf = X[idx]
        yf = None if y is None else np.searchsorted(np.unique(y), y[idx])
        n = len(Xf)
        win = self._windows(Xf)
        self.sfa.fit(win.reshape(-1, self.window))
        syms = self.sfa.symbols(win)  # [n, nw, L]
        codes = encode_symbols(syms, self.alphabet_size)
        self._fit_rarity(list(codes))

        bags = self._counts(codes)
        vocab = np.unique(np.concatenate([u for u, _ in bags]))
        F = np.zeros((n, len(vocab)), dtype=np.float32)  # relative frequency per series
        for i, (u, c) in enumerate(bags):
            F[i, np.searchsorted(vocab, u)] = c / c.sum()
        support = (F > 0).mean(0)
        # the word every zero-variance window maps to (all-zero DFT) carries no shape
        flat_code = encode_symbols(sfa_symbols(np.zeros((1, self.word_len)), self.sfa.edges_), self.alphabet_size)[0]
        shaped = (vocab != flat_code) if self.exclude_flat else np.ones(len(vocab), dtype=bool)
        flat_syms = syms.reshape(-1, self.word_len)
        _, first = np.unique(codes.ravel(), return_index=True)  # vocab-aligned
        vocab_syms = flat_syms[first]

        if yf is None:
            order = np.lexsort((-F.mean(0), -support))
            order = order[shaped[order]][: self.top_k]
            if len(order) == 0:
                self._syms_sel = np.zeros((0, self.word_len), dtype=np.int64)
                return self._set_empty()
            self.info_gain_ = [0.0] * len(order)
            self.word_class_ = [None] * len(order)
        else:
            C = int(yf.max()) + 1
            if not shaped.any():
                self._syms_sel = np.zeros((0, self.word_len), dtype=np.int64)
                return self._set_empty()
            eligible = np.flatnonzero((support >= self.min_support) & shaped)
            if len(eligible) < self.top_k:
                by_support = np.argsort(-support)
                eligible = by_support[shaped[by_support]][: max(self.top_k, len(eligible))]
            igs = np.zeros(len(eligible))
            gaps = np.zeros(len(eligible))
            for i, v in enumerate(eligible):
                igs[i], _, gaps[i] = best_split_info_gain(F[:, v], yf, C)
            o = np.lexsort((-np.abs(gaps), -igs))[: self.top_k]
            order = eligible[o]
            # dominant class: the class in which the word is most frequent on average
            cls_mean = np.stack([F[yf == c][:, order].mean(0) for c in range(C)])
            self.info_gain_ = [float(igs[i]) for i in o]
            self.word_class_ = [int(c) for c in cls_mean.argmax(0)]
        self.words_ = [symbols_to_word(vocab_syms[v]) for v in order]
        self.word_support_ = [float(support[v]) for v in order]
        self._codes_sel = vocab[order]
        self._syms_sel = vocab_syms[order]
        return self

    def match(self, X: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        win = self._windows(X)
        syms = self.sfa.symbols(win)  # [n, nw, L]
        flat = win.std(-1) < 1e-8  # never report a flat window as a match (unless all are flat)
        all_flat = flat.all(1, keepdims=True)
        excl = flat & ~all_flat
        codes = encode_symbols(syms, self.alphabet_size)
        eq = codes[:, :, None] == self._codes_sel[None, None, :]  # [n, nw, K]
        # symbolic distance: mean |symbol diff| / (alphabet-1), min over windows
        K = len(self._codes_sel)
        dist = np.empty((len(X), K), dtype=np.float32)
        loc = np.empty((len(X), K), dtype=np.int64)
        informative = np.empty((len(X), K), dtype=bool)
        for k in range(K):
            d = np.abs(syms - self._syms_sel[k][None, None, :]).mean(-1) / max(1, self.alphabet_size - 1)
            d = np.where(excl, np.inf, d)
            j = d.argmin(1)
            dist[:, k] = d[np.arange(len(X)), j]
            loc[:, k] = j * self.step
            informative[:, k] = (np.where(excl, -np.inf, d).max(1) > dist[:, k]) & ~all_flat[:, 0]
        r, rj = self._rarity(codes)
        return {
            "presence": eq.any(1).astype(np.float32),
            "frequency": eq.mean(1).astype(np.float32),
            "distance": dist,
            "location": loc,
            "informative": informative,
            "rarity": r.astype(np.float32),
            "rarity_location": rj * self.step,
        }
