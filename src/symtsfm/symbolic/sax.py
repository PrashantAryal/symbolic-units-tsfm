"""FastShapelets-style symbolic path: sliding window -> z-norm -> PAA -> SAX,
random-projection collision table, cheap symbolic filter, real-distance
Information-Gain verification.

Reference: Rakthanmanon & Keogh, "Fast Shapelets: A Scalable Algorithm for
Discovering Time Series Shapelets", SDM 2013.

Everything here is plain numpy/scipy and deterministic given ``seed``. Nothing in
this module knows about foundation models; it only turns raw series into a
fixed set of top-K words plus per-series match statistics (see ``match``).
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

EPS = 1e-8
_MEM_BUDGET_FLOATS = 16_000_000  # ~64MB of float32 per distance chunk


# =========================================================================== basics
def sliding_windows(x: np.ndarray, window: int, step: int = 1):
    """Return (windows [n_windows, window], starts [n_windows]) for a 1-D series."""
    x = np.asarray(x, dtype=float)
    if window > x.shape[-1]:
        raise ValueError(f"window {window} longer than series {x.shape[-1]}")
    w = sliding_window_view(x, window, axis=-1)[..., ::step, :]
    starts = np.arange(0, x.shape[-1] - window + 1, step)
    return w, starts


def znormalize(x: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    """Z-normalise along ``axis`` (population std). Zero-variance input maps to zeros."""
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    flat = sd < eps
    return np.where(flat, 0.0, (x - mu) / np.where(flat, 1.0, sd))


def _paa_matrix(length: int, n_segments: int) -> np.ndarray:
    """(length, n_segments) weights so that ``x @ M`` is PAA, exact for any length."""
    M = np.zeros((length, n_segments))
    for i in range(length):
        lo, hi = i * n_segments, (i + 1) * n_segments  # point i spans [lo, hi) in scaled units
        for j in range(n_segments):
            s_lo, s_hi = j * length, (j + 1) * length
            M[i, j] = max(0, min(hi, s_hi) - max(lo, s_lo))
    return M / length  # each segment averages `length / n_segments` points


def paa(x: np.ndarray, n_segments: int) -> np.ndarray:
    """Piecewise Aggregate Approximation along the last axis."""
    x = np.asarray(x, dtype=float)
    return x @ _paa_matrix(x.shape[-1], n_segments)


def sax_breakpoints(alphabet_size: int) -> np.ndarray:
    """Equiprobable N(0,1) breakpoints, e.g. alphabet 3 -> [-0.4307, 0.4307]."""
    nd = NormalDist()
    return np.array([nd.inv_cdf(i / alphabet_size) for i in range(1, alphabet_size)])


def symbols_to_word(symbols) -> str:
    return "".join(chr(ord("a") + int(s)) for s in symbols)


def sax_symbols(windows: np.ndarray, word_len: int, alphabet_size: int) -> np.ndarray:
    """Integer SAX symbols for windows [..., window] -> [..., word_len]."""
    return np.digitize(paa(znormalize(windows), word_len), sax_breakpoints(alphabet_size))


def sax_word(window, word_len: int, alphabet_size: int) -> str:
    return symbols_to_word(sax_symbols(np.asarray(window, float), word_len, alphabet_size))


def sax_words_for_series(x, window: int, word_len: int, alphabet_size: int, step: int = 1):
    w, starts = sliding_windows(x, window, step)
    syms = sax_symbols(w, word_len, alphabet_size)
    return np.array([symbols_to_word(s) for s in syms]), starts


def encode_symbols(syms: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Pack integer symbols [..., L] into one int64 code per word."""
    base = alphabet_size ** np.arange(syms.shape[-1] - 1, -1, -1, dtype=np.int64)
    return (syms.astype(np.int64) * base).sum(-1)


# =========================================================================== scoring
def entropy(counts: np.ndarray, axis: int = -1) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    tot = counts.sum(axis=axis, keepdims=True)
    p = np.where(tot > 0, counts / np.where(tot > 0, tot, 1), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0).sum(axis=axis)
    return h


def best_split_info_gain(values: np.ndarray, y: np.ndarray, n_classes: int):
    """Best threshold Information Gain of a 1-D feature. Returns (ig, threshold, gap).

    ``gap`` (mean distance between the two sides) is the FastShapelets tie-breaker.
    """
    order = np.argsort(values, kind="mergesort")
    v, yy = values[order], y[order]
    onehot = np.eye(n_classes)[yy]
    left = np.cumsum(onehot, axis=0)[:-1]  # split after position i
    total = onehot.sum(0)
    right = total - left
    n = len(v)
    nl = np.arange(1, n)[:, None]
    h_parent = entropy(total)
    h_split = (nl[:, 0] * entropy(left) + (n - nl[:, 0]) * entropy(right)) / n
    valid = v[1:] > v[:-1]  # only split between distinct values
    if not valid.any():
        return 0.0, float(v[0]), 0.0
    ig = np.where(valid, h_parent - h_split, -np.inf)
    i = int(np.argmax(ig))
    thr = 0.5 * (v[i] + v[i + 1])
    gap = float(v[i + 1 :].mean() - v[: i + 1].mean())
    return float(ig[i]), float(thr), gap


def min_znorm_distance(X: np.ndarray, Q: np.ndarray, step: int = 1, return_spread: bool = False):
    """Length-normalised min z-normalised Euclidean distance of each query to each series.

    X: [n, T] series, Q: [k, w] queries (z-normalised internally).
    Returns dist [n, k] (divided by sqrt(w)) and loc [n, k] (window start in X), plus
    with ``return_spread`` the max-min distance over windows [n, k]: a spread of ~0 means
    every window is equidistant (e.g. an all-flat input), so ``loc`` is arbitrary.
    """
    X = np.asarray(X, dtype=np.float32)
    Q = znormalize(np.asarray(Q, dtype=float)).astype(np.float32)
    k, w = Q.shape
    n, T = X.shape
    win = sliding_window_view(X, w, axis=1)[:, ::step, :]  # view, no copy
    nw = win.shape[1]
    q2 = (Q**2).sum(1)
    dist = np.empty((n, k), dtype=np.float32)
    spread = np.empty((n, k), dtype=np.float32)
    loc = np.empty((n, k), dtype=np.int64)
    b = max(1, _MEM_BUDGET_FLOATS // max(1, nw * w))
    for s in range(0, n, b):
        Wz = znormalize(win[s : s + b]).astype(np.float32)  # [b, nw, w]
        w2 = (Wz**2).sum(-1)  # [b, nw]
        d2 = w2[..., None] + q2[None, None, :] - 2.0 * (Wz @ Q.T)  # [b, nw, k]
        # flat windows (all zeros after z-norm) sit at distance ||q|| from every query:
        # they are never a "match". Only an all-flat input falls back to them (spread 0).
        flat = (w2 < EPS)[..., None]
        all_flat = flat.all(1, keepdims=True)
        d2 = np.where(flat & ~all_flat, np.inf, d2)
        j = d2.argmin(1)
        dist[s : s + b] = np.sqrt(np.maximum(np.take_along_axis(d2, j[:, None, :], 1)[:, 0], 0))
        d2max = np.where(np.isinf(d2), -np.inf, d2).max(1)
        spread[s : s + b] = np.where(all_flat[:, 0], 0.0, np.sqrt(np.maximum(d2max, 0)) - dist[s : s + b])
        loc[s : s + b] = j * step
    if return_spread:
        return dist / np.sqrt(w), loc, spread / np.sqrt(w)
    return dist / np.sqrt(w), loc


def _subsample(n: int, max_n: int | None, y, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified subsample of indices (all indices if max_n is None or >= n)."""
    if max_n is None or n <= max_n:
        return np.arange(n)
    if y is None:
        return np.sort(rng.choice(n, max_n, replace=False))
    idx = []
    classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(classes, counts):
        take = max(1, int(round(max_n * cnt / n)))
        idx.append(rng.choice(np.flatnonzero(y == c), min(take, cnt), replace=False))
    return np.sort(np.concatenate(idx))


# =========================================================================== base
class SymbolicSelector:
    """Shared interface of the FastShapelets and BOSS-ST selectors.

    After ``fit``: ``words_`` (list[str]), ``info_gain_`` (list[float]),
    ``word_class_`` (dominant training class per word, or None when fitted
    unsupervised), ``word_support_`` (fraction of training series containing it).
    ``match(X)`` returns per-series, per-word arrays ``presence``, ``frequency``,
    ``distance``, ``location`` plus per-series ``rarity`` and ``rarity_location``.
    """

    kind = "base"
    window: int

    # --- rarity: -log frequency of a window's word among all training windows
    def _fit_rarity(self, codes_per_series: list[np.ndarray]):
        allc = np.concatenate(codes_per_series)
        uniq, cnt = np.unique(allc, return_counts=True)
        self._rare_codes = uniq
        self._rare_logf = np.log(cnt / cnt.sum())
        self._rare_floor = np.log(1.0 / (cnt.sum() + 1))

    def _rarity(self, codes: np.ndarray):
        """codes: [n, nw] -> rarity max [n], argmax window index [n]."""
        pos = np.searchsorted(self._rare_codes, codes)
        pos = np.clip(pos, 0, len(self._rare_codes) - 1)
        hit = self._rare_codes[pos] == codes
        logf = np.where(hit, self._rare_logf[pos], self._rare_floor)
        r = -logf
        return r.max(1), r.argmax(1)

    def _set_empty(self):
        """A channel with no usable shape (e.g. a constant server metric): select no words."""
        self.words_, self.info_gain_, self.word_class_, self.word_support_ = [], [], [], []
        self._codes_sel = np.zeros(0, dtype=np.int64)
        return self

    def describe(self) -> list[dict]:
        return [
            {
                "word": w,
                "info_gain": float(g),
                "class": None if c is None else int(c),
                "support": float(s),
            }
            for w, g, c, s in zip(self.words_, self.info_gain_, self.word_class_, self.word_support_)
        ]


# =========================================================================== FastShapelets
class FastShapeletSelector(SymbolicSelector):
    """FastShapelets filter-then-verify selection of top-K SAX words.

    1. SAX-encode every sliding window of every (sub-sampled) training series.
    2. ``n_projections`` random projections mask ``mask_size`` word positions;
       for each SAX word count, per class, how many distinct training series
       contain a colliding (projected-equal) word -> collision table.
    3. Cheap symbolic filter: score = max_c (p_c - mean_{c'!=c} p_c') with
       p_c = collisions / (R * n_c); keep ``n_candidates`` best words.
    4. Verify: take each candidate's raw subsequence, compute its real min
       z-normalised distance to every fit series, and score by best-split IG.
       Keep ``top_k`` by (IG, gap).

    With ``y=None`` (or a single class, e.g. anomaly training data that is all
    normal) IG is degenerate, so words are selected by collision support
    ("motifs") instead; ``info_gain_`` is then 0.
    """

    kind = "fastshapelets"

    def __init__(
        self,
        window: int,
        word_len: int = 8,
        alphabet_size: int = 4,
        top_k: int = 10,
        n_projections: int = 10,
        mask_size: int = 2,
        n_candidates: int = 50,
        step: int = 1,
        max_fit_series: int | None = 500,
        exclude_flat: bool = True,
        seed: int = 0,
    ):
        self.window = int(window)
        self.word_len = int(word_len)
        self.alphabet_size = int(alphabet_size)
        self.top_k = int(top_k)
        self.exclude_flat = exclude_flat
        self.n_projections = int(n_projections)
        self.mask_size = int(min(mask_size, word_len - 1))
        self.n_candidates = int(n_candidates)
        self.step = int(step)
        self.max_fit_series = max_fit_series
        self.seed = seed

    # ------------------------------------------------------------------ encode
    def _codes(self, X: np.ndarray):
        win = sliding_window_view(np.asarray(X, float), self.window, axis=1)[:, :: self.step, :]
        syms = sax_symbols(win, self.word_len, self.alphabet_size)  # [n, nw, L]
        return syms, encode_symbols(syms, self.alphabet_size)

    def _flat_windows(self, X):
        win = sliding_window_view(np.asarray(X, float), self.window, axis=1)[:, :: self.step, :]
        return win.std(-1) < EPS

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        if y is not None:
            y = np.asarray(y)
            classes = np.unique(y)
            if len(classes) < 2:
                y = None
        idx = _subsample(len(X), self.max_fit_series, y, rng)
        Xf = X[idx]
        yf = None if y is None else np.searchsorted(np.unique(y), y[idx])
        n = len(Xf)

        syms, codes = self._codes(Xf)  # [n, nw, L], [n, nw]
        self._fit_rarity(list(codes))
        uniq, inv = np.unique(codes, return_inverse=True)
        inv = inv.reshape(codes.shape)
        U = len(uniq)
        # zero-variance windows z-normalise to all zeros: they have no shape, and a flat
        # shapelet is equidistant (distance 1) from every non-flat window. With
        # exclude_flat, such words / representatives are never selected (flatness still
        # reaches the model through the rarity feature and Path A).
        flatw = self._flat_windows(Xf)  # [n, nw]
        nf = np.flatnonzero(~flatw.ravel())
        u_nf, pos = np.unique(inv.ravel()[nf], return_index=True)
        has_shape = np.zeros(U, dtype=bool)
        has_shape[u_nf] = True
        flat_code = encode_symbols(np.digitize(np.zeros(self.word_len), sax_breakpoints(self.alphabet_size)),
                                   self.alphabet_size)
        eligible = has_shape & (uniq != flat_code) if self.exclude_flat else np.ones(U, dtype=bool)
        # series x word presence (binary)
        rows = np.repeat(np.arange(n), codes.shape[1])
        P = sparse.csr_matrix((np.ones(rows.size), (rows, inv.ravel())), shape=(n, U))
        P.sum_duplicates()
        P.data[:] = 1.0
        # symbols of every unique word, via its first occurrence (flat index s * nw + j)
        _, first = np.unique(inv.ravel(), return_index=True)
        uniq_syms = syms.reshape(-1, self.word_len)[first]

        C = 1 if yf is None else int(yf.max()) + 1
        Y = np.zeros((n, C))
        Y[np.arange(n), 0 if yf is None else yf] = 1.0
        coll = np.zeros((U, C))
        for _ in range(self.n_projections):
            keep = np.sort(rng.choice(self.word_len, self.word_len - self.mask_size, replace=False))
            pk = encode_symbols(uniq_syms[:, keep], self.alphabet_size)
            pu, pinv = np.unique(pk, return_inverse=True)
            M = sparse.csr_matrix((np.ones(U), (np.arange(U), pinv)), shape=(U, len(pu)))
            Pk = (P @ M).tocsr()
            Pk.data[:] = 1.0  # series contains >=1 word with this projected key
            per_key = np.asarray((Pk.T @ Y))  # [keys, C] distinct series per class
            coll += per_key[pinv]
        n_c = Y.sum(0)
        p = coll / (self.n_projections * n_c[None, :])  # [U, C] in [0, 1]
        support = np.asarray(P.sum(0)).ravel() / n

        nw = codes.shape[1]
        first_shaped = first.copy()  # first occurrence with non-zero variance, if any
        first_shaped[u_nf] = nf[pos]
        if C == 1:
            order = np.lexsort((-support, -p[:, 0]))
            sel = order[eligible[order]][: self.top_k]
            if len(sel) == 0:
                self.shapelets_ = np.zeros((0, self.window))
                return self._set_empty()
            self.words_ = [symbols_to_word(uniq_syms[u]) for u in sel]
            self.info_gain_ = [0.0] * len(sel)
            self.word_class_ = [None] * len(sel)
            self.word_support_ = [float(support[u]) for u in sel]
            self.shapelets_ = self._raw_subsequences(Xf, first_shaped[sel], nw)
            self._codes_sel = uniq[sel]
            return self

        if C == 2:
            score = np.abs(p[:, 0] - p[:, 1])
        else:
            others = (p.sum(1, keepdims=True) - p) / (C - 1)
            score = (p - others).max(1)
        score = np.where(eligible, score, -np.inf)
        cand = np.argsort(-score, kind="mergesort")[: min(self.n_candidates, int(eligible.sum()))]
        if len(cand) == 0:
            self.shapelets_ = np.zeros((0, self.window))
            return self._set_empty()
        # representative occurrence of each candidate: first (non-flat) occurrence in a
        # series of its dominant class (FastShapelets verifies the SAX object's raw subsequence).
        dom = p[cand].argmax(1)
        reps = []
        Pc = P.tocsc()
        for u, c in zip(cand, dom):
            series_with = Pc.indices[Pc.indptr[u] : Pc.indptr[u + 1]]
            in_c = series_with[yf[series_with] == c]
            rep = int(first_shaped[u])
            for s in (in_c if len(in_c) else series_with):
                js = np.flatnonzero((inv[s] == u) & ~flatw[s])
                if len(js):
                    rep = int(s) * nw + int(js[0])
                    break
            reps.append(rep)
        raw = self._raw_subsequences(Xf, np.array(reps), nw)
        dist, _ = min_znorm_distance(Xf, raw, self.step)
        igs, gaps = np.zeros(len(cand)), np.zeros(len(cand))
        for i in range(len(cand)):
            igs[i], _, gaps[i] = best_split_info_gain(dist[:, i], yf, C)
        order = np.lexsort((-np.abs(gaps), -igs))[: self.top_k]
        sel = cand[order]
        self.words_ = [symbols_to_word(uniq_syms[u]) for u in sel]
        self.info_gain_ = [float(igs[o]) for o in order]
        self.word_class_ = [int(dom[o]) for o in order]
        self.word_support_ = [float(support[u]) for u in sel]
        self.shapelets_ = raw[order]
        self._codes_sel = uniq[sel]
        return self

    def _raw_subsequences(self, X, flat_occ, nw):
        s, j = np.divmod(np.asarray(flat_occ), nw)
        starts = j * self.step
        return np.stack([znormalize(X[a, b : b + self.window]) for a, b in zip(s, starts)])

    # ------------------------------------------------------------------ match
    def match(self, X: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        _, codes = self._codes(X)  # [n, nw]
        eq = codes[:, :, None] == self._codes_sel[None, None, :]  # [n, nw, K]
        dist, loc, spread = min_znorm_distance(X, self.shapelets_, self.step, return_spread=True)
        r, rj = self._rarity(codes)
        return {
            "presence": eq.any(1).astype(np.float32),
            "frequency": eq.mean(1).astype(np.float32),
            "distance": dist.astype(np.float32),
            "location": loc,
            "informative": spread > 1e-4,
            "rarity": r.astype(np.float32),
            "rarity_location": rj * self.step,
        }
