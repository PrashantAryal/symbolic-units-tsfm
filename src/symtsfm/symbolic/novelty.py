"""Normal-dictionary symbolic novelty for anomaly detection (Phase 10).

Training data in the anomaly benchmarks is (nominally) normal.  For every
channel we build the vocabulary of symbolic words of *all* sliding
subsequences of the training windows:

  * ``representation="sax"``  -- SAX words, the representation FastShapelets
    mines (HOT-SAX style discord scoring);
  * ``representation="sfa"``  -- SFA words with training-fitted MCB bins, the
    representation BOSS / BOSS-ST mines.

A test subsequence is scored by

    novelty = -log((count(word) + 1) / (N + V + 1))  +  distance_weight * d_min(word)

i.e. symbolic surprise under the normal vocabulary plus the symbolic distance
to the nearest normal word (SAX MINDIST cell table / mean SFA symbol gap).
Seen words have ``d_min = 0``.  A point's score is the maximum novelty of the
subsequences that cover it.  Per-channel scores are robustly standardised on
held-out *normal validation* windows and reduced over channels with a fixed
top-k mean (k = ceil(sqrt(C))), chosen a priori and never tuned on test labels.

Near-constant subsequences (std < ``flat_std``) map to one dedicated FLAT word
instead of z-normalised noise, so a stuck sensor is novel only if the channel
was rarely flat in training.

Every score is traceable: :meth:`explain_point` returns the channel, the exact
subsequence span, its word, its training count and the nearest normal word.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from symtsfm.symbolic.sax import encode_symbols, paa, sax_breakpoints, symbols_to_word, znormalize
from symtsfm.symbolic.sfa import SFA

FLAT_CODE = -1
_PAIR_BUDGET = 4_000_000  # max (unseen word x vocabulary x word_len) cells per distance chunk


def _windows(x: np.ndarray, window: int, step: int) -> np.ndarray:
    """[..., T] -> [..., n_windows, window] view."""
    return sliding_window_view(np.asarray(x, dtype=np.float64), window, axis=-1)[..., ::step, :]


class SymbolicNoveltyScorer:
    def __init__(self, representation: str = "sax", window: int = 24, word_len: int = 8,
                 alphabet_size: int = 4, step: int = 1, distance_weight: float = 4.0,
                 flat_std: float = 1e-3, max_fit_windows: int = 200_000, scale_floor: float = 0.5,
                 channel_top_k: int | None = None, max_nearest_vocab: int = 2048, seed: int = 0):
        if representation not in ("sax", "sfa"):
            raise ValueError("representation must be 'sax' (FastShapelets) or 'sfa' (BOSS-ST)")
        self.representation = representation
        self.window = int(window)
        self.word_len = int(word_len)
        self.alphabet_size = int(alphabet_size)
        self.step = int(step)
        self.distance_weight = float(distance_weight)
        self.flat_std = float(flat_std)
        self.max_fit_windows = int(max_fit_windows)
        self.scale_floor = float(scale_floor)
        self.channel_top_k = channel_top_k
        # Nearest-word search runs against the most frequent normal words only:
        # "distance to the nearest *common* normal pattern" (bounded cost on SMD-size data).
        self.max_nearest_vocab = int(max_nearest_vocab)
        self.seed = int(seed)
        if representation == "sax":
            beta = sax_breakpoints(self.alphabet_size)
            a = self.alphabet_size
            cell = np.zeros((a, a))
            for r in range(a):
                for c in range(a):
                    if abs(r - c) > 1:
                        cell[r, c] = beta[max(r, c) - 1] - beta[min(r, c)]
            self._cell = cell

    # ------------------------------------------------------------------ encoding
    def _symbols(self, channel: int, win: np.ndarray) -> np.ndarray:
        if self.representation == "sax":
            return np.digitize(paa(znormalize(win), self.word_len), sax_breakpoints(self.alphabet_size))
        return self.sfa_[channel].symbols(win)

    def _encode(self, channel: int, win: np.ndarray):
        """win [m, w] -> codes [m] (FLAT_CODE for near-constant windows), symbols [m, L]."""
        flat = win.std(-1) < self.flat_std
        syms = np.zeros((len(win), self.word_len), dtype=np.int64)
        if (~flat).any():
            syms[~flat] = self._symbols(channel, win[~flat])
        codes = encode_symbols(syms, self.alphabet_size)
        codes[flat] = FLAT_CODE
        return codes, syms

    # ------------------------------------------------------------------ fit
    def fit(self, X):
        """Normal training data: windows [n, C, T] or a list of contiguous series [C, T_i]."""
        series = [np.asarray(s, dtype=np.float64) for s in X]
        C = series[0].shape[0]
        self.window_ = int(min(self.window, max(s.shape[1] for s in series)))
        self.n_channels_ = C
        self.top_k_ = int(self.channel_top_k or math.ceil(math.sqrt(C)))
        rng = np.random.default_rng(self.seed)
        self.sfa_ = {}
        self.vocab_codes_, self.vocab_counts_, self.vocab_syms_, self.totals_ = [], [], [], []
        self.near_index_, self._near_memo = [], []
        for c in range(C):
            win = np.concatenate([_windows(s[c], self.window_, self.step).reshape(-1, self.window_)
                                  for s in series if s.shape[1] >= self.window_])
            if len(win) > self.max_fit_windows:
                win = win[np.sort(rng.choice(len(win), self.max_fit_windows, replace=False))]
            if self.representation == "sfa":
                shaped = win[win.std(-1) >= self.flat_std]
                self.sfa_[c] = SFA(self.word_len, self.alphabet_size).fit(shaped if len(shaped) else win)
            codes, syms = self._encode(c, win)
            uniq, first, cnt = np.unique(codes, return_index=True, return_counts=True)
            self.vocab_codes_.append(uniq)
            self.vocab_counts_.append(cnt.astype(np.int64))
            self.vocab_syms_.append(syms[first])
            self.totals_.append(int(cnt.sum()))
            shaped = np.flatnonzero(uniq != FLAT_CODE)
            by_count = shaped[np.argsort(-cnt[shaped], kind="stable")]
            self.near_index_.append(by_count[: self.max_nearest_vocab])
            self._near_memo.append({})
        return self

    # ------------------------------------------------------------------ scoring
    def _word_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pairwise symbolic distance, a [u, L], b [v, L] -> [u, v]."""
        if self.representation == "sax":
            d = self._cell[a[:, None, :], b[None, :, :]]
            return np.sqrt((d ** 2).mean(-1))
        return np.abs(a[:, None, :] - b[None, :, :]).mean(-1) / max(1, self.alphabet_size - 1)

    def _nearest(self, channel: int, syms: np.ndarray, codes: np.ndarray):
        """Nearest common normal word for each unique query word -> (distance [u], vocab index [u])."""
        shaped = self.near_index_[channel]
        memo = self._near_memo[channel]
        dist = np.full(len(syms), np.inf)
        nearest = np.full(len(syms), -1, dtype=np.int64)
        if len(shaped) == 0 or len(syms) == 0:
            return dist, nearest
        todo = np.array([i for i, c in enumerate(codes) if int(c) not in memo and c != FLAT_CODE], dtype=np.int64)
        vs = self.vocab_syms_[channel][shaped]
        chunk = max(1, _PAIR_BUDGET // max(1, len(vs) * self.word_len))
        for s in range(0, len(todo), chunk):
            q = todo[s:s + chunk]
            d = self._word_distance(syms[q], vs)
            j = d.argmin(1)
            for qi, dj, jj in zip(q, d[np.arange(len(j)), j], shaped[j]):
                memo[int(codes[qi])] = (float(dj), int(jj))
        for i, c in enumerate(codes):
            if c == FLAT_CODE:  # a flat query has no shape: distance is not defined
                dist[i] = 0.0
            else:
                dist[i], nearest[i] = memo[int(c)]
        return dist, nearest

    def window_novelty(self, channel: int, win: np.ndarray, return_detail: bool = False):
        """Novelty of subsequences win [m, w] of one channel."""
        codes, syms = self._encode(channel, win)
        vocab = self.vocab_codes_[channel]
        counts = np.zeros(len(codes), dtype=np.int64)
        if len(vocab):
            pos = np.clip(np.searchsorted(vocab, codes), 0, len(vocab) - 1)
            hit = vocab[pos] == codes
            counts[hit] = self.vocab_counts_[channel][pos[hit]]
        denom = self.totals_[channel] + len(vocab) + 1
        surprise = -np.log((counts + 1.0) / denom)
        distance = np.zeros(len(codes))
        nearest = np.full(len(codes), -1, dtype=np.int64)
        unseen = counts == 0
        if unseen.any():
            u_codes, inv = np.unique(codes[unseen], return_inverse=True)
            _, first = np.unique(codes[unseen], return_index=True)
            u_syms = syms[unseen][first]
            d, j = self._nearest(channel, u_syms, u_codes)
            d = np.where(np.isfinite(d), d, 0.0)
            distance[unseen] = d[inv]
            nearest[unseen] = j[inv]
        novelty = surprise + self.distance_weight * distance
        if not return_detail:
            return novelty
        return novelty, {"codes": codes, "symbols": syms, "train_count": counts,
                         "distance": distance, "nearest_vocab_index": nearest}

    def channel_point_scores(self, channel: int, series: np.ndarray) -> np.ndarray:
        """Point score [T] = max novelty of the subsequences covering each point."""
        series = np.asarray(series, dtype=np.float64)
        T, w = len(series), self.window_
        if T < w:
            return np.zeros(T)
        nov = self.window_novelty(channel, _windows(series, w, self.step))
        return self._cover_max(nov[None, :], T)[0]

    def _cover_max(self, nov: np.ndarray, T: int) -> np.ndarray:
        """Window novelty [n, n_windows] -> point score [n, T] (max over covering windows)."""
        w = self.window_
        starts = np.arange(0, T - w + 1, self.step)
        full = np.full((len(nov), T), -np.inf)
        full[:, starts] = nov
        padded = np.concatenate([np.full((len(nov), w - 1), -np.inf), full], axis=1)
        out = sliding_window_view(padded, w, axis=1).max(-1)
        return np.where(np.isfinite(out), out, 0.0)

    def raw_scores(self, S: np.ndarray) -> np.ndarray:
        """S [C, T] -> per-channel raw point novelty [C, T]."""
        return np.stack([self.channel_point_scores(c, S[c]) for c in range(S.shape[0])])

    # ------------------------------------------------------------------ normalisation
    def fit_normalizer(self, X_val: np.ndarray):
        """Robust per-channel location/scale from held-out normal windows X_val [n, C, T]."""
        X_val = np.asarray(X_val, dtype=np.float64)
        n, C, T = X_val.shape
        per = []
        for c in range(C):
            win = _windows(X_val[:, c], self.window_, self.step)  # [n, nw, w]
            nov = self.window_novelty(c, win.reshape(-1, self.window_)).reshape(n, -1)
            per.append(self._cover_max(nov, T).ravel())
        return self._set_normalizer(per)

    def fit_normalizer_heldout(self, segments):
        """Normaliser from held-out contiguous normal segments.

        ``segments``: list of (S [C, T], first_heldout_index).  Points from
        ``first_heldout_index`` on were excluded from the vocabulary; they are
        scored in context, so every subsequence ending in the held-out part counts.
        """
        per = [[] for _ in range(self.n_channels_)]
        for S, cut in segments:
            S = np.asarray(S, dtype=np.float64)
            lo = max(0, int(cut) - self.window_ + 1)
            raw = self.raw_scores(S[:, lo:])[:, int(cut) - lo:]
            for c in range(self.n_channels_):
                per[c].append(raw[c])
        return self._set_normalizer([np.concatenate(p) for p in per])

    def _set_normalizer(self, per):
        """``per``: per-channel held-out normal point scores, aligned across channels."""
        self.loc_ = np.array([np.median(p) for p in per])
        iqr = np.array([np.subtract(*np.percentile(p, [75, 25])) for p in per])
        self.scale_ = np.maximum(iqr, self.scale_floor)
        # Held-out normal distribution of the channel-reduced score: the reference
        # for calibrated (tail-probability) fusion.
        z = (np.stack(per) - self.loc_[:, None]) / self.scale_[:, None]
        k = min(self.top_k_, z.shape[0])
        self.reference_ = np.sort(np.sort(z, axis=0)[-k:].mean(0))
        return self

    def standardized(self, S: np.ndarray):
        """S [C, T] -> (per-channel z [C, T], channel-reduced symbolic score [T])."""
        z = (self.raw_scores(S) - self.loc_[:, None]) / self.scale_[:, None]
        k = min(self.top_k_, z.shape[0])
        reduced = np.sort(z, axis=0)[-k:].mean(0)
        return z, reduced

    # ------------------------------------------------------------------ explanation
    def word_string(self, channel: int, code: int, syms: np.ndarray) -> str:
        return "<flat>" if code == FLAT_CODE else symbols_to_word(syms)

    def explain_point(self, S: np.ndarray, t: int, channel: int) -> dict:
        """Which subsequence of ``channel`` makes point ``t`` novel, and why."""
        w = self.window_
        T = S.shape[1]
        starts = np.arange(0, max(0, T - w + 1), self.step)
        starts = starts[(starts >= t - w + 1) & (starts <= t)]
        if len(starts) == 0:
            return {"channel": int(channel), "start": int(max(0, t - w + 1)), "end": int(min(T, t + 1))}
        win = np.stack([S[channel, s:s + w] for s in starts]).astype(np.float64)
        nov, det = self.window_novelty(channel, win, return_detail=True)
        i = int(np.argmax(nov))
        j = int(det["nearest_vocab_index"][i])
        code = int(det["codes"][i])
        nearest_word = None
        if j >= 0:
            nearest_word = self.word_string(channel, int(self.vocab_codes_[channel][j]), self.vocab_syms_[channel][j])
        return {
            "channel": int(channel),
            "start": int(starts[i]),
            "end": int(starts[i] + w),
            "representation": self.representation,
            "word": self.word_string(channel, code, det["symbols"][i]),
            "train_count": int(det["train_count"][i]),
            "train_total": int(self.totals_[channel]),
            "seen_in_training": bool(det["train_count"][i] > 0),
            "nearest_normal_word": nearest_word,
            "symbolic_distance_to_nearest_normal": float(det["distance"][i]),
            "novelty": float(nov[i]),
        }

    def summary(self) -> dict:
        return {
            "representation": self.representation,
            "miner_family": "FastShapelets (SAX)" if self.representation == "sax" else "BOSS-ST (SFA)",
            "window": self.window_, "word_len": self.word_len, "alphabet_size": self.alphabet_size,
            "step": self.step, "distance_weight": self.distance_weight, "flat_std": self.flat_std,
            "channel_top_k": self.top_k_, "scale_floor": self.scale_floor,
            "vocabulary_size_per_channel": [int(len(v)) for v in self.vocab_codes_],
            "train_subsequences_per_channel": [int(t) for t in self.totals_],
        }


def tail_surprisal(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """-log of the empirical tail probability of ``x`` under held-out normal scores.

    q = -log((1 + #{ref >= x}) / (n + 1)) is a conformal p-value on the log scale,
    bounded by log(n + 1).  Beyond the largest reference value, ranking is kept by a
    log-compressed excess over that maximum (in units of the reference IQR), so a
    heavy-tailed stream cannot dominate another stream by orders of magnitude.
    """
    ref = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    x = np.asarray(x, dtype=np.float64)
    n = len(ref)
    ge = n - np.searchsorted(ref, x, side="left")
    q = -np.log((1.0 + ge) / (n + 1.0))
    q75, q25 = np.percentile(ref, [75, 25])
    s = q75 - q25
    if s <= 1e-12:
        s = ref.std() if ref.std() > 1e-12 else 1.0
    return q + np.log1p(np.maximum(0.0, x - ref[-1]) / s)


def estimate_period(series: list, max_lag: int = 512, min_acf: float = 0.3,
                    max_points: int = 20_000) -> int | None:
    """Dominant period of normal training data (label-free).

    First autocorrelation peak after the first zero crossing, per channel series;
    returns the median accepted period, or None when no channel is clearly periodic.
    This is the usual discord-discovery choice of subsequence length (one cycle).
    """
    periods = []
    for s in series:
        for x in np.atleast_2d(np.asarray(s, dtype=np.float64)):
            x = x[:max_points] - x[:max_points].mean()
            if len(x) < 4 * 16 or x.std() < 1e-8:
                continue
            n = 1 << int(np.ceil(np.log2(2 * len(x))))
            f = np.fft.rfft(x, n)
            ac = np.fft.irfft(f * np.conj(f), n)[: min(max_lag + 1, len(x) // 2)]
            ac = ac / ac[0]
            neg = np.flatnonzero(ac < 0)
            if not len(neg):
                continue
            lag = int(neg[0] + np.argmax(ac[neg[0]:]))
            if ac[lag] >= min_acf:
                periods.append(lag)
    return int(np.median(periods)) if periods else None


def top_events(score: np.ndarray, n: int, min_separation: int) -> list[int]:
    """Greedy non-maximum suppression: indices of the ``n`` highest separated peaks."""
    order = np.argsort(-np.asarray(score), kind="stable")
    chosen: list[int] = []
    for t in order:
        if all(abs(int(t) - c) >= min_separation for c in chosen):
            chosen.append(int(t))
            if len(chosen) >= n:
                break
    return chosen
