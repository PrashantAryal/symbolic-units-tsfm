"""Contrastive symbolic rules for auditing a frozen time-series foundation model.

This module deliberately *does not alter* a foundation-model prediction.  It
mines rules over a validation-only behavioural target (for example, a high
forecast-error regime) and asks whether the same symbolic events predict that
behaviour on the held-out test set.  This avoids the post-hoc forecast
calibration confound exposed by the Phase 4 controls.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


@dataclass(frozen=True)
class ContrastRule:
    antecedent: str
    support: int
    positive_support: int
    confidence: float
    growth: float
    score: float


def _binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Held-out rule quality; returns NA rather than fabricating a PR-AUC."""
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    pred = score >= threshold
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    ap = float("nan") if len(np.unique(y)) < 2 else float(average_precision_score(y, score))
    return {"precision": float(p), "recall": float(r), "f1": float(f), "pr_auc": ap,
            "positive_rate": float(y.mean()), "predicted_positive_rate": float(pred.mean())}


class ContrastiveBehaviorRules:
    """Mine symbolic events enriched for a foundation-model behaviour.

    A rule is retained only when its event occurs sufficiently often and is
    more prevalent in the positive behavioural regime than in the negative
    regime.  FastShapelets/BOSS-ST events are retained explicitly; ordered SAX
    n-grams provide temporal context.  The `permuted_target_control` preserves
    event counts and outcome prevalence while breaking their correspondence.
    """

    def __init__(self, *, ordered_words: int = 24, ngram: int = 3,
                 miner_events_per_channel: int = 2,
                 activation_quantile: float = 0.75, min_support: int = 4,
                 min_confidence: float = 0.55, min_growth: float = 1.25,
                 support_shrinkage: float = 8.0, max_rules_per_query: int = 8,
                 seed: int = 0):
        self.ordered_words = int(ordered_words)
        self.ngram = int(ngram)
        self.miner_events_per_channel = int(miner_events_per_channel)
        self.activation_quantile = float(activation_quantile)
        self.min_support = int(min_support)
        self.min_confidence = float(min_confidence)
        self.min_growth = float(min_growth)
        self.support_shrinkage = float(support_shrinkage)
        self.max_rules_per_query = int(max_rules_per_query)
        self.seed = int(seed)
        if self.ordered_words < self.ngram or self.ngram < 2:
            raise ValueError("ordered_words must be >= ngram >= 2")

    def fit_event_vocabulary(self, train_features: np.ndarray) -> None:
        x = np.asarray(train_features, dtype=np.float32)
        if x.ndim != 3 or x.shape[-1] < self.ordered_words:
            raise ValueError("expected [windows, channels, features] with ordered SAX suffix")
        self.miner_dim_ = x.shape[-1] - self.ordered_words
        self.miner_words_ = max(0, (self.miner_dim_ - 1) // 3)
        if self.miner_words_ == 0 or self.miner_events_per_channel == 0:
            self.activation_threshold_ = None
            return
        a = x[..., :3 * self.miner_words_].reshape(len(x), x.shape[1], self.miner_words_, 3)
        # Presence/frequency are positive evidence; small distance is a match.
        a = a[..., 0] + a[..., 1] - a[..., 2]
        self.activation_threshold_ = np.quantile(a, self.activation_quantile, axis=0)

    def events(self, features: np.ndarray) -> list[set[str]]:
        x = np.asarray(features, dtype=np.float32)
        words = np.rint(x[..., -self.ordered_words:]).astype(np.int16)
        activation = None
        if self.miner_words_ and self.miner_events_per_channel:
            a = x[..., :3 * self.miner_words_].reshape(len(x), x.shape[1], self.miner_words_, 3)
            activation = a[..., 0] + a[..., 1] - a[..., 2]
        answer: list[set[str]] = []
        for i, row in enumerate(words):
            ev: set[str] = set()
            for ch, sequence in enumerate(row):
                # A small set of suffix n-grams retains temporal order but
                # avoids one almost-unique location token per input window.
                for end in (self.ngram, len(sequence) // 2 + self.ngram, len(sequence)):
                    end = min(len(sequence), max(self.ngram, end))
                    loc = "recent" if end == len(sequence) else "context"
                    gram = ".".join(str(int(v)) for v in sequence[end - self.ngram:end])
                    ev.add(f"ch={ch}|loc={loc}|sax={gram}")
                if activation is None:
                    continue
                recent = ".".join(str(int(v)) for v in sequence[-self.ngram:])
                for word in np.argsort(-activation[i, ch])[:self.miner_events_per_channel]:
                    if activation[i, ch, word] < self.activation_threshold_[ch, word]:
                        continue
                    token = f"ch={ch}|miner_word={int(word)}|active"
                    ev.add(token)
                    ev.add(f"ch={ch}|loc=recent|sax={recent}&{token}")
            answer.append(ev)
        return answer

    def _fit_rules(self, event_rows: list[set[str]], target: np.ndarray) -> None:
        y = np.asarray(target, dtype=np.int8).reshape(-1)
        if len(event_rows) != len(y):
            raise ValueError("event rows and target must have equal length")
        pos_n, neg_n = int(y.sum()), int(len(y) - y.sum())
        counts: dict[str, Counter] = defaultdict(Counter)
        for ev, label in zip(event_rows, y):
            for token in ev:
                counts[token][int(label)] += 1
        rules: dict[str, ContrastRule] = {}
        for token, c in counts.items():
            pos, neg = int(c[1]), int(c[0])
            support = pos + neg
            if support < self.min_support or pos_n == 0 or neg_n == 0:
                continue
            confidence = pos / support
            # Smoothed growth is stable for rare but nonzero negative counts.
            growth = ((pos + 0.5) / (pos_n + 1.0)) / ((neg + 0.5) / (neg_n + 1.0))
            if confidence < self.min_confidence or growth < self.min_growth:
                continue
            score = confidence * log(growth) * support / (support + self.support_shrinkage)
            rules[token] = ContrastRule(token, support, pos, confidence, growth, float(score))
        self.rules_ = rules
        self.discovery_positive_rate_ = float(y.mean())

    def _score(self, event_rows: list[set[str]]) -> tuple[np.ndarray, list[list[dict]]]:
        score = np.zeros(len(event_rows), dtype=np.float32)
        matches: list[list[dict]] = []
        for i, ev in enumerate(event_rows):
            selected = sorted((self.rules_[t] for t in ev if t in self.rules_),
                              key=lambda r: (-r.score, r.antecedent))[:self.max_rules_per_query]
            if selected:
                # Max evidence avoids correlated SAX n-grams making every
                # input artificially high confidence.
                score[i] = selected[0].score
            matches.append([{"antecedent": r.antecedent, "support": r.support,
                             "positive_support": r.positive_support,
                             "confidence": r.confidence, "growth": r.growth,
                             "rule_score": r.score} for r in selected])
        return score, matches

    @staticmethod
    def _threshold(scores: np.ndarray, target: np.ndarray) -> float:
        """Choose an F1 threshold on discovery only, including abstention."""
        s, y = np.asarray(scores), np.asarray(target, dtype=np.int8)
        candidates = np.unique(np.r_[0.0, s])
        best = (-1.0, float("inf"))
        selected = float("inf")
        for t in candidates:
            m = _binary_metrics(y, s, float(t))
            key = (m["f1"], -float(t))
            if key > best:
                best, selected = key, float(t)
        return selected

    def fit_apply(self, train_features: np.ndarray, discovery_features: np.ndarray,
                  discovery_target: np.ndarray, test_features: np.ndarray,
                  test_target: np.ndarray, *, behavior: str) -> dict:
        self.fit_event_vocabulary(train_features)
        discover_events = self.events(discovery_features)
        test_events = self.events(test_features)
        y_discovery = np.asarray(discovery_target, dtype=np.int8).reshape(-1)
        y_test = np.asarray(test_target, dtype=np.int8).reshape(-1)
        self._fit_rules(discover_events, y_discovery)
        discovery_score, _ = self._score(discover_events)
        threshold = self._threshold(discovery_score, y_discovery)
        test_score, test_matches = self._score(test_events)
        metrics = _binary_metrics(y_test, test_score, threshold)

        # Required falsification: labels are permuted only while fitting, then
        # assessed against the true held-out behavioural target.
        rng = np.random.default_rng(self.seed + 9176)
        saved = self.rules_
        self._fit_rules(discover_events, y_discovery[rng.permutation(len(y_discovery))])
        perm_discovery_score, _ = self._score(discover_events)
        perm_threshold = self._threshold(perm_discovery_score, y_discovery)
        perm_test_score, _ = self._score(test_events)
        perm_metrics = _binary_metrics(y_test, perm_test_score, perm_threshold)
        perm_n_rules = len(self.rules_)
        self.rules_ = saved

        top_rules = sorted(saved.values(), key=lambda r: (-r.score, r.antecedent))[:20]
        return {
            "summary": {
                "policy": "task_conditioned_contrastive_symbolic_behavior_audit",
                "behavior": behavior,
                "discovery_split": "validation_only",
                "test_target_never_used_for_mining_or_threshold_selection": True,
                "n_rules": len(saved),
                "miner_events_per_channel": self.miner_events_per_channel,
                "ordered_sax_words": self.ordered_words,
                "discovery_positive_rate": float(y_discovery.mean()),
                "test_rule_coverage": float((test_score > 0).mean()),
                "selected_threshold": threshold,
                "held_out_behavior_metrics": metrics,
                "permuted_behavior_target_control": {"n_rules": perm_n_rules,
                                                      "held_out_metrics": perm_metrics},
            },
            "preview": {"query_split": "test", "top_rules": [r.__dict__ for r in top_rules],
                        "test_rule_scores": test_score[:64].tolist(),
                        "test_rule_matches": test_matches[:64]},
        }
