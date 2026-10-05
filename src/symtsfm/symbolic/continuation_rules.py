"""Training-only symbolic continuation rules for forecasting.

This is deliberately different from a generic symbolic kNN feature fusion.
The recent ordered SAX states are converted into human-readable antecedents;
training futures are clustered into continuation states; and only rules
``antecedent -> continuation state`` that meet support/confidence/lift criteria
may contribute a selective forecast correction.

The public entry point never accepts test targets.  All vocabulary, future
states, rule counts and calibration are learned from train/validation inputs
only.  A query with no reliable matching rule receives the base forecast.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class ContinuationRule:
    """A mined rule with an interpretable symbolic antecedent."""

    antecedent: str
    future_state: int
    support: int
    confidence: float
    lift: float


@dataclass
class RuleEvidenceResult:
    output: np.ndarray
    selected_weight: float
    summary: dict
    preview: dict
    counterfactuals: dict[str, np.ndarray]


class SymbolicContinuationRules:
    """Mine and apply ordered symbolic prefix-to-future rules.

    Parameters are deliberately conservative.  The miner uses suffix n-grams
    from the recent ordered SAX descriptor of each channel, rather than a
    BOSS-style unordered histogram.  The descriptor must therefore have the
    ordered SAX words appended as its final feature positions.
    """

    def __init__(self, *, ordered_words: int = 24, ngram: int = 3,
                 n_prototypes: int = 8, min_support: int = 8,
                 min_confidence: float = 0.20, min_lift: float = 1.05,
                 support_shrinkage: float = 12.0,
                 miner_events_per_channel: int = 2,
                 miner_activation_quantile: float = 0.75,
                 max_rules_per_query: int = 8,
                 weights=(0.0, 0.1, 0.25, 0.5, 1.0),
                 mae_tolerance: float = 0.0, mae_relative_tolerance: float = 0.0,
                 permuted_future_state_control: bool = False,
                 seed: int = 0):
        self.ordered_words = int(ordered_words)
        self.ngram = int(ngram)
        self.n_prototypes = int(n_prototypes)
        self.min_support = int(min_support)
        self.min_confidence = float(min_confidence)
        self.min_lift = float(min_lift)
        self.support_shrinkage = float(support_shrinkage)
        self.miner_events_per_channel = int(miner_events_per_channel)
        self.miner_activation_quantile = float(miner_activation_quantile)
        self.max_rules_per_query = int(max_rules_per_query)
        self.weights = tuple(float(w) for w in weights)
        self.mae_tolerance = float(mae_tolerance)
        self.mae_relative_tolerance = float(mae_relative_tolerance)
        self.permuted_future_state_control = bool(permuted_future_state_control)
        self.seed = int(seed)
        if self.ordered_words < self.ngram or self.ngram < 2:
            raise ValueError("ordered_words must be >= ngram >= 2")

    def _fit_miner_thresholds(self, features: np.ndarray) -> None:
        """Fit activation cutoffs for the selected FastShapelets/BOSS words.

        The first `3*K+1` dimensions are the established miner features:
        presence, frequency, distance for each selected word, plus rarity.
        Ordered SAX context is appended after them by the runner.  This is the
        point at which FastShapelets and BOSS-ST now genuinely differ.
        """
        x = np.asarray(features, dtype=np.float32)
        self.miner_dim_ = x.shape[-1] - self.ordered_words
        self.miner_words_ = max(0, (self.miner_dim_ - 1) // 3)
        if self.miner_words_ == 0:
            self.miner_threshold_ = None
            return
        base = x[..., :self.miner_dim_]
        # Higher presence/frequency and lower distance imply a more salient
        # match.  Thresholds are learned only on the training split.
        activation = base[..., :3 * self.miner_words_].reshape(
            len(base), base.shape[1], self.miner_words_, 3)
        activation = activation[..., 0] + activation[..., 1] - activation[..., 2]
        self.miner_threshold_ = np.quantile(activation, self.miner_activation_quantile, axis=0)

    def _events(self, features: np.ndarray) -> list[list[str]]:
        """Return ordered-SAX, miner, and joint symbolic antecedents per window."""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 3 or x.shape[-1] < self.ordered_words:
            raise ValueError("features must be [n, channels, features] with ordered SAX words appended")
        # Ordered SAX states are appended by the runner.  Rounding protects
        # against npz float storage while retaining their discrete identity.
        words = np.rint(x[..., -self.ordered_words:]).astype(np.int16)
        miner = None
        if self.miner_words_:
            base = x[..., :self.miner_dim_]
            miner = base[..., :3 * self.miner_words_].reshape(
                len(base), base.shape[1], self.miner_words_, 3)
            miner = miner[..., 0] + miner[..., 1] - miner[..., 2]
        events: list[list[str]] = []
        for i, row in enumerate(words):
            row_events: list[str] = []
            for channel, sequence in enumerate(row):
                # Mine suffix n-grams at a few recent offsets. Location is
                # explicit but coarsened, avoiding a unique rule per window.
                for end in range(self.ngram, len(sequence) + 1):
                    offset = len(sequence) - end
                    loc = "recent" if offset < self.ngram else "context"
                    gram = ".".join(str(int(v)) for v in sequence[end - self.ngram:end])
                    row_events.append(f"ch={channel}|loc={loc}|sax={gram}")
                # Keep only the most activated selected miner words.  These
                # tokens have method-specific semantics: a FastShapelets match
                # versus a BOSS-ST/SFA word occurrence.
                if miner is not None:
                    ranking = np.argsort(-miner[i, channel])[:self.miner_events_per_channel]
                    recent = ".".join(str(int(v)) for v in sequence[-self.ngram:])
                    for word in ranking:
                        if miner[i, channel, word] < self.miner_threshold_[channel, word]:
                            continue
                        token = f"ch={channel}|miner_word={int(word)}|active"
                        row_events.append(token)
                        # A joint prefix-and-motif antecedent is the primary
                        # rule language.  Its two components remain visible.
                        row_events.append(f"ch={channel}|loc=recent|sax={recent}&{token}")
            events.append(row_events)
        return events

    @staticmethod
    def _mse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))

    @staticmethod
    def _mae(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))) )

    def _fit_future_states(self, train_future: np.ndarray) -> np.ndarray:
        y = np.asarray(train_future, dtype=np.float32)
        self.future_shape_ = y.shape[1:]
        flat = y.reshape(len(y), -1)
        self.future_mean_ = flat.mean(0, keepdims=True)
        self.future_scale_ = np.maximum(flat.std(0, keepdims=True), 1e-5)
        z = (flat - self.future_mean_) / self.future_scale_
        k = min(self.n_prototypes, len(z))
        if k < 2:
            self.kmeans_ = None
            self.future_centres_ = y[:1].copy()
            return np.zeros(len(y), dtype=np.int32)
        self.kmeans_ = KMeans(n_clusters=k, n_init=10, random_state=self.seed)
        labels = self.kmeans_.fit_predict(z).astype(np.int32)
        centres = self.kmeans_.cluster_centers_ * self.future_scale_ + self.future_mean_
        self.future_centres_ = centres.reshape((k,) + self.future_shape_).astype(np.float32)
        return labels

    def fit(self, train_features: np.ndarray, train_future: np.ndarray):
        """Fit continuation prototypes and train-only symbolic rules."""
        self._fit_miner_thresholds(train_features)
        events = self._events(train_features)
        labels = self._fit_future_states(train_future)
        # Negative control: preserve the feature windows and the empirical
        # future-state distribution, but destroy their correspondence before
        # mining. Test targets are never involved.
        if self.permuted_future_state_control:
            rng = np.random.default_rng(self.seed + 104729)
            labels = labels[rng.permutation(len(labels))]
        total = len(labels)
        prior = np.bincount(labels, minlength=len(self.future_centres_)).astype(np.float64) / max(total, 1)
        antecedent_counts: dict[str, Counter] = defaultdict(Counter)
        for row_events, state in zip(events, labels):
            # An antecedent contributes once per context, not once per repeated
            # occurrence, so support means number of training windows.
            for antecedent in set(row_events):
                antecedent_counts[antecedent][int(state)] += 1
        self.rules_: dict[str, list[ContinuationRule]] = defaultdict(list)
        all_rules: list[ContinuationRule] = []
        for antecedent, counts in antecedent_counts.items():
            total_support = sum(counts.values())
            if total_support < self.min_support:
                continue
            for state, support in counts.items():
                confidence = support / total_support
                lift = confidence / max(prior[state], 1e-8)
                if confidence < self.min_confidence or lift < self.min_lift:
                    continue
                rule = ContinuationRule(antecedent, int(state), int(support), float(confidence), float(lift))
                self.rules_[antecedent].append(rule)
                all_rules.append(rule)
        self.n_train_ = total
        self.state_prior_ = prior
        self.all_rules_ = sorted(all_rules, key=lambda r: (r.lift * r.confidence, r.support), reverse=True)
        return self

    def _rule_forecast(self, features: np.ndarray,
                       blocked_antecedents: list[set[str]] | None = None):
        events = self._events(features)
        n = len(events)
        ref = np.zeros((n,) + self.future_shape_, dtype=np.float32)
        reliability = np.zeros(n, dtype=np.float32)
        matched: list[list[dict]] = []
        for i, row_events in enumerate(events):
            selected = []
            for antecedent in set(row_events):
                if blocked_antecedents is not None and antecedent in blocked_antecedents[i]:
                    continue
                for rule in self.rules_.get(antecedent, ()):
                    # Reliability is monotone in confidence, lift and support.
                    score = rule.confidence * log(max(rule.lift, 1.0))
                    score *= rule.support / (rule.support + self.support_shrinkage)
                    if score > 0:
                        selected.append((score, rule))
            if not selected:
                matched.append([])
                continue
            selected.sort(key=lambda item: (-item[0], item[1].antecedent, item[1].future_state))
            selected = selected[:self.max_rules_per_query]
            weights = np.asarray([item[0] for item in selected], dtype=np.float64)
            weights /= weights.sum()
            states = np.asarray([item[1].future_state for item in selected], dtype=np.int32)
            ref[i] = np.tensordot(weights, self.future_centres_[states], axes=(0, 0))
            # Use the strongest retained rule rather than the sum across many
            # correlated n-grams.  This avoids every query saturating to one.
            reliability[i] = float(min(1.0, selected[0][0]))
            matched.append([{
                "antecedent": item[1].antecedent,
                "future_state": item[1].future_state,
                "support": item[1].support,
                "confidence": item[1].confidence,
                "lift": item[1].lift,
                "rule_weight": float(weight),
            } for weight, item in zip(weights, selected)])
        return ref, reliability, matched

    def fit_apply(self, train_features: np.ndarray, train_future: np.ndarray,
                  val_features: np.ndarray, val_base: np.ndarray, val_target: np.ndarray,
                  test_features: np.ndarray, test_base: np.ndarray) -> RuleEvidenceResult:
        """Fit on train, tune global correction weight on validation, apply to test."""
        self.fit(train_features, train_future)
        val_ref, val_reliability, _ = self._rule_forecast(val_features)
        base_mse, base_mae = self._mse(val_base, val_target), self._mae(val_base, val_target)
        allowed_mae = base_mae * (1.0 + self.mae_relative_tolerance) + self.mae_tolerance
        candidates, selected = [], 0.0
        best = (base_mse, 0.0)
        # The per-window reliability avoids correction without a matching rule.
        for alpha in sorted(set(self.weights + (0.0,))):
            pred = np.asarray(val_base) + alpha * val_reliability[:, None, None] * (val_ref - val_base)
            mse, mae = self._mse(pred, val_target), self._mae(pred, val_target)
            admissible = mae <= allowed_mae + 1e-12
            candidates.append({"weight": alpha, "validation_mse": mse, "validation_mae": mae,
                               "mae_noninferior": bool(admissible)})
            if admissible and (mse, alpha) < best:
                best, selected = (mse, alpha), alpha
        test_ref, test_reliability, matched = self._rule_forecast(test_features)
        output = np.asarray(test_base) + selected * test_reliability[:, None, None] * (test_ref - test_base)

        # Faithfulness diagnostic: remove the strongest matched rule for each
        # query and compare it with removal of one randomly chosen *matched*
        # rule. This is a post-selection perturbation diagnostic; it is not
        # used to choose alpha or any rule/miner hyperparameter.
        rng = np.random.default_rng(self.seed + 7919)
        top_blocks, random_blocks = [], []
        n_with_rule = 0
        for row in matched:
            if not row:
                top_blocks.append(set())
                random_blocks.append(set())
                continue
            n_with_rule += 1
            top_blocks.append({row[0]["antecedent"]})
            random_blocks.append({row[int(rng.integers(len(row)))]["antecedent"]})
        top_ref, top_reliability, _ = self._rule_forecast(test_features, top_blocks)
        random_ref, random_reliability, _ = self._rule_forecast(test_features, random_blocks)
        top_removed = np.asarray(test_base) + selected * top_reliability[:, None, None] * (top_ref - test_base)
        random_removed = np.asarray(test_base) + selected * random_reliability[:, None, None] * (random_ref - test_base)
        top_change = float(np.abs(output - top_removed).mean())
        random_change = float(np.abs(output - random_removed).mean())
        preview_count = min(64, len(matched))
        top_rules = [{
            "antecedent": r.antecedent, "future_state": r.future_state,
            "support": r.support, "confidence": r.confidence, "lift": r.lift,
        } for r in self.all_rules_[:20]]
        summary = {
            "policy": "ordered_symbolic_continuation_rules",
            "selection_rule": "min_validation_mse_subject_to_mae_noninferiority",
            "validation_baseline_mse": base_mse,
            "validation_baseline_mae": base_mae,
            "validation_mae_tolerance": self.mae_tolerance,
            "validation_mae_relative_tolerance": self.mae_relative_tolerance,
            "validation_mae_allowed": allowed_mae,
            "weight_candidates": candidates,
            "selected_weight": selected,
            "n_future_states": int(len(self.future_centres_)),
            "n_mined_rules": int(len(self.all_rules_)),
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_lift": self.min_lift,
            "miner_events_per_channel": self.miner_events_per_channel,
            "miner_activation_quantile": self.miner_activation_quantile,
            "max_rules_per_query": self.max_rules_per_query,
            "miner_event_mode": "sax_only" if self.miner_events_per_channel == 0 else "joint_sax_and_miner_events",
            "permuted_future_state_control": self.permuted_future_state_control,
            "validation_rule_coverage": float((val_reliability > 0).mean()),
            "test_rule_coverage": float((test_reliability > 0).mean()),
            "test_mean_rule_reliability": float(test_reliability.mean()),
            "test_mean_abs_output_change": float(np.abs(output - test_base).mean()),
            "faithfulness": {
                "protocol": "remove_top_matched_rule_per_query_vs_random_matched_rule",
                "queries_with_matched_rule": int(n_with_rule),
                "query_fraction_with_matched_rule": float(n_with_rule / max(len(matched), 1)),
                "top_rule_removal_mean_abs_output_change": top_change,
                "random_rule_removal_mean_abs_output_change": random_change,
                "top_to_random_change_ratio": float(top_change / max(random_change, 1e-12)),
            },
        }
        preview = {
            "query_split": "test", "n_preview": preview_count,
            "matching_rules": matched[:preview_count],
            "rule_reliability": test_reliability[:preview_count].tolist(),
            "top_global_rules": top_rules,
        }
        counterfactuals = {
            "top_rule_removed": top_removed.astype(np.float32),
            "random_matched_rule_removed": random_removed.astype(np.float32),
        }
        return RuleEvidenceResult(output.astype(np.float32), selected, summary, preview, counterfactuals)
