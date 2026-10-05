"""Out-of-fold symbolic residual rules for classification guidance.

Rules are learned from *out-of-fold* UniTS errors, never from in-sample logits.
Thus a rule states a reproducible conditional correction to the frozen-model
output, rather than merely restating which class occurs after a pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symtsfm.symbolic.behavior_rules import ContrastiveBehaviorRules


def _softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _metrics(logits, target):
    y = np.asarray(target, dtype=np.int64).reshape(-1)
    p = _softmax(logits)
    return {"accuracy": float((p.argmax(1) == y).mean()),
            "negative_log_likelihood": float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())}


def log_probability_residual(logits, target, smoothing=0.05, clip=4.0):
    """Bounded correction from an OOF probability to a smoothed true target."""
    p = _softmax(logits)
    y = np.asarray(target, dtype=np.int64).reshape(-1)
    n_classes = p.shape[1]
    q = np.full_like(p, smoothing / n_classes)
    q[np.arange(len(y)), y] = 1.0 - smoothing + smoothing / n_classes
    return np.clip(np.log(q) - np.log(p + 1e-12), -clip, clip).astype(np.float32)


@dataclass
class ResidualRule:
    correction: np.ndarray
    support: int
    strength: float


class OOFClassificationResidualRules:
    """Sparse event-conditioned logit corrections, selected on validation only."""

    def __init__(self, *, ordered_words=24, ngram=3, miner_events_per_channel=2,
                 activation_quantile=0.75, min_support=8, support_shrinkage=12.0,
                 max_rules_per_query=8, alphas=(0.0, 0.05, 0.1, 0.2, 0.5),
                 target_smoothing=0.05, residual_clip=4.0, seed=0):
        self.event_kwargs = dict(
            ordered_words=int(ordered_words), ngram=int(ngram),
            miner_events_per_channel=int(miner_events_per_channel),
            activation_quantile=float(activation_quantile), min_support=2,
            min_confidence=0.0, min_growth=0.0,
            support_shrinkage=float(support_shrinkage),
            max_rules_per_query=int(max_rules_per_query), seed=int(seed))
        self.min_support = int(min_support)
        self.support_shrinkage = float(support_shrinkage)
        self.max_rules_per_query = int(max_rules_per_query)
        self.alphas = tuple(float(x) for x in alphas)
        self.target_smoothing = float(target_smoothing)
        self.residual_clip = float(residual_clip)
        self.seed = int(seed)

    def _events(self, vocabulary_features, features, n_events):
        miner = ContrastiveBehaviorRules(
            **{**self.event_kwargs, "miner_events_per_channel": int(n_events)})
        miner.fit_event_vocabulary(vocabulary_features)
        return miner.events(features)

    def _fit_bank(self, rows, residual, *, permute=False):
        r = np.asarray(residual, dtype=np.float32)
        if permute:
            r = r[np.random.default_rng(self.seed + 3181).permutation(len(r))]
        global_correction = r.mean(axis=0).astype(np.float32)
        centered_scale = max(float(np.sqrt(np.mean((r - global_correction) ** 2))), 1e-6)
        sums, counts = {}, {}
        for events, value in zip(rows, r):
            for event in events:
                sums[event] = sums.get(event, np.zeros_like(value, dtype=np.float64)) + value
                counts[event] = counts.get(event, 0) + 1
        rules = {}
        for event, total in sums.items():
            support = counts[event]
            if support < self.min_support:
                continue
            correction = (total / support).astype(np.float32)
            effect = float(np.sqrt(np.mean((correction - global_correction) ** 2)) / centered_scale)
            strength = effect * support / (support + self.support_shrinkage)
            rules[event] = ResidualRule(correction, int(support), float(strength))
        return rules, global_correction

    def _reference(self, rows, rules, width):
        ref = np.zeros((len(rows), width), dtype=np.float32)
        coverage = np.zeros(len(rows), dtype=np.float32)
        preview = []
        for i, events in enumerate(rows):
            chosen = sorted(((rule.strength, event, rule) for event, rule in rules.items() if event in events),
                            key=lambda item: (-item[0], item[1]))[:self.max_rules_per_query]
            if not chosen:
                preview.append([])
                continue
            weights = np.asarray([item[0] for item in chosen], dtype=np.float64)
            weights /= max(weights.sum(), 1e-12)
            ref[i] = np.tensordot(weights, np.stack([item[2].correction for item in chosen]), axes=(0, 0))
            coverage[i] = 1.0
            preview.append([{"antecedent": event, "support": rule.support,
                             "rule_strength": strength, "weight": float(weight),
                             "class_logit_correction": rule.correction.tolist()}
                            for weight, (strength, event, rule) in zip(weights, chosen)])
        return ref, coverage, preview

    def _select(self, base, target, reference):
        baseline = _metrics(base, target)
        selected, best, candidates = 0.0, (baseline["negative_log_likelihood"], 0.0), []
        for alpha in sorted(set(self.alphas + (0.0,))):
            candidate = _metrics(np.asarray(base) + alpha * reference, target)
            eligible = candidate["accuracy"] + 1e-12 >= baseline["accuracy"]
            candidates.append({"alpha": alpha, "validation": candidate,
                               "accuracy_noninferior": bool(eligible)})
            if eligible and (candidate["negative_log_likelihood"], alpha) < best:
                selected, best = alpha, (candidate["negative_log_likelihood"], alpha)
        return selected, candidates, baseline

    def fit_apply(self, vocabulary_features, train_features, oof_logits, train_target,
                  validation_features, validation_logits, validation_target,
                  test_features, test_logits, test_target):
        residual = log_probability_residual(
            oof_logits, train_target, smoothing=self.target_smoothing, clip=self.residual_clip)
        width = residual.shape[1]
        modes = {"joint_symbolic_residual_rules": self.event_kwargs["miner_events_per_channel"],
                 "sax_only_control": 0,
                 "permuted_residual_control": self.event_kwargs["miner_events_per_channel"]}
        records, outputs = {}, {}
        for name, n_events in modes.items():
            train_rows = self._events(vocabulary_features, train_features, n_events)
            val_rows = self._events(vocabulary_features, validation_features, n_events)
            test_rows = self._events(vocabulary_features, test_features, n_events)
            rules, _ = self._fit_bank(train_rows, residual, permute=(name == "permuted_residual_control"))
            val_ref, _, _ = self._reference(val_rows, rules, width)
            test_ref, coverage, preview = self._reference(test_rows, rules, width)
            alpha, candidates, baseline = self._select(validation_logits, validation_target, val_ref)
            output = np.asarray(test_logits) + alpha * test_ref
            outputs[name] = output
            records[name] = {
                "selected_alpha": alpha, "selection_baseline": baseline,
                "selection_candidates": candidates, "n_rules": len(rules),
                "test_rule_coverage": float(coverage.mean()),
                "test_mean_abs_logit_change": float(np.abs(output - test_logits).mean()),
                "test_metrics_evaluation_only": _metrics(output, test_target),
                "test_match_preview": preview[:64],
            }

        # Event-free residual calibration with identical OOF/validation inputs.
        train_rows = self._events(vocabulary_features, train_features, 0)
        _, global_correction = self._fit_bank(train_rows, residual)
        val_ref = np.broadcast_to(global_correction, np.asarray(validation_logits).shape)
        test_ref = np.broadcast_to(global_correction, np.asarray(test_logits).shape)
        alpha, candidates, baseline = self._select(validation_logits, validation_target, val_ref)
        global_output = np.asarray(test_logits) + alpha * test_ref
        outputs["global_residual_control"] = global_output
        records["global_residual_control"] = {
            "selected_alpha": alpha, "selection_baseline": baseline,
            "selection_candidates": candidates, "n_rules": 0, "test_rule_coverage": 0.0,
            "test_mean_abs_logit_change": float(np.abs(global_output - test_logits).mean()),
            "test_metrics_evaluation_only": _metrics(global_output, test_target),
        }
        main = records["joint_symbolic_residual_rules"]
        return outputs["joint_symbolic_residual_rules"], {
            "policy": "out_of_fold_event_conditioned_class_logit_residual_rules",
            "rule_target": "smoothed_true_log_probability_minus_out_of_fold_foundation_log_probability",
            "test_targets_never_used_for_rule_mining_or_alpha_selection": True,
            "main": {k: v for k, v in main.items() if k != "test_match_preview"},
            "controls": {k: {q: w for q, w in value.items() if q != "test_match_preview"}
                         for k, value in records.items() if k != "joint_symbolic_residual_rules"},
        }, {"query_split": "test", "joint_rule_matches": main["test_match_preview"]}, outputs
