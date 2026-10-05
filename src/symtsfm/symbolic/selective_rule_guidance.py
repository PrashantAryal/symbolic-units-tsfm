"""Validation-gated, event-conditioned corrections for frozen TSFMs.

This is intentionally not another neural fusion layer.  A symbolic event earns
the right to modify a foundation-model output only if its correction is learned
on an *early* validation block and the correction size is selected on a later
validation block.  Alpha=0 is always a candidate, so abstention reproduces the
paired foundation-model prediction exactly.

The module also returns three mandatory controls: SAX-only events, permuted
rule consequents, and a global (event-free) class-prior correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symtsfm.symbolic.behavior_rules import ContrastiveBehaviorRules


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = _softmax(logits)
    return {
        "accuracy": float((p.argmax(1) == y).mean()),
        "negative_log_likelihood": float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()),
    }


@dataclass
class _ClassRule:
    delta: np.ndarray
    support: int
    confidence: float
    strength: float


class ClassificationRuleGuidance:
    """Sparse symbolic class-logit correction with held-out test untouched.

    The label distribution after a matched event is represented as a log-prior
    shift.  Multiple matches are averaged using support-shrunk effect strength.
    A correction is selected without looking at test labels.
    """

    def __init__(self, *, ordered_words=24, ngram=3, miner_events_per_channel=2,
                 activation_quantile=0.75, min_support=4, support_shrinkage=8.0,
                 max_rules_per_query=8, alphas=(0.0, 0.05, 0.1, 0.2, 0.5), seed=0):
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
        self.seed = int(seed)

    def _events(self, train_features, features, n_events):
        miner = ContrastiveBehaviorRules(
            **{**self.event_kwargs, "miner_events_per_channel": int(n_events)})
        miner.fit_event_vocabulary(train_features)
        return miner.events(features)

    def _fit_bank(self, rows, target, n_classes, *, permute=False):
        y = np.asarray(target, dtype=np.int64).reshape(-1)
        if permute:
            y = y[np.random.default_rng(self.seed + 721).permutation(len(y))]
        prior_counts = np.bincount(y, minlength=n_classes).astype(np.float64) + 1.0
        log_prior = np.log(prior_counts / prior_counts.sum())
        counts: dict[str, np.ndarray] = {}
        for events, label in zip(rows, y):
            for event in events:
                counts.setdefault(event, np.zeros(n_classes, dtype=np.float64))[label] += 1.0
        rules: dict[str, _ClassRule] = {}
        for event, count in counts.items():
            support = int(count.sum())
            if support < self.min_support:
                continue
            log_event = np.log((count + 1.0) / (support + n_classes))
            delta = (log_event - log_prior).astype(np.float32)
            strength = float(np.sqrt(np.mean(delta ** 2)) * support / (support + self.support_shrinkage))
            rules[event] = _ClassRule(
                delta=delta, support=support, confidence=float(count.max() / support), strength=strength)
        # An explicit event-free correction control.
        return rules, (log_prior - log_prior.mean()).astype(np.float32)

    def _reference(self, rows, rules):
        width = next(iter(rules.values())).delta.shape[0] if rules else 0
        ref = np.zeros((len(rows), width), dtype=np.float32)
        coverage = np.zeros(len(rows), dtype=np.float32)
        preview = []
        for i, events in enumerate(rows):
            chosen = sorted(((r.strength, token, r) for token, r in rules.items() if token in events),
                            key=lambda v: (-v[0], v[1]))[:self.max_rules_per_query]
            if not chosen:
                preview.append([])
                continue
            weights = np.asarray([v[0] for v in chosen], dtype=np.float64)
            weights /= max(weights.sum(), 1e-12)
            ref[i] = np.tensordot(weights, np.stack([v[2].delta for v in chosen]), axes=(0, 0))
            coverage[i] = 1.0
            preview.append([{"antecedent": token, "support": rule.support,
                             "confidence": rule.confidence, "rule_strength": strength,
                             "weight": float(weight)}
                            for weight, (strength, token, rule) in zip(weights, chosen)])
        return ref, coverage, preview

    def _select(self, base, target, ref):
        baseline = _metrics(base, target)
        candidates, selected, best = [], 0.0, (baseline["negative_log_likelihood"], 0.0)
        for alpha in sorted(set(self.alphas + (0.0,))):
            candidate = _metrics(np.asarray(base) + alpha * ref, target)
            # Pre-registered safeguard: do not trade validation accuracy down
            # merely to gain a tiny calibration/NLL change.
            eligible = candidate["accuracy"] + 1e-12 >= baseline["accuracy"]
            candidates.append({"alpha": alpha, "validation": candidate, "accuracy_noninferior": bool(eligible)})
            if eligible and (candidate["negative_log_likelihood"], alpha) < best:
                selected, best = alpha, (candidate["negative_log_likelihood"], alpha)
        return selected, candidates, baseline

    def fit_apply(self, train_features, discovery_features, discovery_target,
                  selection_features, selection_base, selection_target,
                  test_features, test_base, test_target):
        n_classes = int(np.asarray(test_base).shape[1])
        modes = {
            "joint_symbolic_rules": self.event_kwargs["miner_events_per_channel"],
            "sax_only_control": 0,
            "permuted_consequent_control": self.event_kwargs["miner_events_per_channel"],
        }
        details, outputs = {}, {}
        for name, events_per_channel in modes.items():
            discovery_rows = self._events(train_features, discovery_features, events_per_channel)
            selection_rows = self._events(train_features, selection_features, events_per_channel)
            test_rows = self._events(train_features, test_features, events_per_channel)
            rules, _ = self._fit_bank(
                discovery_rows, discovery_target, n_classes,
                permute=(name == "permuted_consequent_control"))
            selection_ref, _, _ = self._reference(selection_rows, rules)
            test_ref, coverage, preview = self._reference(test_rows, rules)
            # A no-rule bank has no class width; retain exact foundation output.
            if test_ref.shape[1] == 0:
                test_ref = np.zeros_like(test_base, dtype=np.float32)
                selection_ref = np.zeros_like(selection_base, dtype=np.float32)
            alpha, candidates, selection_baseline = self._select(selection_base, selection_target, selection_ref)
            output = np.asarray(test_base) + alpha * test_ref
            outputs[name] = output
            details[name] = {
                "selected_alpha": alpha, "selection_baseline": selection_baseline,
                "selection_candidates": candidates, "n_rules": len(rules),
                "test_rule_coverage": float(coverage.mean()),
                "test_mean_abs_logit_change": float(np.abs(output - test_base).mean()),
                "test_metrics_evaluation_only": _metrics(output, test_target),
                "test_match_preview": preview[:64],
            }

        # Event-free global-prior correction, split and selected identically.
        discovery_rows = self._events(train_features, discovery_features, 0)
        _, prior_shift = self._fit_bank(discovery_rows, discovery_target, n_classes)
        global_selection = np.broadcast_to(prior_shift, np.asarray(selection_base).shape)
        global_test = np.broadcast_to(prior_shift, np.asarray(test_base).shape)
        alpha, candidates, selection_baseline = self._select(selection_base, selection_target, global_selection)
        global_output = np.asarray(test_base) + alpha * global_test
        details["global_prior_control"] = {
            "selected_alpha": alpha, "selection_baseline": selection_baseline,
            "selection_candidates": candidates, "n_rules": 0, "test_rule_coverage": 0.0,
            "test_mean_abs_logit_change": float(np.abs(global_output - test_base).mean()),
            "test_metrics_evaluation_only": _metrics(global_output, test_target),
        }
        outputs["global_prior_control"] = global_output
        main = details["joint_symbolic_rules"]
        summary = {
            "policy": "validation_gated_event_conditioned_class_logit_rules",
            "discovery_split": "early_validation_block", "selection_split": "later_validation_block",
            "test_targets_never_used_for_rule_mining_or_alpha_selection": True,
            "main": {k: v for k, v in main.items() if k != "test_match_preview"},
            "controls": {k: {q: w for q, w in v.items() if q != "test_match_preview"}
                         for k, v in details.items() if k != "joint_symbolic_rules"},
        }
        preview = {"query_split": "test", "joint_rule_matches": main["test_match_preview"]}
        return outputs["joint_symbolic_rules"], summary, preview, outputs
