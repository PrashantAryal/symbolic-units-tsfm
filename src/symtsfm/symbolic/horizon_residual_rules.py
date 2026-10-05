"""Chronological, horizon-aware symbolic residual rules for forecasting.

The method deliberately differs from future-state retrieval.  It learns a
mapping from an interpretable historical symbolic event to the *signed UniTS
residual* for each channel and forecast horizon.  A first chronological
validation half is the calibration block; the later half selects a bounded
correction.  The test targets are used only for final evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from symtsfm.symbolic.behavior_rules import ContrastiveBehaviorRules


def _mse(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _mae(p, y):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


@dataclass
class _RuleBank:
    residual: dict[str, np.ndarray]
    score: dict[str, float]
    support: dict[str, int]
    global_residual: np.ndarray


class HorizonResidualRules:
    """Event-conditioned residual correction with required negative controls."""

    def __init__(self, *, ordered_words=24, ngram=3, miner_events_per_channel=2,
                 activation_quantile=0.75, min_support=32, support_shrinkage=24.0,
                 max_rules_per_query=8, weights=(0.0, 0.05, 0.1, 0.2),
                 mae_relative_tolerance=0.01, seed=0):
        self.event_kwargs = dict(ordered_words=int(ordered_words), ngram=int(ngram),
                                 miner_events_per_channel=int(miner_events_per_channel),
                                 activation_quantile=float(activation_quantile),
                                 min_support=2, min_confidence=0.0, min_growth=0.0,
                                 support_shrinkage=float(support_shrinkage),
                                 max_rules_per_query=int(max_rules_per_query), seed=int(seed))
        self.min_support = int(min_support)
        self.support_shrinkage = float(support_shrinkage)
        self.max_rules_per_query = int(max_rules_per_query)
        self.weights = tuple(float(x) for x in weights)
        self.mae_relative_tolerance = float(mae_relative_tolerance)
        self.seed = int(seed)

    def _events(self, train_features, features, miner_events_per_channel):
        miner = ContrastiveBehaviorRules(
            **{**self.event_kwargs, "miner_events_per_channel": miner_events_per_channel})
        miner.fit_event_vocabulary(train_features)
        return miner.events(features)

    def _fit_bank(self, rows, residual, *, permute=False):
        r = np.asarray(residual, dtype=np.float32)
        if r.ndim != 3:
            raise ValueError("forecast residual must be [windows, channels, horizon]")
        if permute:
            rng = np.random.default_rng(self.seed + 1931)
            r = r[rng.permutation(len(r))]
        global_residual = r.mean(0)
        residual_scale = max(float(np.sqrt(np.mean((r - global_residual) ** 2))), 1e-6)
        sums, count = {}, {}
        for events, value in zip(rows, r):
            for event in events:
                if event not in sums:
                    sums[event] = value.astype(np.float64).copy()
                    count[event] = 1
                else:
                    sums[event] += value
                    count[event] += 1
        values, score, support = {}, {}, {}
        for event, total in sums.items():
            n = count[event]
            if n < self.min_support:
                continue
            mean = (total / n).astype(np.float32)
            # Score rules by their event-specific residual effect relative to
            # the global calibration residual, with support shrinkage.
            effect = float(np.sqrt(np.mean((mean - global_residual) ** 2)) / residual_scale)
            score[event] = effect * n / (n + self.support_shrinkage)
            values[event], support[event] = mean, n
        return _RuleBank(values, score, support, global_residual.astype(np.float32))

    def _reference(self, rows, bank: _RuleBank):
        if not rows:
            return np.zeros((0,) + bank.global_residual.shape, dtype=np.float32), np.zeros(0), []
        ref = np.zeros((len(rows),) + bank.global_residual.shape, dtype=np.float32)
        reliability = np.zeros(len(rows), dtype=np.float32)
        preview = []
        for i, events in enumerate(rows):
            selected = sorted(((bank.score[e], e) for e in events if e in bank.score), reverse=True)
            selected = selected[:self.max_rules_per_query]
            if not selected:
                preview.append([])
                continue
            score = np.asarray([s for s, _ in selected], dtype=np.float64)
            weight = score / max(score.sum(), 1e-12)
            refs = np.stack([bank.residual[e] for _, e in selected])
            ref[i] = np.tensordot(weight, refs, axes=(0, 0))
            reliability[i] = float(min(1.0, selected[0][0]))
            preview.append([{"antecedent": e, "support": bank.support[e],
                             "rule_score": float(s), "weight": float(w)}
                            for w, (s, e) in zip(weight, selected)])
        return ref, reliability, preview

    def _select(self, base, target, reference):
        base_mse, base_mae = _mse(base, target), _mae(base, target)
        allowed = base_mae * (1.0 + self.mae_relative_tolerance)
        candidates, chosen, best = [], 0.0, (base_mse, 0.0)
        for alpha in sorted(set(self.weights + (0.0,))):
            p = np.asarray(base) + alpha * np.asarray(reference)
            mse, mae = _mse(p, target), _mae(p, target)
            ok = mae <= allowed + 1e-12
            candidates.append({"alpha": alpha, "validation_mse": mse, "validation_mae": mae,
                               "mae_noninferior": bool(ok)})
            if ok and (mse, alpha) < best:
                best, chosen = (mse, alpha), alpha
        return chosen, candidates, {"mse": base_mse, "mae": base_mae, "mae_allowed": allowed}

    def fit_apply(self, train_features, calibration_features, calibration_base, calibration_target,
                  selection_features, selection_base, selection_target,
                  test_features, test_base, test_target):
        """Fit on early validation; select on later validation; evaluate test only."""
        train_features = np.asarray(train_features)
        calibration_residual = np.asarray(calibration_target) - np.asarray(calibration_base)
        modes = {
            "joint_symbolic_rules": self.event_kwargs["miner_events_per_channel"],
            "sax_only_control": 0,
            "permuted_residual_control": self.event_kwargs["miner_events_per_channel"],
        }
        result, outputs = {}, {}
        for mode, miner_events in modes.items():
            cal_rows = self._events(train_features, calibration_features, miner_events)
            sel_rows = self._events(train_features, selection_features, miner_events)
            test_rows = self._events(train_features, test_features, miner_events)
            bank = self._fit_bank(cal_rows, calibration_residual,
                                  permute=(mode == "permuted_residual_control"))
            sel_ref, _, _ = self._reference(sel_rows, bank)
            test_ref, test_reliability, test_matches = self._reference(test_rows, bank)
            alpha, candidates, base = self._select(selection_base, selection_target, sel_ref)
            output = np.asarray(test_base) + alpha * test_ref
            outputs[mode] = output
            result[mode] = {
                "selected_alpha": alpha, "selection_baseline": base,
                "selection_candidates": candidates, "n_rules": len(bank.residual),
                "test_rule_coverage": float((test_reliability > 0).mean()),
                "test_mean_rule_reliability": float(test_reliability.mean()),
                "test_mean_abs_output_change": float(np.abs(output - test_base).mean()),
                "test_metrics_evaluation_only": {"mse": _mse(output, test_target),
                                                   "mae": _mae(output, test_target)},
                "test_match_preview": test_matches[:64],
            }

        # The no-symbol global residual correction is a mandatory calibration
        # control.  It shares calibration/selection/test boundaries exactly.
        global_reference = calibration_residual.mean(0, keepdims=True)
        global_sel = np.broadcast_to(global_reference, np.asarray(selection_base).shape)
        global_test = np.broadcast_to(global_reference, np.asarray(test_base).shape)
        alpha, candidates, base = self._select(selection_base, selection_target, global_sel)
        global_output = np.asarray(test_base) + alpha * global_test
        outputs["global_residual_control"] = global_output
        result["global_residual_control"] = {
            "selected_alpha": alpha, "selection_baseline": base, "selection_candidates": candidates,
            "test_metrics_evaluation_only": {"mse": _mse(global_output, test_target),
                                               "mae": _mae(global_output, test_target)},
        }
        main = result["joint_symbolic_rules"]
        summary = {
            "policy": "chronological_horizon_aware_symbolic_residual_rules",
            "calibration_protocol": "first_validation_half_mines_rules; later_validation_half_selects_alpha",
            "test_target_never_used_for_rule_mining_or_alpha_selection": True,
            "mae_relative_noninferiority_margin": self.mae_relative_tolerance,
            "main": {k: v for k, v in main.items() if k != "test_match_preview"},
            "controls": {k: v for k, v in result.items() if k != "joint_symbolic_rules"},
        }
        preview = {"query_split": "test", "joint_rule_matches": main["test_match_preview"],
                   "test_control_metrics": {k: v["test_metrics_evaluation_only"] for k, v in result.items()}}
        return outputs["joint_symbolic_rules"], summary, preview
