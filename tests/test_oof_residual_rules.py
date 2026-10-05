import numpy as np

from symtsfm.symbolic.oof_residual_rules import OOFClassificationResidualRules, log_probability_residual


def _features(n):
    x = np.zeros((n, 1, 55), dtype=np.float32)
    for i in range(n):
        label = i % 2
        x[i, 0, 3 * label] = 2.0
        x[i, 0, -24:] = label + 1
    return x


def test_oof_residual_rules_returns_joint_and_required_controls():
    train = _features(30)
    val = _features(20)
    test = _features(20)
    y_train = np.arange(30) % 2
    y_val = np.arange(20) % 2
    y_test = np.arange(20) % 2
    oof = np.zeros((30, 2), dtype=np.float32)
    base = np.zeros((20, 2), dtype=np.float32)
    residual = log_probability_residual(oof, y_train)
    assert residual.shape == oof.shape
    method = OOFClassificationResidualRules(min_support=2, max_rules_per_query=4, seed=9)
    output, summary, preview, controls = method.fit_apply(
        train, train, oof, y_train, val, base, y_val, test, base, y_test)
    assert output.shape == base.shape
    assert summary["test_targets_never_used_for_rule_mining_or_alpha_selection"]
    assert {"sax_only_control", "permuted_residual_control", "global_residual_control"} <= set(summary["controls"])
    assert len(preview["joint_rule_matches"]) <= len(test)
    assert controls["global_residual_control"].shape == base.shape
