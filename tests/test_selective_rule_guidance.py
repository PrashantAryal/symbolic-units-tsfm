import numpy as np

from symtsfm.symbolic.selective_rule_guidance import ClassificationRuleGuidance


def _features(n, offset=0):
    """Small valid [windows, channels, miner+ordered-SAX] symbolic tensor."""
    # The implementation expects a 24-word ordered suffix, so use 55 columns.
    x = np.zeros((n, 1, 55), dtype=np.float32)
    for i in range(n):
        label = (i + offset) % 2
        x[i, 0, label * 3] = 2.0
        x[i, 0, -24:] = label + 1
    return x


def test_class_rule_guidance_has_controls_and_preserves_shapes():
    train = _features(24)
    discovery = _features(20, 1)
    selection = _features(20)
    test = _features(20, 1)
    y_discovery = np.arange(20) % 2
    y_selection = np.arange(20) % 2
    y_test = np.arange(20) % 2
    # Deliberately weak base logits: the test checks protocol/shape, not a
    # performance claim on synthetic data.
    val_logits = np.zeros((20, 2), dtype=np.float32)
    test_logits = np.zeros((20, 2), dtype=np.float32)
    method = ClassificationRuleGuidance(min_support=2, max_rules_per_query=4, seed=3)
    output, summary, preview, controls = method.fit_apply(
        train, discovery, y_discovery, selection, val_logits, y_selection,
        test, test_logits, y_test)
    assert output.shape == test_logits.shape
    assert summary["test_targets_never_used_for_rule_mining_or_alpha_selection"]
    assert {"sax_only_control", "permuted_consequent_control", "global_prior_control"} <= set(summary["controls"])
    assert len(preview["joint_rule_matches"]) <= len(test)
    assert controls["global_prior_control"].shape == test_logits.shape
