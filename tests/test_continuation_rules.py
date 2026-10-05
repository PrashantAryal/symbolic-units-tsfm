import numpy as np

from symtsfm.symbolic.continuation_rules import SymbolicContinuationRules


def _features(words):
    """One channel; ordered SAX words are intentionally final positions."""
    return np.asarray(words, dtype=np.float32)[:, None, :]


def test_continuation_rules_use_train_futures_and_emit_provenance():
    # The suffix 0,0,1 is predictive of +1; 3,3,2 predicts -1.
    x = _features([[0, 0, 1], [0, 0, 1], [0, 0, 1], [3, 3, 2], [3, 3, 2], [3, 3, 2]])
    y = np.asarray([[[1.]], [[1.]], [[1.]], [[-1.]], [[-1.]], [[-1.]]], dtype=np.float32)
    base = np.zeros_like(y)
    result = SymbolicContinuationRules(
        ordered_words=3, ngram=3, n_prototypes=2, min_support=2,
        min_confidence=0.5, min_lift=1.0, weights=(0.0, 1.0), seed=11,
    ).fit_apply(x, y, x, base, y, x, base)
    assert result.selected_weight == 1.0
    assert result.summary["n_mined_rules"] >= 2
    assert result.summary["test_rule_coverage"] == 1.0
    assert result.preview["matching_rules"][0][0]["support"] >= 2
    assert "sax=0.0.1" in result.preview["matching_rules"][0][0]["antecedent"]


def test_continuation_rules_abstain_without_matching_symbolic_rule():
    train_x = _features([[0, 0, 1], [0, 0, 1], [3, 3, 2], [3, 3, 2]])
    future = np.asarray([[[1.]], [[1.]], [[-1.]], [[-1.]]], dtype=np.float32)
    query = _features([[1, 2, 3]])
    base = np.asarray([[[0.25]]], dtype=np.float32)
    result = SymbolicContinuationRules(
        ordered_words=3, ngram=3, n_prototypes=2, min_support=2,
        min_confidence=0.5, min_lift=1.0, weights=(0.0, 1.0), seed=3,
    ).fit_apply(train_x, future, query, base, base, query, base)
    assert result.summary["test_rule_coverage"] == 0.0
    np.testing.assert_allclose(result.output, base)


def test_joint_rules_include_the_miner_specific_event_token():
    # Four miner dimensions correspond to one selected word's
    # presence/frequency/distance and one rarity feature; the final three are
    # ordered SAX words.  The high miner activation separates the two states.
    x = np.asarray([
        [[2.0, 2.0, -1.0, 0.0, 0, 0, 1]],
        [[2.0, 2.0, -1.0, 0.0, 0, 0, 1]],
        [[-2.0, -2.0, 1.0, 0.0, 3, 3, 2]],
        [[-2.0, -2.0, 1.0, 0.0, 3, 3, 2]],
    ], dtype=np.float32)
    y = np.asarray([[[1.]], [[1.]], [[-1.]], [[-1.]]], dtype=np.float32)
    result = SymbolicContinuationRules(
        ordered_words=3, ngram=3, n_prototypes=2, min_support=2,
        min_confidence=0.5, min_lift=1.0, miner_events_per_channel=1,
        miner_activation_quantile=0.5, weights=(0.0, 1.0), seed=5,
    ).fit_apply(x, y, x, np.zeros_like(y), y, x, np.zeros_like(y))
    all_antecedents = [r["antecedent"] for row in result.preview["matching_rules"] for r in row]
    assert any("miner_word=0" in antecedent for antecedent in all_antecedents)


def test_sax_only_control_removes_miner_event_tokens():
    x = np.asarray([
        [[2.0, 2.0, -1.0, 0.0, 0, 0, 1]],
        [[2.0, 2.0, -1.0, 0.0, 0, 0, 1]],
        [[-2.0, -2.0, 1.0, 0.0, 3, 3, 2]],
        [[-2.0, -2.0, 1.0, 0.0, 3, 3, 2]],
    ], dtype=np.float32)
    y = np.asarray([[[1.]], [[1.]], [[-1.]], [[-1.]]], dtype=np.float32)
    result = SymbolicContinuationRules(
        ordered_words=3, ngram=3, n_prototypes=2, min_support=2,
        min_confidence=0.5, min_lift=1.0, miner_events_per_channel=0,
        weights=(0.0, 1.0), seed=5,
    ).fit_apply(x, y, x, np.zeros_like(y), y, x, np.zeros_like(y))
    all_antecedents = [r["antecedent"] for row in result.preview["matching_rules"] for r in row]
    assert result.summary["miner_event_mode"] == "sax_only"
    assert not any("miner_word=" in antecedent for antecedent in all_antecedents)


def test_permuted_state_control_and_faithfulness_counterfactuals_are_recorded():
    x = _features([[0, 0, 1], [0, 0, 1], [0, 0, 1], [3, 3, 2], [3, 3, 2], [3, 3, 2]])
    y = np.asarray([[[1.]], [[1.]], [[1.]], [[-1.]], [[-1.]], [[-1.]]], dtype=np.float32)
    base = np.zeros_like(y)
    result = SymbolicContinuationRules(
        ordered_words=3, ngram=3, n_prototypes=2, min_support=2,
        min_confidence=0.5, min_lift=1.0, weights=(0.0, 1.0),
        permuted_future_state_control=True, seed=11,
    ).fit_apply(x, y, x, base, y, x, base)
    assert result.summary["permuted_future_state_control"] is True
    assert set(result.counterfactuals) == {"top_rule_removed", "random_matched_rule_removed"}
    assert result.counterfactuals["top_rule_removed"].shape == y.shape
    assert "top_rule_removal_mean_abs_output_change" in result.summary["faithfulness"]
