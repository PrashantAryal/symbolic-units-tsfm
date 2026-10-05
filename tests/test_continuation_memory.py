import numpy as np

from symtsfm.symbolic.continuation_memory import SymbolicContinuationMemory
from symtsfm.symbolic.features import forecast_state_key, ordered_sax_context_key


def test_memory_recovers_cluster_continuations_and_zero_is_a_valid_fallback():
    # One symbolic feature cleanly identifies two future-trajectory groups.
    x = np.array([[[0.0]], [[0.1]], [[10.0]], [[10.1]]], dtype=np.float32)
    y = np.array([[[1.0, 1.0]], [[1.0, 1.0]], [[-1.0, -1.0]], [[-1.0, -1.0]]], dtype=np.float32)
    mem = SymbolicContinuationMemory(n_prototypes=2, seed=7).fit(x, y)
    pred, meta = mem.predict(x, return_retrieval=True)
    np.testing.assert_allclose(pred, y)
    assert meta["prototype_index"].shape == (4, 1)

    baseline = np.zeros_like(y)
    alpha, info = mem.choose_alpha(baseline, pred, y, n_grid=21)
    assert alpha == 1.0
    assert info["validation_selected_mse"] < info["validation_baseline_mse"]

    # A harmful memory is rejected by validation because alpha=0 is in the grid.
    alpha_bad, _ = mem.choose_alpha(baseline, -pred, y, n_grid=21)
    assert alpha_bad == 0.0


def test_alpha_rejects_an_mse_gain_when_it_harms_mae():
    # Baseline errors [0, 2] have MSE 2.0 and MAE 1.0. Memory errors
    # [1.1, 1.1] improve MSE (1.21), but worsen MAE (1.1). Strict MAE
    # non-inferiority must preserve the baseline instead of taking that tradeoff.
    target = np.zeros((1, 1, 2), dtype=np.float32)
    baseline = np.array([[[0.0, 2.0]]], dtype=np.float32)
    memory = np.array([[[1.1, 1.1]]], dtype=np.float32)
    alpha, info = SymbolicContinuationMemory.choose_alpha(
        baseline, memory, target, n_grid=21, mae_tolerance=0.0)
    assert alpha == 0.0
    assert info["validation_memory_mse"] < info["validation_baseline_mse"]
    assert info["validation_memory_mae"] > info["validation_baseline_mae"]
    assert info["selection_rule"] == "min_validation_mse_subject_to_mae_noninferiority"


def test_alpha_can_admit_a_residual_correction():
    target = np.zeros((1, 1, 2), dtype=np.float32)
    baseline = np.array([[[1.0, -1.0]]], dtype=np.float32)
    retrieved_residual = -baseline
    corrected = baseline + retrieved_residual
    alpha, info = SymbolicContinuationMemory.choose_alpha(
        baseline, corrected, target, n_grid=21, mae_tolerance=0.0)
    assert alpha == 1.0
    assert info["validation_selected_mae"] == 0.0


def test_individual_joint_knn_recovers_seen_symbolic_continuations():
    # This is deliberately not a prototype average: both channels share one
    # retrieved historical context and retain its multivariate future path.
    x = np.array([
        [[0.0], [0.0]], [[0.1], [0.1]], [[8.0], [8.0]], [[8.1], [8.1]],
    ], dtype=np.float32)
    y = np.array([
        [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
        [[-1.0, -1.0], [-2.0, -2.0]], [[-1.0, -1.0], [-2.0, -2.0]],
    ], dtype=np.float32)
    mem = SymbolicContinuationMemory(
        retrieval_mode="knn", n_neighbors=2, candidate_limit=4,
        joint_channels=True, chunk_size=2, seed=3).fit(x, y)
    pred, retrieval = mem.predict(x, return_retrieval=True)
    np.testing.assert_allclose(pred, y, atol=1e-5)
    assert retrieval["reliability"].shape == (4, 1)
    assert np.all(retrieval["effective_support"] > 1.0)
    assert mem.summary()["retrieval_mode"] == "individual_knn"


def test_ordered_sax_key_distinguishes_reversed_contexts():
    x = np.array([[[0., 1., 2., 3., 4., 5., 6., 7.]],
                  [[7., 6., 5., 4., 3., 2., 1., 0.]]], dtype=np.float32)
    key = ordered_sax_context_key(x, n_words=4, alphabet_size=4)
    assert key.shape == (2, 1, 4)
    assert not np.array_equal(key[0], key[1])


def test_forecast_state_key_retains_level_and_slope():
    x = np.array([[[0., 1., 2., 3., 4., 5., 6., 7.]],
                  [[3., 3., 3., 3., 3., 3., 3., 3.]]], dtype=np.float32)
    key = forecast_state_key(x, n_segments=4, recent=4)
    assert key.shape == (2, 1, 10)  # four PAA values plus six state summaries
    assert not np.array_equal(key[0], key[1])


def test_selective_alpha_rejects_a_low_reliability_harmful_window():
    target = np.zeros((2, 1, 1), dtype=np.float32)
    baseline = np.array([[[1.0]], [[1.0]]], dtype=np.float32)
    memory = np.array([[[0.0]], [[10.0]]], dtype=np.float32)
    reliability = np.array([[[0.9]], [[0.1]]], dtype=np.float32)
    alpha, threshold, info = SymbolicContinuationMemory.choose_selective_alpha(
        baseline, memory, target, reliability, n_grid=2, threshold_grid=3)
    assert alpha == 1.0
    assert threshold > 0.1
    assert info["validation_selected_mae"] < info["validation_baseline_mae"]
