import numpy as np

from symtsfm.symbolic.task_evidence import AnomalyEvidence, ClassificationEvidence, ForecastEvidence


def test_classification_policy_can_select_symbolic_evidence():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 1, 3)).astype(np.float32)
    y = (x[:, 0, 0] > 0).astype(int)
    val = x[:20]; yv = y[:20]
    base_val = np.zeros((20, 2), np.float32)
    result = ClassificationEvidence(seed=3).fit_apply(x[20:], y[20:], val, yv, base_val, val, base_val)
    assert result.output.shape == base_val.shape
    assert result.selected_weight > 0


def test_forecast_policy_retrieves_train_only_continuations():
    x = np.array([[[0.]], [[.1]], [[10.]], [[10.1]]], np.float32)
    y = np.array([[[1.]], [[1.]], [[-1.]], [[-1.]]], np.float32)
    base = np.zeros_like(y)
    result = ForecastEvidence(n_neighbors=1).fit_apply(x, y, x, base, y, x, base)
    assert result.output.shape == y.shape
    assert max(max(row) for row in result.preview["neighbor_train_indices"]) < len(x)


def test_anomaly_policy_returns_point_scores():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(30, 2, 4)).astype(np.float32)
    val = rng.normal(size=(8, 2, 4)).astype(np.float32)
    test = rng.normal(size=(6, 2, 4)).astype(np.float32)
    score = np.zeros((8, 10), np.float32)
    labels = np.zeros((8, 10), np.int8); labels[-2:] = 1
    result = AnomalyEvidence().fit_apply(train, score, val, labels, np.zeros((6, 10), np.float32), test)
    assert result.output.shape == (6, 10)
