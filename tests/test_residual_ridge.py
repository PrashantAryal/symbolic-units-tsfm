import numpy as np

from symtsfm.symbolic.residual_ridge import SymbolicResidualRidge


def test_channelwise_ridge_recovers_a_small_residual_mapping():
    x = np.array([[[0.0, 1.0]], [[1.0, 0.0]], [[2.0, 1.0]], [[3.0, 2.0]]], dtype=np.float32)
    # Per horizon: [2*x0 - x1, -x0 + 0.5*x1].
    y = np.stack([2 * x[..., 0] - x[..., 1], -x[..., 0] + 0.5 * x[..., 1]], axis=-1)
    model = SymbolicResidualRidge(ridge_alpha=0.0).fit(x, y)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-5)
    summary = model.summary(["w12:f0", "w12:f1"])
    assert summary["feature_dim"] == 2
    assert summary["horizon"] == 2
