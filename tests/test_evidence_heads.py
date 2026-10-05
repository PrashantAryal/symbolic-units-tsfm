"""Phase 10 evidence heads: novelty scorer, reliability head and both readouts.

Pure NumPy/scikit-learn; no GPU or UniTS checkpoint is needed.
"""
import numpy as np
import pytest

from symtsfm.data.loaders import synthetic_task
from symtsfm.symbolic.evidence_heads import (
    anomaly_readout,
    assemble,
    cheap_anomaly_metrics,
    entity_series,
    reliability_readout,
    top1_hit_rate,
)
from symtsfm.symbolic.novelty import FLAT_CODE, SymbolicNoveltyScorer, top_events
from symtsfm.symbolic.reliability import SparseReliability, risk_coverage, summary_statistics

SCFG = {"window": 16, "word_len": 4, "alphabet_size": 4, "top_k": 3, "step": 1, "chunk": 64,
        "fastshapelets": {"n_projections": 4, "mask_size": 1, "n_candidates": 10},
        "bossst": {"numerosity_reduction": True, "min_support": 0.02}}


# --------------------------------------------------------------------------- novelty
@pytest.mark.parametrize("rep", ["sax", "sfa"])
def test_unseen_shape_is_more_novel_than_training_shape(rep):
    t = np.arange(2000)
    normal = np.sin(t / 6.0)[None, None, :]
    sc = SymbolicNoveltyScorer(rep, window=24, word_len=6, alphabet_size=4).fit(normal)
    seen = sc.window_novelty(0, np.sin(np.arange(500, 524) / 6.0)[None])[0]
    spike = np.sin(np.arange(500, 524) / 6.0)
    spike[10:14] += 6.0
    unseen, det = sc.window_novelty(0, spike[None], return_detail=True)
    assert unseen[0] > seen
    assert det["train_count"][0] == 0


def test_flat_windows_use_dedicated_code_and_are_not_noise():
    rng = np.random.default_rng(0)
    X = np.sin(np.arange(1000) / 5.0)[None, None, :]
    sc = SymbolicNoveltyScorer("sax", window=20, word_len=4).fit(X)
    flat = np.full((1, 20), 3.0) + rng.normal(0, 1e-6, (1, 20))  # stuck sensor with float noise
    codes, _ = sc._encode(0, flat)
    assert codes[0] == FLAT_CODE
    d = sc.explain_point(np.full((1, 60), 3.0), 30, 0)
    assert d["word"] == "<flat>" and not d["seen_in_training"]


def test_point_scores_cover_every_window_position():
    sc = SymbolicNoveltyScorer("sax", window=10, word_len=4).fit(np.sin(np.arange(500) / 4.0)[None, None, :])
    s = np.sin(np.arange(200) / 4.0)
    s[100:105] += 5
    p = sc.channel_point_scores(0, s)
    assert p.shape == (200,)
    assert p[100:105].max() >= p[:80].max()
    # explanation span must cover the queried point
    ev = sc.explain_point(s[None], 102, 0)
    assert ev["start"] <= 102 < ev["end"]


def test_top_events_respects_separation():
    s = np.zeros(100)
    s[[10, 12, 50, 90]] = [5, 4, 3, 2]
    assert top_events(s, 3, 5) == [10, 50, 90]


# --------------------------------------------------------------------------- reliability
def test_risk_coverage_oracle_and_random():
    loss = np.r_[np.zeros(80), np.ones(20)]
    oracle = risk_coverage(loss, loss)
    assert oracle["risk_at_80"] == 0.0 and oracle["e_aurc"] == pytest.approx(0.0)
    reverse = risk_coverage(-loss, loss)
    assert reverse["aurc"] > oracle["aurc"]
    assert oracle["aurc_gain_vs_random"] > 0


def test_sparse_reliability_learns_planted_signal_and_decomposes_exactly():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(400, 30))
    y = (F[:, 3] + 0.3 * rng.normal(size=400) > 0.8).astype(int)
    h = SparseReliability("classification", seed=0).fit(F, y)
    assert np.argmax(np.abs(h.coef_)) == 3
    Z = h.scaler_.transform(F[:5])
    logit = h.model_.decision_function(Z)
    assert np.allclose(h.contributions(F[:5]).sum(1) + h.model_.intercept_[0], logit)
    reg = SparseReliability("regression", seed=0, time_ordered=True).fit(F, F[:, 7] * 2 + 0.1 * rng.normal(size=400))
    assert np.argmax(np.abs(reg.coef_)) == 7


def test_reliability_abstains_with_too_few_errors():
    h = SparseReliability("classification").fit(np.random.default_rng(0).normal(size=(30, 4)), np.r_[1, np.zeros(29)])
    assert h.constant_ is not None and h.n_nonzero == 0


def test_summary_statistics_shape():
    assert summary_statistics(np.zeros((5, 3, 40))).shape == (5, 24)


# --------------------------------------------------------------------------- readouts
@pytest.mark.parametrize("variant", ["fastshapelets", "bossst"])
def test_anomaly_readout_end_to_end(variant):
    data = synthetic_task("anomaly", n_train=64)
    rng = np.random.default_rng(0)
    test = data.test
    # A deliberately weak "foundation model": noise only, so the symbolic stream must carry the signal.
    fm = {"val_window_scores": rng.random((len(data.val), 512)),
          "test_window_scores": rng.random((len(test), 512))}
    res = anomaly_readout(data=data, variant=variant, seed=0, fm=fm, metric_fn=cheap_anomaly_metrics,
                          ncfg={"window": 24, "word_len": 6, "alphabet_size": 4, "n_examples": 2})
    m = res["metrics"]
    assert m["symbolic_novelty"]["pr_auc"] > m["fm_reconstruction"]["pr_auc"]
    assert m["fused"]["pr_auc"] > m["fm_reconstruction"]["pr_auc"]
    top = res["explanations"][0]
    assert {"word", "channel", "start", "end", "train_count", "nearest_normal_word"} <= set(top)
    assert len(res["examples"]) == 2 and len(res["examples"][0]["labels_segment"]) > 0
    # the FM stream is a monotone transform of the raw reconstruction score: baseline metrics are exact
    raw = assemble(fm["test_window_scores"], test)
    labels = {e: v["labels"] for e, v in test.meta["entities"].items()}
    assert cheap_anomaly_metrics(raw, labels)["pr_auc"] == pytest.approx(m["fm_reconstruction"]["pr_auc"])


def test_estimate_period_and_window_rule():
    from symtsfm.symbolic.evidence_heads import choose_window
    from symtsfm.symbolic.novelty import estimate_period

    t = np.arange(6000)
    x = np.sin(2 * np.pi * t / 50) + 0.1 * np.random.default_rng(0).normal(size=len(t))
    assert abs(estimate_period([x[None]]) - 50) <= 1
    assert estimate_period([np.random.default_rng(1).normal(size=(1, 6000))]) is None
    w, info = choose_window({"window": "auto"}, 1, [x[None]])
    assert info["rule"] == "training_period" and abs(w - 50) <= 1
    assert choose_window({"window": "auto"}, 3, [np.tile(x, (3, 1))])[0] == 24
    assert choose_window({"window": 40}, 1, None)[0] == 40


def test_anomaly_readout_uses_contiguous_normal_series():
    data = synthetic_task("anomaly", n_train=64)
    rng = np.random.default_rng(0)
    normal = {"syn": np.sin(np.arange(6000) / 6.0)[None] + rng.normal(0, 0.05, (1, 6000))}
    fm = {"val_window_scores": rng.random((len(data.val), 512)),
          "test_window_scores": rng.random((len(data.test), 512))}
    res = anomaly_readout(data=data, variant="fastshapelets", seed=0, fm=fm, metric_fn=cheap_anomaly_metrics,
                          ncfg={"window": "auto", "word_len": 6}, normal_series=normal)
    assert res["window_selection"]["rule"] == "training_period"
    assert res["symbolic_normalizer_source"].startswith("held-out")
    m = res["metrics"]
    assert m["fused"]["pr_auc"] > m["fm_reconstruction"]["pr_auc"]


def test_tail_surprisal_is_monotone_and_tames_heavy_tails():
    from symtsfm.symbolic.novelty import tail_surprisal

    ref = np.random.default_rng(0).normal(size=999)
    x = np.array([-5.0, 0.0, 2.0, 5.0, 500.0])
    q = tail_surprisal(x, ref)
    assert np.all(np.diff(q) > 0)
    assert q[1] == pytest.approx(-np.log(0.5), abs=0.1)
    assert q[-1] < 20  # a 500-sigma value stays on the same order as an extreme-but-plausible one


def test_calibrated_fusion_is_reported():
    data = synthetic_task("anomaly", n_train=64)
    rng = np.random.default_rng(0)
    test_scores = rng.random((len(data.test), 512))
    test_scores[0, 5] = 1e4  # one huge reconstruction outlier must not swamp the symbolic stream
    fm = {"val_window_scores": rng.random((len(data.val), 512)), "test_window_scores": test_scores}
    res = anomaly_readout(data=data, variant="bossst", seed=0, fm=fm, metric_fn=cheap_anomaly_metrics,
                          ncfg={"window": 24, "word_len": 6, "n_examples": 1})
    m = res["metrics"]
    assert "fused_calibrated" in m
    assert m["fused_calibrated"]["pr_auc"] > m["fm_reconstruction"]["pr_auc"]
    assert "fused_calibrated_segment" in res["examples"][0]


def test_entity_series_rebuilds_test_signal():
    data = synthetic_task("anomaly")
    S = entity_series(data.test)["syn"]
    X = data.test.x_raw(np.arange(len(data.test)))
    for x, s in zip(X, data.test.meta["start"]):
        assert np.allclose(S[:, s:s + x.shape[-1]], x)


def test_top1_hit_rate():
    s = {"a": np.r_[np.zeros(50), 1.0, np.zeros(49)]}
    assert top1_hit_rate(s, {"a": np.r_[np.zeros(55), np.ones(5), np.zeros(40)]}, 10) == 1.0
    assert top1_hit_rate(s, {"a": np.r_[np.zeros(95), np.ones(5)]}, 10) == 0.0


@pytest.mark.parametrize("variant", ["fastshapelets", "bossst"])
def test_classification_reliability_readout(variant):
    data = synthetic_task("classification", n_train=120, n_test=80, length=96)
    rng = np.random.default_rng(1)
    ytr = data.train.target(np.arange(len(data.train)))
    yte = data.test.target(np.arange(len(data.test)))

    def logits(y):  # wrong on ~25% of rows
        flip = rng.random(len(y)) < 0.25
        pred = np.where(flip, 1 - y, y)
        return np.eye(2)[pred] * 2 + rng.normal(0, 0.5, (len(y), 2))

    fm = {"fit_logits": logits(ytr), "test_logits": logits(yte),
          "fit_embed": rng.normal(size=(len(ytr), 8)), "test_embed": rng.normal(size=(len(yte), 8))}
    res = reliability_readout(task="classification", data=data, variant=variant, scfg=SCFG,
                              rcfg={"n_permutations": 3, "n_examples": 2}, seed=0, fm=fm)
    rc = res["risk_coverage"]
    assert {"random", "fm_confidence", "statistics", "symbolic", "embedding", "fm_signal+symbolic"} <= set(rc)
    assert 0 < res["symbolic_permutation_control"]["p_value"] <= 1
    assert len(res["examples"]) == 2 and "evidence_terms" in res["examples"][0]


def test_forecasting_reliability_readout():
    data = synthetic_task("forecasting", n_channels=2, horizon=16)
    rng = np.random.default_rng(2)
    tv, tt = data.val.target(np.arange(len(data.val))), data.test.target(np.arange(len(data.test)))
    fm = {"fit_out": tv + rng.normal(0, 0.2, tv.shape), "test_out": tt + rng.normal(0, 0.2, tt.shape)}
    cfg = {**SCFG, "window": 24}
    res = reliability_readout(task="forecasting", data=data, variant="fastshapelets", scfg=cfg,
                              rcfg={"n_permutations": 2, "n_examples": 1}, seed=0, fm=fm)
    assert "context_volatility" in res["risk_coverage"] and "embedding" not in res["risk_coverage"]
    assert len(res["examples"][0]["forecast"]) == 2
