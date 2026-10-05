import json

import numpy as np

from symtsfm.evaluation.metrics import anomaly_metrics, assemble_scores, best_f1, classification_metrics, forecasting_metrics, point_adjust
from symtsfm.evaluation.report import load_results, write_report


def test_point_adjust_expands_detected_segments():
    lab = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(point_adjust(pred, lab), [0, 1, 1, 1, 0, 0, 0, 0])


def test_best_f1_perfect_scores():
    lab = np.r_[np.zeros(90), np.ones(10)]
    f, p, r, _ = best_f1(lab + 0.01 * np.arange(100) / 100, lab)
    assert f == p == r == 1.0


def test_anomaly_metrics_skip_entities_without_anomalies():
    s = {"a": np.r_[np.zeros(50), np.ones(10)], "b": np.random.rand(60)}
    labels = {"a": np.r_[np.zeros(50), np.ones(10)], "b": np.zeros(60)}
    m = anomaly_metrics(s, labels)
    assert m["n_entities_scored"] == 1 and m["pr_auc"] == 1.0 and m["adjusted_best_f1"] == 1.0
    assert "vus_roc" in m  # NaN is allowed in environments without the optional vus package.


def test_assemble_scores_handles_overlap_and_padding():
    ents = {"e": {"length": 10}, "short": {"length": 3}}
    w = np.arange(20, dtype=float).reshape(4, 5)
    out = assemble_scores(w, np.array(["e", "e", "e", "short"]), np.array([0, 5, 5, -2]), ents)
    np.testing.assert_array_equal(out["e"], [0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
    np.testing.assert_array_equal(out["short"], [17, 18, 19])


def test_classification_and_forecasting_metrics():
    m = classification_metrics(np.array([0, 1, 1, 0]), np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]))
    assert m["accuracy"] == 0.75 and m["pr_auc"] == 1.0
    f = forecasting_metrics(np.ones((2, 1, 3)), np.zeros((2, 1, 3)), [10.0], [1.0])
    assert f["mse"] == 1.0 and f["mae"] == 1.0


def _row(regime, variant, acc):
    return {"backbone": "units", "regime": regime, "task": "classification", "dataset": "FordA", "variant": variant,
            "model_name": "m", "metrics": {"accuracy": acc, "macro_f1": acc, "precision": acc, "recall": acc, "pr_auc": acc},
            "gate": {"gate_mean": None if variant == "baseline" else 0.4}, "timings": {}}


def test_report_never_mixes_regimes(tmp_path):
    rows = [_row("R1", "baseline", 0.8), _row("R1", "fastshapelets", 0.85), _row("R2", "bossst", 0.9),
            _row("R1", "baseline", 0.81)]  # re-run: latest wins
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert len(load_results(tmp_path / "results.jsonl")) == 3
    rep = write_report(tmp_path)
    md = (rep / "results_table.md").read_text()
    assert "regime: R1 - frozen backbone" in md and "regime: R2 - full fine-tune" in md
    assert "no baseline in this regime" in md  # the R2 guided row is NOT compared with the R1 baseline
    assert "0.8500 (+0.0400 better)" in md
