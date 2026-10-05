"""End-to-end smoke test: all three tasks x three variants on synthetic data with the
randomly initialised tiny UniTS. Checks results, report, gate logging and explanation sanity."""
import json
from pathlib import Path

import pytest
import yaml

from symtsfm.experiments.run import main


UNITS_REPO = Path(__file__).resolve().parents[1] / "third_party" / "UniTS"
pytestmark = pytest.mark.skipif(not UNITS_REPO.exists(), reason="needs the UniTS repository at third_party/UniTS")

CFG = {
    "experiment": {"name": "smoke", "phase": 1, "regime": "R1", "seed": 0,
                   "variants": ["baseline", "fastshapelets", "bossst"], "cells": []},
    "model": {"backbone": "units", "name": "debug-tiny-random", "checkpoint": None,
              "repo_dir": str(Path(__file__).resolve().parents[1] / "third_party" / "UniTS")},
    "data": {"classification": {"n_train": 40, "n_test": 12, "length": 96},
             "forecasting": {"n_channels": 2, "horizon": 16}, "anomaly": {"n_train": 40}},
    "symbolic": {"window": 16, "word_len": 4, "alphabet_size": 4, "top_k": 3, "max_fit_series": 64,
                 "fastshapelets": {"n_projections": 4, "mask_size": 1, "n_candidates": 10}},
    "fusion": {"proj_dim": 4},
    "train": {"epochs": 2, "batch_size": 8, "lr_head": 1e-2, "ckpt_every_steps": 3},
    "eval": {"batch_size": 16, "explain": True, "explain_forecasting": True},
}


@pytest.mark.parametrize("task", ["classification", "forecasting", "anomaly"])
def test_pipeline_end_to_end(tmp_path, task):
    cfg = json.loads(json.dumps(CFG))
    cfg["experiment"]["cells"] = [{"task": task, "dataset": "synthetic"}]
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    main(["--config", str(p), "--output-dir", str(out), "--device", "cpu"])
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [r["variant"] for r in rows] == ["baseline", "fastshapelets", "bossst"]
    for r in rows:
        assert r["regime"] == "R1"
        assert all(v == v for v in r["metrics"].values()), r["metrics"]  # no NaN
        if r["variant"] == "baseline":
            assert r["gate"]["gate_mean"] is None
        else:
            assert 0 < r["gate"]["gate_mean"] < 1
            assert r["explanations"]["n_records"] > 0 and r["explanations"]["n_problems"] == 0
            recs = [json.loads(line) for line in (Path(r["cell_dir"]) / "explanations.jsonl").read_text().splitlines()]
            assert len(recs[0]["top_words"]) == 3
            assert all(0 <= w["location"]["start"] < w["location"]["end"] for rec in recs for w in rec["top_words"])
    md = (out / "report" / "results_table.md").read_text()
    assert "regime: R1" in md and "fastshapelets" in md
    assert list((out / "report" / "plots").glob("*.png"))
