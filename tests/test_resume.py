"""A run killed mid-epoch must continue from last.pt on --resume, not restart.

The first process is hard-killed (``os._exit`` right after a training step: no
cleanup, no finally blocks -- equivalent to a Colab disconnect) in the middle
of epoch 2. The resumed process must (a) log a resume event at the last
checkpointed step, (b) finish with the same total number of optimizer steps as
an uninterrupted run, and (c) end with identical weights, which is only
possible if optimizer, scheduler, RNG and data position were all restored.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


UNITS_REPO = Path(__file__).resolve().parents[1] / "third_party" / "UniTS"
pytestmark = pytest.mark.skipif(not UNITS_REPO.exists(), reason="needs the UniTS repository at third_party/UniTS")

CFG = {
    "experiment": {"name": "resume_test", "phase": 1, "regime": "R1", "seed": 3, "variants": ["fastshapelets"],
                   "cells": [{"task": "classification", "dataset": "synthetic"}]},
    "model": {"backbone": "units", "name": "debug-tiny-random", "checkpoint": None,
              "repo_dir": str(Path(__file__).resolve().parents[1] / "third_party" / "UniTS")},
    "data": {"classification": {"n_train": 48, "n_test": 16, "length": 128}},
    "symbolic": {"window": 16, "word_len": 4, "alphabet_size": 4, "top_k": 3, "max_fit_series": 64,
                 "fastshapelets": {"n_projections": 4, "mask_size": 1, "n_candidates": 10}},
    "fusion": {"proj_dim": 4},
    "train": {"epochs": 3, "batch_size": 8, "lr_head": 1e-2, "ckpt_every_steps": 2, "cache_backbone_features": True},
    "eval": {"batch_size": 16, "explain": True},
}


def _run(cfg_path, out, *extra):
    return subprocess.run([sys.executable, "-m", "symtsfm.experiments.run", "--config", str(cfg_path), "--output-dir", str(out),
                           "--device", "cpu", *extra], cwd=ROOT, capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])})


def _events(cell):
    return [json.loads(line) for line in (cell / "train_log.jsonl").read_text().splitlines()]


def test_killed_run_resumes_mid_epoch(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG))
    cell = lambda out: out / "units" / "R1" / "classification_synthetic" / "fastshapelets"  # noqa: E731

    # reference: uninterrupted
    ref = _run(cfg_path, tmp_path / "ref")
    assert ref.returncode == 0, ref.stdout[-3000:] + ref.stderr[-3000:]
    ref_last = torch.load(cell(tmp_path / "ref") / "last.pt", weights_only=False)
    steps_per_epoch = next(e for e in _events(cell(tmp_path / "ref")) if e["event"] == "epoch")["global_step"]
    kill_at = steps_per_epoch + 3  # inside epoch 2 (index 1)
    assert steps_per_epoch >= 4

    # killed mid-epoch
    out = tmp_path / "killed"
    k = _run(cfg_path, out, "--debug-exit-after-steps", str(kill_at))
    assert k.returncode == 17, k.stdout[-3000:] + k.stderr[-3000:]
    assert not (cell(out) / "done.json").exists()
    ck = torch.load(cell(out) / "last.pt", weights_only=False)
    saved = ck["state"]
    assert saved["epoch"] == 1 and 0 < saved["batch"] < steps_per_epoch  # mid-epoch checkpoint
    assert saved["global_step"] == kill_at - (kill_at % 2)

    # without --resume we refuse to clobber the checkpoint
    again = _run(cfg_path, out)
    assert again.returncode != 0 and "--resume" in (again.stdout + again.stderr)

    # resume
    r = _run(cfg_path, out, "--resume")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ev = _events(cell(out))
    resume = [e for e in ev if e["event"] == "resume"]
    assert resume and resume[0]["global_step"] == saved["global_step"] > 0
    first_after = next(e for e in ev[ev.index(resume[0]) + 1 :] if "global_step" in e)
    assert first_after["global_step"] > saved["global_step"]  # continued, did not restart at 0
    fin = torch.load(cell(out) / "last.pt", weights_only=False)
    assert fin["state"]["global_step"] == ref_last["state"]["global_step"]
    for key, v in ref_last["model"].items():
        torch.testing.assert_close(fin["model"][key], v, msg=f"weights diverged after resume: {key}")
    assert (cell(out) / "done.json").exists()
    results = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(results) == 1 and results[0]["regime"] == "R1"

    # a second --resume skips the finished cell instead of re-running it
    r2 = _run(cfg_path, out, "--resume")
    assert r2.returncode == 0 and "skip completed" in r2.stdout
    assert len((out / "results.jsonl").read_text().splitlines()) == 1
