"""CLI entry point: run every (cell, variant) of a YAML experiment config.

    python -m symtsfm.experiments.run --config configs/units_classification_r1_full.yaml \
        --output-dir /content/drive/MyDrive/fm_runs [--resume]

Persistence (everything under --output-dir, which must be persistent storage):
  <out>/results.jsonl                         one line appended per finished (cell, variant)
  <out>/report/                               comparison tables + plots, rebuilt after each cell
  <out>/<backbone>/<regime>/<task>_<dataset>/<variant>/
      run_meta.json      regime / cell / variant / resolved config (written at start)
      last.pt            rolling checkpoint (model trainables + optimizer + scheduler + RNG + position)
      best.pt            best validation checkpoint, never overwritten by a worse one
      train_log.jsonl    per-epoch metrics, appended
      symbolic.pkl / symbolic_features.npz   fitted Path B (so resume does not refit)
      explanations.jsonl / explanations_preview.json
      done.json          marks the cell complete (skipped on --resume)
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from symtsfm.data.loaders import load_task, synthetic_task
from symtsfm.evaluation.metrics import (
    ANOMALY_THRESHOLD_PROTOCOL,
    anomaly_metrics,
    assemble_scores,
    classification_metrics,
    forecasting_metrics,
)
from symtsfm.evaluation.report import write_report
from symtsfm.fusion.explain import build_explanations, placeholder_check, symbolic_attributions, verify_record, write_jsonl
from symtsfm.fusion.gate import gate_summary
from symtsfm.symbolic.continuation_memory import SymbolicContinuationMemory
from symtsfm.symbolic.features import SymbolicFeatureExtractor, forecast_state_key, ordered_sax_context_key
from symtsfm.symbolic.retrieval_mixer import AdaptiveRetrievalMixer, SymbolicEmbeddingRetriever
from symtsfm.symbolic.residual_ridge import SymbolicResidualRidge
from symtsfm.symbolic.task_evidence import AnomalyEvidence, ClassificationEvidence, ForecastEvidence
from symtsfm.symbolic.continuation_rules import SymbolicContinuationRules
from symtsfm.symbolic.behavior_rules import ContrastiveBehaviorRules
from symtsfm.symbolic.horizon_residual_rules import HorizonResidualRules
from symtsfm.symbolic.selective_rule_guidance import ClassificationRuleGuidance
from symtsfm.symbolic.oof_residual_rules import OOFClassificationResidualRules

log = logging.getLogger("experiments.run")
VARIANTS = ("baseline", "fastshapelets", "bossst")


# =========================================================================== config
def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def cfg_hash(cfg) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(obj, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=_jsonable) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def check_persistent(output_dir: Path, allow_ephemeral: bool):
    """On Colab, /content is wiped on disconnect: insist on the mounted Drive."""
    p = str(output_dir.resolve()).replace("\\", "/")
    if Path("/content").exists() and p.startswith("/content") and not p.startswith("/content/drive"):
        msg = f"--output-dir {p} is on Colab's ephemeral disk; mount Drive and use /content/drive/..."
        if not allow_ephemeral:
            raise SystemExit(msg + " (or pass --allow-ephemeral for throwaway runs)")
        log.warning(msg)


# =========================================================================== data plumbing
def batches(n, bs):
    return [np.arange(i, min(i + bs, n)) for i in range(0, n, bs)]


def compute_symbolic(split, extractor, chunk, keep_meta=False):
    feats, metas = [], []
    for idx in batches(len(split), chunk):
        f, m = extractor.transform(split.x_raw(idx))
        feats.append(f)
        if keep_meta:
            metas.append(m)
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]} if keep_meta else None
    return np.concatenate(feats), meta


def append_match_locations(features, meta: dict, series_length: int) -> np.ndarray:
    """Append train-only extractor match locations to symbolic feature vectors.

    The existing symbolic values remain standardised. Locations are separately
    scaled to [0, 1] and are zero for a non-informative match. This preserves
    both *which* word matched and *where* it matched for a location-aware
    forecast adapter, without inspecting future targets.
    """
    x = np.asarray(features, dtype=np.float32)
    k = (x.shape[-1] - 1) // 3
    if 3 * k + 1 != x.shape[-1]:
        raise ValueError(f"expected standard symbolic [3*K+1] features, got {x.shape}")
    loc = np.asarray(meta["location"][..., :k], dtype=np.float32)
    informative = np.asarray(meta["informative"][..., :k], dtype=np.float32)
    loc = loc / max(1, int(series_length) - 1)
    loc *= informative
    return np.concatenate([x, loc], axis=-1)


def append_ordered_sax_key(features: np.ndarray, split, n_words: int,
                           alphabet_size: int, chunk: int) -> np.ndarray:
    """Append a train-free ordered SAX context signature in bounded batches."""
    signatures = []
    for idx in batches(len(split), chunk):
        signatures.append(ordered_sax_context_key(
            split.x_raw(idx), n_words=n_words, alphabet_size=alphabet_size))
    return np.concatenate([np.asarray(features, dtype=np.float32),
                           np.concatenate(signatures, axis=0)], axis=-1)


def append_forecast_state_key(features: np.ndarray, split, n_segments: int,
                              recent: int, chunk: int) -> np.ndarray:
    """Append past-only state evidence for state-aware symbolic retrieval."""
    states = []
    for idx in batches(len(split), chunk):
        states.append(forecast_state_key(
            split.x_raw(idx), n_segments=n_segments, recent=recent))
    return np.concatenate([np.asarray(features, dtype=np.float32),
                           np.concatenate(states, axis=0)], axis=-1)


def symbolic_feature_descriptions(extractor, indices, total_features: int) -> list[dict]:
    """Decode flat evidence-feature indices into word/channel descriptions."""
    rows = []
    base_width = extractor.dim_per_channel
    for raw_i in indices:
        i = int(raw_i)
        channel, within = divmod(i, total_features)
        if channel >= extractor.n_channels:
            continue
        if within < base_width:
            word_i, kind = extractor.word_of_feature(within)
            word = None if word_i is None else extractor.words(channel)[word_i]
            rows.append({"flat_feature": i, "channel": channel, "kind": kind,
                         "word": None if word is None else word.get("word"),
                         "support": None if word is None else word.get("support")})
        else:
            rows.append({"flat_feature": i, "channel": channel, "kind": "ordered_sax_state",
                         "word": None, "support": None})
    return rows


def fit_symbolic(data, variant, scfg, seed):
    tr = data.train
    rng = np.random.default_rng(seed)
    n_fit = scfg.get("max_fit_series") or len(tr)
    idx = np.sort(rng.choice(len(tr), min(n_fit, len(tr)), replace=False))
    ext = SymbolicFeatureExtractor(
        method=variant, window=scfg["window"], word_len=scfg["word_len"], alphabet_size=scfg["alphabet_size"],
        top_k=scfg["top_k"], step=scfg.get("step", 1), max_fit_series=scfg.get("max_fit_series"), seed=seed,
        method_kwargs=scfg.get(variant, {}), chunk=scfg.get("chunk", 256),
    )
    labels = tr.fit_labels(idx)
    t0 = time.perf_counter()
    ext.fit_transform(tr.x_raw(idx), labels)
    return ext, time.perf_counter() - t0


def fit_multiscale_symbolic(data, variant, scfg, seed, windows, cell_dir: Path, resume: bool):
    """Fit/transform each requested symbolic scale, concatenated feature-wise.

    Each extractor is fit using train windows only. The compressed transformed
    arrays make an interrupted V8 run resume without refitting symbolic words.
    """
    npz = cell_dir / "multiscale_symbolic_features.npz"
    if resume and npz.exists():
        z = np.load(npz)
        features = {s: z[s] for s in ("train", "val", "test")}
        layout = json.loads(str(z["layout_json"].item()))
        timing = json.loads(str(z["timings"].item()))
        return features, layout, timing
    features = {s: [] for s in ("train", "val", "test")}
    layout, scale_summaries = [], []
    timing = {"symbolic_fit_s": 0.0, "symbolic_transform_s": 0.0, "symbolic_transform_test_s": 0.0}
    for scale_i, window in enumerate(windows):
        scale_cfg = copy.deepcopy(scfg)
        scale_cfg["window"] = int(window)
        ext, fit_s = fit_symbolic(data, variant, scale_cfg, seed + scale_i)
        timing["symbolic_fit_s"] += fit_s
        t0 = time.perf_counter()
        for split_name in ("train", "val"):
            f, _ = compute_symbolic(getattr(data, split_name), ext, scale_cfg.get("chunk", 256))
            features[split_name].append(f)
        timing["symbolic_transform_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        f, _ = compute_symbolic(data.test, ext, scale_cfg.get("chunk", 256))
        features["test"].append(f)
        timing["symbolic_transform_test_s"] += time.perf_counter() - t0
        ext.save(cell_dir / f"symbolic_window_{int(window)}.pkl")
        summary = ext.summary()
        scale_summaries.append(summary)
        layout.extend([f"window_{int(window)}:feature_{j}" for j in range(summary["dim_per_channel"])])
    features = {s: np.concatenate(v, axis=-1) for s, v in features.items()}
    timing["multiscale_windows"] = [int(w) for w in windows]
    timing["multiscale_scale_summaries"] = scale_summaries
    np.savez_compressed(npz, **features, layout_json=json.dumps(layout), timings=json.dumps(timing))
    return features, layout, timing


def time_blocked_residual_indices(n_rows: int, horizon: int, holdout_fraction: float = 0.20,
                                  shadow_validation_fraction: float = 0.10):
    """Return train/validation/residual blocks separated by a horizon-sized gap."""
    if not 0 < holdout_fraction < 0.5 or not 0 < shadow_validation_fraction < 0.5:
        raise ValueError("holdout_fraction and shadow_validation_fraction must be in (0, 0.5)")
    residual_start = int(round(n_rows * (1.0 - holdout_fraction)))
    shadow_end = residual_start - int(horizon)
    shadow_validation_n = max(32, int(round(shadow_end * shadow_validation_fraction)))
    shadow_validation_start = shadow_end - shadow_validation_n
    if shadow_validation_start < 32 or residual_start >= n_rows:
        raise ValueError("not enough training windows for time-blocked symbolic residual fitting")
    return (np.arange(shadow_validation_start),
            np.arange(shadow_validation_start, shadow_end),
            np.arange(residual_start, n_rows),
            {"n_train_rows": int(n_rows), "shadow_train_rows": int(shadow_validation_start),
             "shadow_validation_rows": int(shadow_validation_n), "purged_windows": int(horizon),
             "residual_fit_rows": int(n_rows - residual_start)})


def sym_for_model(sym, task, per_channel=False):
    """[n, C, F] -> classification head input [n, C*F]; everything else stays per channel."""
    return sym.reshape(len(sym), -1) if task == "classification" and not per_channel else sym


def build_model(cfg, data, task, sym_dim, seed):
    mcfg = cfg["model"]
    if mcfg["backbone"] == "units":
        from symtsfm.models.units_wrapper import UniTSDualPath

        length = data.train.x_raw([0]).shape[-1]
        return UniTSDualPath(task=task, dataset=mcfg.get("dataset_key") or data.name, n_channels=data.n_channels,
                             seq_len=length, out_dim=data.out_dim, sym_dim=sym_dim, fusion_cfg=cfg.get("fusion", {}),
                             repo_dir=mcfg.get("repo_dir", "third_party/UniTS"), checkpoint=mcfg["checkpoint"],
                             seed=seed, debug_random=mcfg["name"] == "debug-tiny-random",
                             partial_unfreeze_blocks=mcfg.get("partial_unfreeze_blocks", 1),
                             partial_unfreeze_block_norms=mcfg.get("partial_unfreeze_block_norms", True))
    raise SystemExit(f"unknown backbone {mcfg['backbone']}")


def to_t(a, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


class PathACache:
    """R1 only: the frozen Path A is run once per split and shared by all variants."""

    def __init__(self):
        self.store, self.timings = {}, {}

    def build(self, model, data, split_name, bs, device, dtype):
        if split_name in self.store:
            return self.store[split_name]
        split = getattr(data, split_name)
        z, mu, sd = [], [], []
        t0 = time.perf_counter()
        model.eval()
        for idx in batches(len(split), bs):
            x, m = split.x_model(idx)
            enc = model.encode(to_t(x, device), to_t(m, device))
            z.append(enc["z_a"].to("cpu", dtype))
            mu.append(enc["mean"].cpu())
            sd.append(enc["stdev"].cpu())
        self.timings[split_name] = time.perf_counter() - t0
        self.store[split_name] = {"z_a": torch.cat(z), "mean": torch.cat(mu), "stdev": torch.cat(sd)}
        log.info("cached Path A for %s: %s in %.1fs", split_name, tuple(self.store[split_name]["z_a"].shape),
                 self.timings[split_name])
        return self.store[split_name]


class Runner:
    """Batch-level forward / loss for one (cell, variant), cached or end-to-end."""

    def __init__(self, model, data, sym, cache, device, task):
        self.model, self.data, self.sym, self.cache, self.device, self.task = model, data, sym, cache, device, task

    def forward(self, split_name, idx, sym_override=None, force_gate=None):
        split = getattr(self.data, split_name)
        sym = None
        if self.sym is not None:
            sym = sym_override if sym_override is not None else to_t(
                sym_for_model(self.sym[split_name][idx], self.task, self.model.sym_per_channel), self.device)
        if self.cache is not None:
            c = self.cache[split_name]
            t = torch.as_tensor(idx)
            out = self.model.forward_from_cache(c["z_a"][t].to(self.device, torch.float32), c["mean"][t].to(self.device),
                                                c["stdev"][t].to(self.device), sym=sym, force_gate=force_gate)
        else:
            x, m = self.model.batch_inputs(split, idx)
            out = self.model(to_t(x, self.device), to_t(m, self.device), sym=sym, force_gate=force_gate)
        return out

    def target(self, split_name, idx):
        split = getattr(self.data, split_name)
        if self.task == "classification":
            return to_t(split.target(idx), self.device, torch.long), None
        if self.task == "forecasting":
            return to_t(split.target(idx), self.device), None
        x, m = self.model.batch_inputs(split, idx)
        return to_t(x, self.device), to_t(m, self.device)

    def loss(self, out, tgt, mask):
        if self.task == "classification":
            return F.cross_entropy(out, tgt)
        if self.task == "forecasting":
            return F.mse_loss(out, tgt)
        mk = mask[:, None, :]
        return (((out - tgt) ** 2) * mk).sum() / (mk.sum() * out.shape[1]).clamp_min(1)


# =========================================================================== training
def evaluate_loss(runner, split_name, bs, indices=None):
    runner.model.eval()
    n = len(getattr(runner.data, split_name)) if indices is None else len(indices)
    all_indices = np.arange(len(getattr(runner.data, split_name))) if indices is None else np.asarray(indices, dtype=int)
    tot, correct, cnt = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, n, bs):
            idx = all_indices[start : start + bs]
            out = runner.forward(split_name, idx)["out"]
            tgt, m = runner.target(split_name, idx)
            tot += runner.loss(out, tgt, m).item() * len(idx)
            if runner.task == "classification":
                correct += (out.argmax(1) == tgt).sum().item()
            cnt += len(idx)
    res = {"val_loss": tot / max(cnt, 1)}
    if runner.task == "classification":
        res["val_accuracy"] = correct / max(cnt, 1)
    return res


def train(runner, tcfg, cell_dir: Path, seed, resume: bool, debug_exit_after: int | None,
          train_indices=None, validation_indices=None):
    model = runner.model
    source_train_n = len(runner.data.train)
    train_indices = np.arange(source_train_n) if train_indices is None else np.asarray(train_indices, dtype=int)
    if len(train_indices) == 0:
        raise ValueError("train_indices must contain at least one row")
    n_train = len(train_indices)
    bs = tcfg["batch_size"]
    epochs = tcfg["epochs"]
    n_batches = len(batches(n_train, bs))
    opt = torch.optim.AdamW(model.param_groups(tcfg["lr_head"], tcfg.get("lr_backbone", 1e-5)),
                            weight_decay=tcfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * n_batches))
    select_on = tcfg.get("select_on") or ("val_accuracy" if runner.task == "classification" else "val_loss")
    higher_better = select_on != "val_loss"
    st = {"epoch": 0, "batch": 0, "global_step": 0, "best": None, "bad_epochs": 0, "train_s": 0.0, "stopped": False}
    last, best = cell_dir / "last.pt", cell_dir / "best.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        st = ck["state"]
        log.info("RESUMED from %s at epoch %d batch %d (global step %d)", last, st["epoch"], st["batch"], st["global_step"])
        append_jsonl({"event": "resume", "epoch": st["epoch"], "batch": st["batch"], "global_step": st["global_step"]},
                     cell_dir / "train_log.jsonl")

    def save_last():
        atomic_save({"model": model.trainable_state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
                     "state": copy.deepcopy(st), "regime": model.regime}, last)

    every = tcfg.get("ckpt_every_steps", 100)
    patience = tcfg.get("early_stop_patience")
    # The gate is in [0, 1], so mean(g) is its L1 norm up to a constant.
    # A positive coefficient makes opening the symbolic path earn its cost through
    # a lower task loss.  Validation and model selection remain task-metric-only.
    gate_l1 = float(tcfg.get("gate_l1", 0.0))
    while st["epoch"] < epochs and not st["stopped"]:
        perm = np.random.default_rng(seed * 1000 + st["epoch"]).permutation(n_train)
        order = [perm[i : i + bs] for i in range(0, n_train, bs)]
        model.train()
        t0 = time.perf_counter()
        run_loss, run_task_loss, run_gate, run_n = 0.0, 0.0, 0.0, 0
        while st["batch"] < n_batches:
            idx = np.sort(train_indices[order[st["batch"]]])
            out = runner.forward("train", idx)
            tgt, m = runner.target("train", idx)
            task_loss = runner.loss(out["out"], tgt, m)
            gate_mean = out["gate"].mean() if out["gate"] is not None else task_loss.new_zeros(())
            gate_penalty = gate_l1 * gate_mean
            loss = task_loss + gate_penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tcfg["grad_clip"])
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
            run_task_loss += task_loss.item() * len(idx)
            run_gate += gate_mean.detach().item() * len(idx)
            run_n += len(idx)
            st["batch"] += 1
            st["global_step"] += 1
            if st["global_step"] % every == 0:
                st["train_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                save_last()
                append_jsonl({"event": "checkpoint", "epoch": st["epoch"], "batch": st["batch"],
                              "global_step": st["global_step"], "loss": loss.item()}, cell_dir / "train_log.jsonl")
            if debug_exit_after and st["global_step"] >= debug_exit_after:
                log.warning("debug: hard exit after global step %d (simulated kill)", st["global_step"])
                logging.shutdown()
                os._exit(17)
        st["train_s"] += time.perf_counter() - t0
        validation_split = "train" if validation_indices is not None else "val"
        ev = evaluate_loss(runner, validation_split, tcfg.get("eval_batch_size", bs), validation_indices)
        score = ev[select_on]
        improved = st["best"] is None or (score > st["best"] if higher_better else score < st["best"])
        if improved:
            st["best"], st["bad_epochs"] = score, 0
            atomic_save({"model": model.trainable_state_dict(), "epoch": st["epoch"], select_on: score,
                         "regime": model.regime}, best)
        else:
            st["bad_epochs"] += 1
        rec = {"event": "epoch", "epoch": st["epoch"], "global_step": st["global_step"],
               "train_loss": run_loss / max(run_n, 1),
               "train_task_loss": run_task_loss / max(run_n, 1),
               "train_gate_mean": run_gate / max(run_n, 1),
               "gate_l1": gate_l1, **ev, "improved": improved,
               "lr": [g["lr"] for g in opt.param_groups]}
        append_jsonl(rec, cell_dir / "train_log.jsonl")
        log.info("epoch %d  train_loss %.4f  %s", st["epoch"], rec["train_loss"],
                 "  ".join(f"{k} {v:.4f}" for k, v in ev.items()))
        st["epoch"] += 1
        st["batch"] = 0
        if patience and st["bad_epochs"] >= patience:
            st["stopped"] = True
            log.info("early stopping (patience %d)", patience)
        save_last()
    if best.exists():
        model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["model"], strict=False)
    return st


# =========================================================================== evaluation
def predict(runner, split_name, bs, indices=None, force_gate=None):
    """Return model outputs and gates.

    ``force_gate=0`` is used only for the post-training symbolic ablation.  For
    residual adapters it disables the symbolic correction without modifying the
    trained UniTS path, making the resulting prediction a direct counterfactual
    for whether symbols affected this particular model output.
    """
    runner.model.eval()
    n = len(getattr(runner.data, split_name)) if indices is None else len(indices)
    all_indices = np.arange(len(getattr(runner.data, split_name))) if indices is None else np.asarray(indices, dtype=int)
    outs, gates = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, bs):
            idx = all_indices[start : start + bs]
            o = runner.forward(split_name, idx, force_gate=force_gate)
            outs.append(o["out"].float().cpu())
            if o["gate"] is not None:
                gates.append(o["gate"].float().cpu().reshape(len(idx), -1).mean(1))
    return torch.cat(outs).numpy(), (torch.cat(gates).numpy() if gates else None), time.perf_counter() - t0


def stratified_oof_folds(target, n_folds, seed):
    """Deterministic stratified folds without introducing a test-set split."""
    y = np.asarray(target, dtype=np.int64).reshape(-1)
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    if not len(nonzero) or nonzero.min() < 2:
        raise ValueError("out-of-fold residual guidance needs at least two examples per class")
    k = min(int(n_folds), int(nonzero.min()))
    if k < 2:
        raise ValueError("out-of-fold residual guidance needs at least two folds")
    rng = np.random.default_rng(int(seed) + 6061)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, k)):
            folds[fold].extend(part.tolist())
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def out_of_fold_classification_logits(cfg, data, task, regime, device, train_cfg, cell_dir, seed, n_folds):
    """Train K paired frozen-UniTS heads and return only held-out train logits."""
    y = data.train.target(np.arange(len(data.train))).reshape(-1)
    folds = stratified_oof_folds(y, n_folds, seed)
    all_indices = np.arange(len(data.train), dtype=int)
    logits = None
    records = []
    for fold_id, heldout in enumerate(folds):
        set_seed(seed + 1000 + fold_id)
        model = build_model(cfg, data, task, sym_dim=0, seed=seed + 1000 + fold_id).to(device)
        model.set_regime(regime)
        runner = Runner(model, data, None, None, device, task)
        fold_dir = cell_dir / "oof_residual_models" / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fit_indices = np.setdiff1d(all_indices, heldout, assume_unique=True)
        fold_state = train(runner, train_cfg, fold_dir, seed + 1000 + fold_id,
                           resume=False, debug_exit_after=None,
                           train_indices=fit_indices, validation_indices=heldout)
        held_logits, _, _ = predict(runner, "train", train_cfg.get("eval_batch_size", train_cfg["batch_size"]),
                                    indices=heldout)
        if logits is None:
            logits = np.empty((len(data.train), held_logits.shape[1]), dtype=np.float32)
        logits[heldout] = held_logits
        records.append({"fold": int(fold_id), "fit_rows": int(len(fit_indices)),
                        "heldout_rows": int(len(heldout)), "best": fold_state["best"]})
        del runner, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return logits, records


def predict_with_pooled(runner, split_name, bs, indices=None):
    """Return forecasts and compact UniTS backbone descriptors for retrieval."""
    runner.model.eval()
    split = getattr(runner.data, split_name)
    all_indices = np.arange(len(split)) if indices is None else np.asarray(indices, dtype=int)
    outs, embeds = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(all_indices), bs):
            idx = all_indices[start:start + bs]
            result = runner.forward(split_name, idx)
            pooled = result.get("pooled")
            if pooled is None or pooled.ndim < 2:
                raise RuntimeError("retrieval mixer requires UniTS pooled backbone states")
            dims = tuple(range(1, pooled.ndim - 1))
            embed = pooled.mean(dims) if dims else pooled
            outs.append(result["out"].float().cpu())
            embeds.append(embed.float().cpu())
    return torch.cat(outs).numpy(), torch.cat(embeds).numpy(), time.perf_counter() - t0


def task_metrics(data, task, out, model=None):
    test = data.test
    if task == "classification":
        prob = torch.softmax(torch.as_tensor(out), 1).numpy()
        return classification_metrics(test.target(np.arange(len(test))), prob), None
    if task == "forecasting":
        return forecasting_metrics(out, test.target(np.arange(len(test))), data.info.get("scaler_mean"),
                                   data.info.get("scaler_std")), None
    x, mask = model.batch_inputs(test, np.arange(len(test)))
    scores = ((out - x) ** 2).mean(1) * mask  # [n, L], padded points get 0
    ents = test.meta["entities"]
    full = assemble_scores(scores, test.meta["entity"], test.meta["start"], ents)
    return anomaly_metrics(full, {e: v["labels"] for e, v in ents.items()}), scores


def anomaly_window_scores(split, out, model):
    """Pointwise reconstruction errors for a named anomaly split."""
    x, mask = model.batch_inputs(split, np.arange(len(split)))
    return ((np.asarray(out) - x) ** 2).mean(1) * mask


def _behavior_match_regions(matches, sym_meta, extractor, row_index: int, input_length: int):
    """Return honest plot regions for matched audit rules.

    A FastShapelets/BOSS-ST miner-word event has an exact stored match
    location.  A SAX-only event has only its deliberately coarse ``recent``
    or ``context`` descriptor; the output records that distinction rather than
    pretending it is a precise shapelet span.
    """
    regions = []
    for match in matches:
        text = str(match.get("antecedent", ""))
        ch_m = re.search(r"ch=(\d+)", text)
        word_m = re.search(r"miner_word=(\d+)", text)
        loc_m = re.search(r"loc=(recent|context)", text)
        ch = int(ch_m.group(1)) if ch_m else 0
        if word_m and "location" in sym_meta:
            word = int(word_m.group(1))
            locations = np.asarray(sym_meta["location"])
            if row_index < len(locations) and ch < locations.shape[1] and word < locations.shape[2]:
                start = int(locations[row_index, ch, word])
                regions.append({"channel": ch, "start": max(0, start),
                                "end": min(input_length, start + int(extractor.window_)),
                                "precision": "exact_miner_match", "antecedent": text})
                continue
        loc = loc_m.group(1) if loc_m else "recent"
        if loc == "recent":
            start, end = max(0, input_length - max(4, input_length // 3)), input_length
        else:
            start, end = input_length // 3, min(input_length, 2 * input_length // 3)
        regions.append({"channel": ch, "start": int(start), "end": int(end),
                        "precision": "coarse_sax_context", "antecedent": text})
    return regions


def write_behavior_visualization_payload(*, cell_dir: Path, task: str, variant: str,
                                         data, out: np.ndarray, test_target: np.ndarray,
                                         test_rule_score: np.ndarray, test_matches: list[list[dict]],
                                         behavior_target: np.ndarray, sym_meta: dict,
                                         extractor, behavior: str, cfg: dict):
    """Persist compact, test-only evidence examples for an external PNG renderer.

    The audit never changes ``out``.  Saving inputs, original outputs and rule
    matches makes that claim inspectable and avoids a visualisation that has to
    retrain a model or re-mine rules after the fact.
    """
    vcfg = cfg.get("evidence_visualization", {})
    if not bool(vcfg.get("enabled", False)):
        return None
    n_examples = max(1, int(vcfg.get("n_examples", 6)))
    scores = np.asarray(test_rule_score, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    selected = order[:min(n_examples, len(order))]
    test = data.test
    x_raw = np.asarray(test.x_raw(selected))
    arrays = {
        "test_indices": selected.astype(np.int64),
        "input": x_raw,
        "output": np.asarray(out)[selected],
        "rule_score": scores[selected],
        "behavior_target": np.asarray(behavior_target, dtype=np.int8)[selected],
    }
    if task != "anomaly":
        arrays["target"] = np.asarray(test_target)[selected]
    if task == "anomaly":
        arrays["point_labels"] = np.asarray(test.meta["point_labels"])[selected]
    np.savez_compressed(cell_dir / "symbolic_evidence_examples.npz", **arrays)

    examples = []
    for local_i, test_i in enumerate(selected):
        input_length = int(x_raw[local_i].shape[-1])
        item = {
            "test_index": int(test_i),
            "rule_score": float(scores[test_i]),
            "behavior_target": int(behavior_target[test_i]),
            "matched_rules": test_matches[test_i],
            "match_regions": _behavior_match_regions(
                test_matches[test_i], sym_meta, extractor, int(test_i), input_length),
        }
        if task == "classification":
            logits = np.asarray(out)[test_i]
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.maximum(probs.sum(), 1e-12)
            item.update({"true_class": int(test_target[test_i]),
                         "predicted_class": int(np.argmax(logits)),
                         "predicted_probability": float(np.max(probs))})
        elif task == "forecasting":
            item["window_start"] = int(np.asarray(test.meta.get("starts", np.arange(len(test))))[test_i])
        else:
            item.update({"entity": str(np.asarray(test.meta["entity"])[test_i]),
                         "window_start": int(np.asarray(test.meta["start"])[test_i]),
                         "labelled_anomalous_points": int((np.asarray(test.meta["point_labels"])[test_i] > 0).sum())})
        examples.append(item)
    payload = {
        "schema": "symbolic_evidence_examples_v1",
        "task": task,
        "variant": variant,
        "behavior": behavior,
        "prediction_preserving": True,
        "foundation_output_modified": False,
        "selection": "top_test_rule_score; no test labels used to mine rules or choose rule threshold",
        "n_available_test_windows": int(len(test)),
        "n_saved_examples": int(len(selected)),
        "examples": examples,
    }
    write_json(payload, cell_dir / "symbolic_evidence_examples.json")
    return payload


def anomaly_metrics_from_window_scores(split, scores):
    """Evaluate externally adjusted anomaly scores without altering reconstruction."""
    ents = split.meta["entities"]
    full = assemble_scores(np.asarray(scores), split.meta["entity"], split.meta["start"], ents)
    return anomaly_metrics(full, {e: v["labels"] for e, v in ents.items()})


def explain(runner, extractor, sym_meta, task, variant, out, window_scores, ecfg, bs):
    data = runner.data
    test = data.test
    n = len(test)
    limit = ecfg.get("max_records") or n
    idx_all = np.arange(min(n, limit))
    records = []
    for idx in batches(len(idx_all), bs):
        idx = idx_all[idx]
        if task == "classification":
            pred = out[idx].argmax(1)
            pt = torch.as_tensor(pred, device=runner.device)
            fn = lambda s: runner.forward("test", idx, sym_override=s)["out"].gather(1, pt[:, None])[:, 0]  # noqa: E731
            preds = [int(p) for p in pred]
            targets = [int(t) for t in test.target(idx)]
            extra, offsets, ids = None, None, [int(i) for i in idx]
        elif task == "anomaly":
            x, m = runner.model.batch_inputs(test, idx)
            xt, mt = to_t(x, runner.device), to_t(m, runner.device)
            fn = lambda s: ((((runner.forward("test", idx, sym_override=s)["out"] - xt) ** 2).mean(1) * mt).sum(1)  # noqa: E731
                            / mt.sum(1).clamp_min(1))
            ws = window_scores[idx]
            preds = [{"window_score_max": float(w.max()), "argmax_point": int(w.argmax()),
                      "window_score_mean": float(w.mean())} for w in ws]
            pl = test.meta["point_labels"][idx]
            targets = [{"window_has_anomaly": bool((p > 0).any()), "anomalous_points": int((p > 0).sum())} for p in pl]
            offsets = np.maximum(test.meta["start"][idx], 0)
            ids = [f"{e}:{s}" for e, s in zip(test.meta["entity"][idx], test.meta["start"][idx])]
            extra = [{"entity": str(e), "window_start": int(s)} for e, s in zip(test.meta["entity"][idx], test.meta["start"][idx])]
        else:  # forecasting (optional)
            # The generic forecasting explanation attributes symbols to the
            # average output level across variables and horizons.  Preserve
            # the corresponding target summary and label it explicitly so
            # explanation records cannot be mistaken for per-horizon claims.
            fn = lambda s: runner.forward("test", idx, sym_override=s)["out"].mean((1, 2))  # noqa: E731
            preds = [float(v) for v in out[idx].mean((1, 2))]
            targets = [float(v) for v in test.target(idx).mean((1, 2))]
            extra = [{"attribution_target": "mean_forecast_over_all_channels_and_horizons"}
                     for _ in idx]
            offsets = test.meta["starts"][idx]
            ids = [int(i) for i in idx]
        sym = to_t(sym_for_model(runner.sym["test"][idx], task, runner.model.sym_per_channel), runner.device)
        runner.model.eval()
        attr = symbolic_attributions(fn, sym).cpu().numpy()
        with torch.no_grad():
            g = runner.forward("test", idx)["gate"].float().cpu().reshape(len(idx), -1).mean(1).numpy()
        meta_b = {k: v[idx] for k, v in sym_meta.items()}
        recs = build_explanations(extractor=extractor, attributions=attr, meta=meta_b, x_raw=test.x_raw(idx),
                                  gate=g, predictions=preds, targets=targets, sample_ids=ids, offsets=offsets,
                                  top_n=ecfg.get("top_n", 3), extra=extra, task=task, variant=variant)
        if task == "anomaly":  # for manual plausibility checks: does the evidence touch labelled points?
            for r, lab in zip(recs, test.meta["point_labels"][idx]):
                for w in r["top_words"] + [r["rarest_window"]]:
                    w["labelled_anomalous_points_in_span"] = int((lab[w["location"]["start"]:w["location"]["end"]] > 0).sum())
        records += recs
    problems = []
    for r, i in zip(records, idx_all):
        p = verify_record(r, extractor, test.x_raw(np.array([i]))[0])
        if p:
            problems.append({"sample_id": r["sample_id"], "problems": p})
    return records, problems


# =========================================================================== one cell
def run_variant(cfg, data, task, variant, out_dir: Path, cache: PathACache | None, args):
    ecfg, tcfg, mcfg, scfg = cfg.get("eval", {}), cfg["train"], cfg["model"], cfg["symbolic"]
    regime = cfg["experiment"]["regime"]
    seed = cfg["experiment"].get("seed", 0)
    backbone = mcfg["backbone"]
    memcfg = cfg.get("forecasting_memory", {})
    mixcfg = cfg.get("forecasting_retrieval_mixer", {})
    evidcfg = cfg.get("task_conditioned_evidence", {})
    rulecfg = cfg.get("symbolic_rule_evidence", {})
    behaviorcfg = cfg.get("contrastive_behavior_audit", {})
    guidancecfg = cfg.get("selective_rule_guidance", {})
    oofguidancecfg = cfg.get("oof_symbolic_residual_guidance", {})
    use_continuation_memory = bool(memcfg.get("enabled", False) and task == "forecasting" and variant != "baseline")
    use_retrieval_mixer = bool(mixcfg.get("enabled", False) and task == "forecasting" and variant != "baseline")
    use_task_evidence = bool(evidcfg.get("enabled", False) and variant != "baseline")
    use_rule_evidence = bool(rulecfg.get("enabled", False) and task == "forecasting" and variant != "baseline")
    use_selective_guidance = bool(guidancecfg.get("enabled", False) and variant != "baseline")
    use_oof_residual_guidance = bool(oofguidancecfg.get("enabled", False) and task == "classification" and variant != "baseline")
    # The audit preserves the UniTS output, but is isolated so a run cannot
    # accidentally mix it with a correction and misattribute a metric change.
    use_behavior_audit = bool(behaviorcfg.get("enabled", False) and variant != "baseline")
    if sum((use_continuation_memory, use_retrieval_mixer, use_task_evidence, use_rule_evidence,
            use_behavior_audit, use_selective_guidance, use_oof_residual_guidance)) > 1:
        raise ValueError("choose only one of forecasting_memory, forecasting_retrieval_mixer, "
                          "task_conditioned_evidence, symbolic_rule_evidence, contrastive_behavior_audit, "
                          "selective_rule_guidance, oof_symbolic_residual_guidance")
    memory_mode = str(memcfg.get("mode", "continuation"))
    if use_continuation_memory and memory_mode not in {"continuation", "knn_continuation", "residual", "multiscale_ridge"}:
        raise ValueError("forecasting_memory.mode must be 'continuation', 'knn_continuation', 'residual', or 'multiscale_ridge'")
    use_residual_ridge = use_continuation_memory and memory_mode == "multiscale_ridge"
    cell_dir = out_dir / backbone / regime / f"{task}_{data.name}" / variant
    if (cell_dir / "done.json").exists():
        if args.resume:
            log.info("skip completed %s", cell_dir)
            return None
        if not args.fresh:
            raise SystemExit(f"{cell_dir} already complete; pass --resume to skip or --fresh to redo")
    if cell_dir.exists() and not args.resume and not args.fresh and (cell_dir / "last.pt").exists():
        raise SystemExit(f"{cell_dir} has a checkpoint; pass --resume to continue or --fresh to restart")
    if args.fresh and cell_dir.exists():
        for f in cell_dir.iterdir():
            f.unlink()
    cell_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    set_seed(seed)
    write_json({"regime": regime, "phase": cfg["experiment"].get("phase"), "backbone": backbone,
                "model_name": mcfg["name"], "task": task, "dataset": data.name, "variant": variant,
                "config_hash": cfg_hash(cfg), "config": cfg, "started": dt.datetime.now().isoformat(),
                "device": str(device), "n_train": len(data.train), "n_val": len(data.val), "n_test": len(data.test)},
               cell_dir / "run_meta.json")
    timings = {}
    continuation_memory = None
    ridge_features = ridge_feature_labels = None
    # ---------------- Path B
    extractor, sym, sym_meta = None, None, None
    location_aware_fusion = cfg.get("fusion", {}).get("type") == "horizon_location_prehead_cross_attention"
    # V13 retrieval uses symbolic match locations as part of its search key.
    # This is distinct from V11: locations are not injected into the UniTS
    # head; they only make historical symbolic retrieval query-specific.
    location_aware_memory = bool(use_continuation_memory and
                                 memcfg.get("append_match_locations", False))
    use_match_locations = location_aware_fusion or location_aware_memory
    ordered_sax_words = int(memcfg.get("ordered_sax_words", 0))
    use_ordered_sax = bool(use_continuation_memory and ordered_sax_words)
    state_segments = int(memcfg.get("state_key_segments", 0))
    use_state_key = bool(use_continuation_memory and state_segments)
    # Classical BOSS is a bag.  The new evidence policy additionally retains
    # ordered SAX states so forecasting/anomaly rules retain temporal order.
    evidence_cfg = (oofguidancecfg if use_oof_residual_guidance else guidancecfg if use_selective_guidance else rulecfg if use_rule_evidence else
                    evidcfg if use_task_evidence else behaviorcfg)
    evidence_ordered_sax_words = int(evidence_cfg.get("ordered_sax_words", 0))
    use_evidence_order = bool((use_task_evidence or use_rule_evidence or use_behavior_audit or
                               use_selective_guidance or use_oof_residual_guidance) and evidence_ordered_sax_words)
    if variant != "baseline":
        pk, npz = cell_dir / "symbolic.pkl", cell_dir / "symbolic_features.npz"
        if args.resume and pk.exists() and npz.exists():
            extractor = SymbolicFeatureExtractor.load(pk)
            z = np.load(npz)
            sym = {s: z[s] for s in ("train", "val", "test")}
            sym_meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
            timings.update(json.loads(str(z["timings"])))
        else:
            extractor, timings["symbolic_fit_s"] = fit_symbolic(data, variant, scfg, seed)
            t0 = time.perf_counter()
            sym = {}
            for s in ("train", "val"):
                sym[s], split_meta = compute_symbolic(
                    getattr(data, s), extractor, scfg.get("chunk", 256), keep_meta=use_match_locations)
                if use_match_locations:
                    sym[s] = append_match_locations(
                        sym[s], split_meta, getattr(data, s).x_raw(np.arange(1)).shape[-1])
                if use_ordered_sax:
                    sym[s] = append_ordered_sax_key(
                        sym[s], getattr(data, s), ordered_sax_words,
                        memcfg.get("ordered_sax_alphabet", scfg.get("alphabet_size", 4)),
                        scfg.get("chunk", 256))
                if use_state_key:
                    sym[s] = append_forecast_state_key(
                        sym[s], getattr(data, s), state_segments,
                        memcfg.get("state_key_recent", scfg.get("window", 24)),
                        scfg.get("chunk", 256))
                if use_evidence_order:
                    sym[s] = append_ordered_sax_key(
                        sym[s], getattr(data, s), evidence_ordered_sax_words,
                        evidence_cfg.get("ordered_sax_alphabet", scfg.get("alphabet_size", 4)),
                        scfg.get("chunk", 256))
            t1 = time.perf_counter()
            sym["test"], sym_meta = compute_symbolic(data.test, extractor, scfg.get("chunk", 256), keep_meta=True)
            if use_match_locations:
                sym["test"] = append_match_locations(
                    sym["test"], sym_meta, data.test.x_raw(np.arange(1)).shape[-1])
            if use_ordered_sax:
                sym["test"] = append_ordered_sax_key(
                    sym["test"], data.test, ordered_sax_words,
                    memcfg.get("ordered_sax_alphabet", scfg.get("alphabet_size", 4)),
                    scfg.get("chunk", 256))
            if use_state_key:
                sym["test"] = append_forecast_state_key(
                    sym["test"], data.test, state_segments,
                    memcfg.get("state_key_recent", scfg.get("window", 24)),
                    scfg.get("chunk", 256))
            if use_evidence_order:
                sym["test"] = append_ordered_sax_key(
                    sym["test"], data.test, evidence_ordered_sax_words,
                        evidence_cfg.get("ordered_sax_alphabet", scfg.get("alphabet_size", 4)),
                    scfg.get("chunk", 256))
            timings["symbolic_transform_test_s"] = time.perf_counter() - t1
            timings["symbolic_transform_s"] = time.perf_counter() - t0
            extractor.save(pk)
            np.savez_compressed(npz, **sym, **{f"meta_{k}": v for k, v in sym_meta.items()},
                                timings=json.dumps(timings))
        log.info("[%s] Path B: %d words/channel x %d channels, fit %.1fs", variant, extractor.K, extractor.n_channels,
                 timings.get("symbolic_fit_s", float("nan")))
        # V6 stores full future trajectories before fitting the head. V7 stores
        # post-adaptation foundation-model residuals, so it is fit after the
        # selected UniTS checkpoint has been restored below.
        if use_continuation_memory and memory_mode in {"continuation", "knn_continuation"}:
            continuation_memory = SymbolicContinuationMemory(
                n_prototypes=memcfg.get("n_prototypes", 64), seed=seed,
                chunk_size=memcfg.get("fit_chunk", 2048), candidate_limit=memcfg.get("prototype_candidates", 4096),
                support_shrinkage=memcfg.get("support_shrinkage", 0.0),
                retrieval_mode="knn" if memory_mode == "knn_continuation" else "prototype",
                n_neighbors=memcfg.get("n_neighbors", 8),
                joint_channels=memcfg.get("joint_channels", False),
            ).fit(sym["train"], data.train.target(np.arange(len(data.train))))
            timings["continuation_memory_fit_s"] = continuation_memory.fit_seconds_
            write_json(continuation_memory.summary(), cell_dir / "continuation_memory.json")
        if use_residual_ridge:
            ridge_cfg = memcfg.get("multiscale_ridge", {})
            ridge_features, ridge_feature_labels, ridge_timing = fit_multiscale_symbolic(
                data, variant, scfg, seed, ridge_cfg.get("windows", [12, 24, 48]), cell_dir, args.resume)
            timings["multiscale_symbolic_fit_s"] = ridge_timing["symbolic_fit_s"]
            timings["multiscale_symbolic_transform_s"] = ridge_timing["symbolic_transform_s"]
            timings["multiscale_symbolic_transform_test_s"] = ridge_timing["symbolic_transform_test_s"]
            timings["multiscale_windows"] = ridge_timing["multiscale_windows"]
            log.info("[%s] V8 multi-scale Path B: windows %s, %d features/channel", variant,
                     ridge_timing["multiscale_windows"], ridge_features["train"].shape[-1])
    # ---------------- model
    per_channel = mcfg["backbone"] == "units"
    # V6 is a post-head, validation-calibrated symbolic forecast blend.  Keep
    # its UniTS path exactly baseline-shaped; symbols never enter the neural head.
    # Post-model memory/retrieval/rule evidence must not enter the UniTS head.
    # Otherwise alpha=0 is not an exact foundation-model baseline.
    sym_dim = 0 if (sym is None or use_continuation_memory or use_retrieval_mixer or
                    use_task_evidence or use_rule_evidence or use_behavior_audit or
                    use_selective_guidance or use_oof_residual_guidance) else sym_for_model(sym["train"][:1], task, per_channel).shape[-1]
    if use_continuation_memory or use_retrieval_mixer or use_task_evidence or use_rule_evidence or use_behavior_audit or use_selective_guidance or use_oof_residual_guidance:
        # Symbolic mining can use NumPy/PyTorch randomness.  Reset immediately
        # before model construction so paired baseline and post-model evidence
        # arms begin from the identical UniTS/task-head initialization.
        set_seed(seed)
    model = build_model(cfg, data, task, sym_dim, seed).to(device)
    model.set_regime(regime)
    use_cache = regime == "R1" and tcfg.get("cache_backbone_features", True) and hasattr(model, "encode")
    cache_store = None
    if use_cache:
        dtype = getattr(torch, tcfg.get("cache_dtype", "float32"))
        for s in ("train", "val", "test"):  # built by the first variant, reused by the others
            cache.build(model, data, s, cfg.get("eval", {}).get("batch_size", 64), device, dtype)
        cache_store = cache.store
        timings["path_a_encode_s"] = sum(cache.timings.values())
        timings["path_a_encode_test_s"] = cache.timings["test"]
    runner = Runner(model, data, None if (use_continuation_memory or use_retrieval_mixer or
                                          use_task_evidence or use_rule_evidence or use_behavior_audit or
                                          use_selective_guidance or use_oof_residual_guidance) else sym,
                    cache_store, device, task)
    # ---------------- zero-shot reference (anomaly: the pretrained floor)
    zero_shot = None
    if task == "anomaly" and variant == "baseline" and ecfg.get("zero_shot_reference", True):
        o, _, _ = predict(runner, "test", ecfg.get("batch_size", 64))
        zero_shot, _ = task_metrics(data, task, o, model)
    # ---------------- train
    st = train(runner, tcfg, cell_dir, seed, args.resume, args.debug_exit_after_steps)
    timings["train_s"] = st["train_s"]
    # ---------------- test
    out, gates, head_s = predict(runner, "test", ecfg.get("batch_size", 64))
    continuation = None
    retrieval_mixer_summary = None
    if use_continuation_memory and memory_mode == "residual":
        # Fit on train-only residuals after the adapted UniTS head has selected
        # its best validation checkpoint. No validation/test target is used.
        train_base, _, train_predict_s = predict(runner, "train", ecfg.get("batch_size", 64))
        train_target = data.train.target(np.arange(len(data.train)))
        continuation_memory = SymbolicContinuationMemory(
            n_prototypes=memcfg.get("n_prototypes", 64), seed=seed,
            chunk_size=memcfg.get("fit_chunk", 2048), candidate_limit=memcfg.get("prototype_candidates", 4096),
            support_shrinkage=memcfg.get("support_shrinkage", 0.0),
        ).fit(sym["train"], train_target - train_base)
        timings["continuation_memory_fit_s"] = continuation_memory.fit_seconds_
        timings["residual_reference_train_s"] = train_predict_s
        write_json({**continuation_memory.summary(), "memory_mode": memory_mode,
                    "residual_source": "post_adaptation_train_predictions"},
                   cell_dir / "continuation_memory.json")
    if use_residual_ridge:
        ridge_cfg = memcfg.get("multiscale_ridge", {})
        horizon = int(data.train.target(np.arange(1)).shape[-1])
        shadow_train, shadow_val, residual_fit, block_meta = time_blocked_residual_indices(
            len(data.train), horizon, ridge_cfg.get("holdout_fraction", 0.20),
            ridge_cfg.get("shadow_validation_fraction", 0.10))
        # The shadow model never sees the later residual-fit block. Its errors
        # provide out-of-sample, time-ordered targets for the symbolic ridge.
        set_seed(seed + 314159)
        shadow_model = build_model(cfg, data, task, sym_dim=0, seed=seed + 314159).to(device)
        shadow_model.set_regime(regime)
        shadow_runner = Runner(shadow_model, data, None, None, device, task)
        shadow_cfg = copy.deepcopy(tcfg)
        shadow_cfg["epochs"] = ridge_cfg.get("shadow_epochs", tcfg["epochs"])
        shadow_dir = cell_dir / "shadow_reference"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_st = train(shadow_runner, shadow_cfg, shadow_dir, seed + 314159, args.resume,
                          args.debug_exit_after_steps, train_indices=shadow_train,
                          validation_indices=shadow_val)
        shadow_pred, _, shadow_predict_s = predict(shadow_runner, "train", ecfg.get("batch_size", 64), residual_fit)
        residual_target = data.train.target(residual_fit) - shadow_pred
        ridge = SymbolicResidualRidge(ridge_cfg.get("ridge_alpha", 10.0)).fit(
            ridge_features["train"][residual_fit], residual_target)
        ridge_summary = {**ridge.summary(ridge_feature_labels), **block_meta,
                         "residual_source": "time_blocked_shadow_predictions",
                         "shadow_train_s": float(shadow_st["train_s"]),
                         "shadow_prediction_s": float(shadow_predict_s)}
        timings["shadow_train_s"] = shadow_st["train_s"]
        timings["shadow_reference_predict_s"] = shadow_predict_s
        write_json(ridge_summary, cell_dir / "symbolic_residual_ridge.json")
        val_base, _, _ = predict(runner, "val", ecfg.get("batch_size", 64))
        val_candidate = val_base + ridge.predict(ridge_features["val"])
        alpha, calibration = SymbolicContinuationMemory.choose_alpha(
            val_base, val_candidate, data.val.target(np.arange(len(data.val))),
            n_grid=memcfg.get("alpha_grid", 21), mae_tolerance=memcfg.get("mae_tolerance", 0.0))
        out = out + alpha * ridge.predict(ridge_features["test"])
        continuation = {**ridge_summary, **calibration, "memory_mode": memory_mode, "selected_alpha": alpha}
        write_json(continuation, cell_dir / "continuation_calibration.json")
    if continuation_memory is not None:
        # Alpha is selected exclusively on validation predictions/targets.  It
        # is not re-fit on test data, and alpha=0 means exact foundation baseline.
        val_base, _, _ = predict(runner, "val", ecfg.get("batch_size", 64))
        val_values, val_retrieval = continuation_memory.predict(sym["val"], return_retrieval=True)
        reliability_val = np.asarray(val_retrieval.get("reliability", 1.0))
        while reliability_val.ndim < val_base.ndim:
            reliability_val = reliability_val[..., None]
        if memory_mode == "residual":
            val_memory = val_base + reliability_val * val_values
        else:
            val_memory = val_base + reliability_val * (val_values - val_base)
        selective_retrieval = bool(memcfg.get("selective_reliability", False))
        if selective_retrieval:
            alpha, min_reliability, calibration = continuation_memory.choose_selective_alpha(
                val_base, val_memory, data.val.target(np.arange(len(data.val))), reliability_val,
                n_grid=memcfg.get("alpha_grid", 21),
                threshold_grid=memcfg.get("reliability_threshold_grid", 21),
                mae_tolerance=memcfg.get("mae_tolerance", 0.0),
            )
        else:
            alpha, calibration = continuation_memory.choose_alpha(
                val_base, val_memory, data.val.target(np.arange(len(data.val))),
                n_grid=memcfg.get("alpha_grid", 21),
                mae_tolerance=memcfg.get("mae_tolerance", 0.0),
            )
            min_reliability = -np.inf
        test_values, retrieval = continuation_memory.predict(sym["test"], return_retrieval=True)
        reliability_test = np.asarray(retrieval.get("reliability", 1.0))
        while reliability_test.ndim < out.ndim:
            reliability_test = reliability_test[..., None]
        selection_mask = reliability_test >= min_reliability
        out = out + alpha * selection_mask * reliability_test * test_values if memory_mode == "residual" else \
            out + alpha * selection_mask * reliability_test * (test_values - out)
        continuation = {**continuation_memory.summary(), **calibration, "memory_mode": memory_mode,
                        "selected_alpha": alpha,
                        "test_mean_retrieval_distance": float(retrieval.get("neighbor_distance", retrieval.get("prototype_distance")).mean()),
                        "test_mean_retrieval_support": float(retrieval.get("effective_support", retrieval.get("prototype_support")).mean()),
                        "test_mean_reliability": float(np.asarray(reliability_test).mean()),
                        "test_selected_fraction": float(np.asarray(selection_mask).mean())}
        write_json(continuation, cell_dir / "continuation_calibration.json")
        if "neighbor_train_indices" in retrieval:
            # Keep a compact, target-free audit trail for local forecasting
            # explanations: which train contexts were retrieved, how close they
            # were, and whether their future paths agreed. The full train bank
            # remains in the fitted memory and is never built from test data.
            preview = min(64, len(retrieval["neighbor_train_indices"]))
            write_json({
                "query_split": "test", "n_preview": int(preview),
                "neighbor_train_indices": np.asarray(retrieval["neighbor_train_indices"])[:preview].tolist(),
                "neighbor_distance": np.asarray(retrieval["neighbor_distance"])[:preview].tolist(),
                "effective_support": np.asarray(retrieval["effective_support"])[:preview].tolist(),
                "future_disagreement": np.asarray(retrieval["future_disagreement"])[:preview].tolist(),
                "reliability": np.asarray(retrieval["reliability"])[:preview].tolist(),
            }, cell_dir / "retrieval_preview.json")
    if use_retrieval_mixer:
        # The bank is strictly earlier than the mixer-fitting queries. This
        # prevents a continuation from being retrieved from an overlapping
        # future window while the mixer is trained.
        train_base, train_embed, train_embed_s = predict_with_pooled(runner, "train", ecfg.get("batch_size", 64))
        val_base, val_embed, val_embed_s = predict_with_pooled(runner, "val", ecfg.get("batch_size", 64))
        test_base, test_embed, test_embed_s = predict_with_pooled(runner, "test", ecfg.get("batch_size", 64))
        timings["retrieval_embedding_s"] = train_embed_s + val_embed_s + test_embed_s
        n_train, horizon = len(train_base), int(train_base.shape[-1])
        bank_end = int(n_train * float(mixcfg.get("bank_fraction", 0.55)))
        fit_start = bank_end + int(mixcfg.get("purge_windows", horizon))
        if bank_end < 128 or fit_start >= n_train - 32:
            raise ValueError("not enough time-ordered rows for the retrieval-mixer bank and fit block")
        bank_idx, fit_idx = np.arange(bank_end), np.arange(fit_start, n_train)
        # Select the symbolic contribution *only on validation*.  This replaces
        # V17's hand-selected beta=0.20 with a predefined grid containing the
        # no-symbolic retrieval control (beta=0). Test outputs are constructed
        # only after the selection is final.
        weight_grid = [float(v) for v in mixcfg.get("symbolic_weight_grid", [mixcfg.get("symbolic_weight", 0.20)])]
        if 0.0 not in weight_grid:
            weight_grid = [0.0] + weight_grid
        weight_grid = sorted(set(weight_grid))
        base_val_target = data.val.target(np.arange(len(data.val)))
        base_val_mse = float(np.mean((val_base - base_val_target) ** 2))
        base_val_mae = float(np.mean(np.abs(val_base - base_val_target)))
        mae_tolerance = float(mixcfg.get("validation_mae_tolerance", 0.0))
        candidates = []
        selected = None
        for beta in weight_grid:
            fit_retriever = SymbolicEmbeddingRetriever(
                n_neighbors=mixcfg.get("n_neighbors", 8), symbolic_weight=beta,
                chunk_size=mixcfg.get("retrieval_chunk", 256),
            ).fit(train_embed[bank_idx], sym["train"][bank_idx], data.train.target(bank_idx), indices=bank_idx)
            fit_retrieval = fit_retriever.retrieve(train_embed[fit_idx], sym["train"][fit_idx])
            # Validation/test can use all observed training continuations, but
            # never validation/test targets.
            deployment_retriever = SymbolicEmbeddingRetriever(
                n_neighbors=mixcfg.get("n_neighbors", 8), symbolic_weight=beta,
                chunk_size=mixcfg.get("retrieval_chunk", 256),
            ).fit(train_embed, sym["train"], data.train.target(np.arange(n_train)), indices=np.arange(n_train))
            val_retrieval = deployment_retriever.retrieve(val_embed, sym["val"])
            mixer = AdaptiveRetrievalMixer(
                train_embed.shape[1], train_base.shape[1], train_base.shape[2],
                hidden_dim=mixcfg.get("hidden_dim", 128), lr=mixcfg.get("lr", 1e-3),
                weight_decay=mixcfg.get("weight_decay", 1e-4), gate_l1=mixcfg.get("gate_l1", 1e-3),
                epochs=mixcfg.get("epochs", 40), batch_size=mixcfg.get("batch_size", 128),
                patience=mixcfg.get("early_stop_patience", 6), device=device, seed=seed,
            ).fit(train_base[fit_idx], train_embed[fit_idx], fit_retrieval, data.train.target(fit_idx),
                  val_base, val_embed, val_retrieval, base_val_target)
            val_out, val_gate = mixer.predict(val_base, val_embed, val_retrieval)
            val_mse = float(np.mean((val_out - base_val_target) ** 2))
            val_mae = float(np.mean(np.abs(val_out - base_val_target)))
            item = {"symbolic_weight": beta, "validation_mse": val_mse, "validation_mae": val_mae,
                    "validation_gate_mean": float(val_gate.mean()),
                    "mae_noninferior": bool(val_mae <= base_val_mae + mae_tolerance + 1e-12)}
            candidates.append(item)
            if item["mae_noninferior"] and (selected is None or (val_mse, beta) <
                                             (selected["item"]["validation_mse"], selected["item"]["symbolic_weight"])):
                selected = {"item": item, "mixer": mixer, "retriever": deployment_retriever}
        # beta=0 is included, but retain an exact UniTS fallback if all learned
        # retrieval mixers violate the prespecified validation MAE constraint.
        if selected is None:
            out, retrieval_gate = test_base, np.zeros((len(test_base), test_base.shape[1], 1), dtype=np.float32)
            test_retrieval = None
            selected_item = {"symbolic_weight": None, "validation_mse": base_val_mse,
                             "validation_mae": base_val_mae, "validation_gate_mean": 0.0,
                             "mae_noninferior": True, "fallback": "exact_units_baseline"}
            mixer = None; retriever = None
        else:
            mixer, retriever, selected_item = selected["mixer"], selected["retriever"], selected["item"]
            test_retrieval = retriever.retrieve(test_embed, sym["test"])
            out, retrieval_gate = mixer.predict(test_base, test_embed, test_retrieval)
        gates = retrieval_gate.reshape(len(retrieval_gate), -1).mean(1)
        if mixer is not None:
            # Intervention audit only; it is not presented as the separately
            # trained beta=0 performance control in ``weight_candidates``.
            plain_retrieval = retriever.retrieve(test_embed, sym["test"], symbolic_weight=0.0)
            plain_out, _ = mixer.predict(test_base, test_embed, plain_retrieval)
            plain_metrics, _ = task_metrics(data, task, plain_out, model)
            effect = np.asarray(out, dtype=np.float64) - np.asarray(plain_out, dtype=np.float64)
        else:
            plain_metrics, effect = task_metrics(data, task, out, model)[0], np.zeros_like(out)
        retrieval_mixer_summary = {
            **({} if mixer is None else mixer.summary()), "mode": "units_embedding_retrieval_symbolic_rerank",
            "selection_rule": "min_validation_mse_subject_to_mae_noninferiority",
            "validation_baseline_mse": base_val_mse, "validation_baseline_mae": base_val_mae,
            "validation_mae_tolerance": mae_tolerance, "weight_candidates": candidates,
            "selected_symbolic_weight": selected_item["symbolic_weight"],
            "n_neighbors": int(mixcfg.get("n_neighbors", 8)),
            "bank_rows_for_mixer_fit": int(len(bank_idx)), "mixer_fit_rows": int(len(fit_idx)),
            "purged_windows": int(fit_start - bank_end), "test_gate_mean": float(retrieval_gate.mean()),
            "symbolic_rerank_ablation": {
                "protocol": "same_trained_mixer_retrieve_without_symbolic_reranking",
                "metrics": {k: float(v) for k, v in plain_metrics.items()},
                "mean_absolute_output_change": float(np.abs(effect).mean()),
            },
        }
        write_json(retrieval_mixer_summary, cell_dir / "retrieval_mixer.json")
        if test_retrieval is not None:
            preview = min(64, len(test_retrieval.indices))
            write_json({
                "query_split": "test", "n_preview": int(preview),
                "selected_symbolic_weight": selected_item["symbolic_weight"],
                "neighbor_train_indices": test_retrieval.indices[:preview].tolist(),
                "embedding_similarity": test_retrieval.embedding_score[:preview].tolist(),
                "symbolic_similarity": test_retrieval.symbolic_score[:preview].tolist(),
                "retrieval_weight": test_retrieval.weights[:preview].tolist(),
            }, cell_dir / "retrieval_provenance_preview.json")
        log.info("[%s] retrieval mixer: selected_beta=%s bank=%d fit=%d gate=%.4f", variant,
                 selected_item["symbolic_weight"], len(bank_idx), len(fit_idx), retrieval_gate.mean())
    task_evidence_summary = None
    rule_evidence_summary = None
    behavior_audit_summary = None
    selective_guidance_summary = None
    oof_residual_guidance_summary = None
    external_window_scores = None
    if use_task_evidence:
        # The evidence policy sees symbolic events while the UniTS model sees
        # no symbolic input.  This cleanly tests task-specific evidence rather
        # than another generic neural fusion pathway.
        train_base, _, _ = predict(runner, "train", ecfg.get("batch_size", 64))
        val_base, _, _ = predict(runner, "val", ecfg.get("batch_size", 64))
        if task == "classification":
            evidence = ClassificationEvidence(
                c=evidcfg.get("classification_c", 0.1),
                weights=evidcfg.get("classification_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0]), seed=seed)
            evidence_result = evidence.fit_apply(
                sym["train"], data.train.target(np.arange(len(data.train))),
                sym["val"], data.val.target(np.arange(len(data.val))), val_base,
                sym["test"], out)
            out = evidence_result.output
        elif task == "forecasting":
            evidence = ForecastEvidence(
                n_neighbors=evidcfg.get("n_neighbors", 8),
                weights=evidcfg.get("forecast_weight_grid", [0.0, 0.1, 0.25, 0.5, 1.0]),
                mae_tolerance=evidcfg.get("validation_mae_tolerance", 0.0),
                chunk_size=evidcfg.get("retrieval_chunk", 256))
            evidence_result = evidence.fit_apply(
                sym["train"], data.train.target(np.arange(len(data.train))),
                sym["val"], val_base, data.val.target(np.arange(len(data.val))), sym["test"], out)
            out = evidence_result.output
        else:
            # Normal-event memory uses train windows only.  Validation windows
            # set a pre-declared normal false-alarm budget; test labels are
            # never inspected here.
            val_score = anomaly_window_scores(data.val, val_base, model)
            test_score = anomaly_window_scores(data.test, out, model)
            evidence = AnomalyEvidence(
                weights=evidcfg.get("anomaly_weight_grid", [0.0, 0.1, 0.25, 0.5, 1.0]),
                normal_score_tolerance=evidcfg.get("normal_score_tolerance", 0.05))
            validation_labels = data.val.meta.get("point_labels", data.val.meta.get("window_labels"))
            evidence_result = evidence.fit_apply(
                sym["train"], val_score, sym["val"], validation_labels, test_score, sym["test"])
            external_window_scores = evidence_result.output
        task_evidence_summary = evidence_result.summary
        task_evidence_summary.update({"method": variant, "ordered_sax_words": evidence_ordered_sax_words,
                                      "active_on_test": bool(evidence_result.selected_weight > 0)})
        write_json(task_evidence_summary, cell_dir / "task_conditioned_evidence.json")
        preview_payload = {**evidence_result.preview, "symbolic_vocabulary": extractor.summary()}
        if "top_global_feature_indices" in preview_payload:
            preview_payload["top_global_feature_descriptions"] = symbolic_feature_descriptions(
                extractor, preview_payload["top_global_feature_indices"], sym["train"].shape[-1])
        write_json(preview_payload,
                   cell_dir / "task_conditioned_evidence_preview.json")
        # Stored in the established summary field solely to make abstention
        # visible in result tables.  It is a selected policy weight, not an
        # internal Transformer attention weight.
        gates = np.full(len(data.test), evidence_result.selected_weight, dtype=np.float32)
        log.info("[%s] task-conditioned evidence: policy=%s selected_weight=%.4f", variant,
                 task_evidence_summary["policy"], evidence_result.selected_weight)
    if use_rule_evidence:
        # Phase 4 forecasting evidence is rule based, not nearest-neighbour
        # retrieval.  The miner sees only symbolic contexts and known training
        # continuations; test futures never enter fitting or calibration.
        val_base, _, _ = predict(runner, "val", ecfg.get("batch_size", 64))
        rules = SymbolicContinuationRules(
            ordered_words=rulecfg.get("ordered_sax_words", 24),
            ngram=rulecfg.get("ngram", 3),
            n_prototypes=rulecfg.get("n_future_states", 8),
            min_support=rulecfg.get("min_support", 8),
            min_confidence=rulecfg.get("min_confidence", 0.20),
            min_lift=rulecfg.get("min_lift", 1.05),
            support_shrinkage=rulecfg.get("support_shrinkage", 12.0),
            miner_events_per_channel=rulecfg.get("miner_events_per_channel", 2),
            miner_activation_quantile=rulecfg.get("miner_activation_quantile", 0.75),
            max_rules_per_query=rulecfg.get("max_rules_per_query", 8),
            weights=rulecfg.get("forecast_weight_grid", [0.0, 0.1, 0.25, 0.5, 1.0]),
            mae_tolerance=rulecfg.get("validation_mae_tolerance", 0.0),
            mae_relative_tolerance=rulecfg.get("validation_mae_relative_tolerance", 0.0),
            permuted_future_state_control=rulecfg.get("permuted_future_state_control", False),
            seed=seed,
        )
        evidence_result = rules.fit_apply(
            sym["train"], data.train.target(np.arange(len(data.train))),
            sym["val"], val_base, data.val.target(np.arange(len(data.val))),
            sym["test"], out,
        )
        out = evidence_result.output
        rule_evidence_summary = evidence_result.summary
        rule_evidence_summary.update({
            "method": variant,
            "ordered_sax_words": int(rulecfg.get("ordered_sax_words", 24)),
            "active_on_test": bool(evidence_result.selected_weight > 0),
        })
        preview_payload = {**evidence_result.preview, "symbolic_vocabulary": extractor.summary()}
        write_json(preview_payload, cell_dir / "symbolic_rule_evidence_preview.json")
        # This table field holds a validation-selected correction weight, not
        # an internal Transformer attention/gate value.
        gates = np.full(len(data.test), evidence_result.selected_weight, dtype=np.float32)
        log.info("[%s] symbolic continuation rules: rules=%d coverage=%.3f selected_weight=%.4f", variant,
                 rule_evidence_summary["n_mined_rules"], rule_evidence_summary["test_rule_coverage"],
                 evidence_result.selected_weight)
    if use_oof_residual_guidance:
        # The rule target is an out-of-fold UniTS residual.  This avoids the
        # invalid shortcut of learning corrections from a head's in-sample,
        # potentially overfit predictions.  The official validation set only
        # selects alpha; no test label appears before final reporting.
        oof_train_cfg = dict(tcfg)
        oof_train_cfg["epochs"] = int(oofguidancecfg.get("oof_epochs", tcfg["epochs"]))
        oof_logits, oof_folds = out_of_fold_classification_logits(
            cfg, data, task, regime, device, oof_train_cfg, cell_dir, seed,
            oofguidancecfg.get("n_oof_folds", 3))
        val_base, _, _ = predict(runner, "val", ecfg.get("batch_size", 64))
        train_y = data.train.target(np.arange(len(data.train))).reshape(-1)
        val_y = data.val.target(np.arange(len(data.val))).reshape(-1)
        test_y = data.test.target(np.arange(len(data.test))).reshape(-1)
        guidance = OOFClassificationResidualRules(
            ordered_words=oofguidancecfg.get("ordered_sax_words", 24),
            ngram=oofguidancecfg.get("ngram", 3),
            miner_events_per_channel=oofguidancecfg.get("miner_events_per_channel", 2),
            activation_quantile=oofguidancecfg.get("miner_activation_quantile", 0.75),
            min_support=oofguidancecfg.get("min_support", 8),
            support_shrinkage=oofguidancecfg.get("support_shrinkage", 12.0),
            max_rules_per_query=oofguidancecfg.get("max_rules_per_query", 8),
            alphas=oofguidancecfg.get("alpha_grid", [0.0, 0.05, 0.1, 0.2, 0.5]),
            target_smoothing=oofguidancecfg.get("target_smoothing", 0.05),
            residual_clip=oofguidancecfg.get("residual_clip", 4.0), seed=seed)
        out, oof_residual_guidance_summary, oof_preview, oof_controls = guidance.fit_apply(
            sym["train"], sym["train"], oof_logits, train_y,
            sym["val"], val_base, val_y, sym["test"], out, test_y)
        main = oof_residual_guidance_summary["main"]
        oof_residual_guidance_summary.update({
            "method": variant, "task": task, "n_oof_folds": len(oof_folds),
            "oof_fold_training": oof_folds,
            "validation_used_only_for_alpha_selection": True,
            "active_on_test": bool(main["selected_alpha"] > 0),
            "foundation_output_modified": bool(main["selected_alpha"] > 0),
            "ordered_sax_words": int(oofguidancecfg.get("ordered_sax_words", 24)),
        })
        gates = np.full(len(data.test), main["selected_alpha"], dtype=np.float32)
        write_json(oof_residual_guidance_summary, cell_dir / "oof_symbolic_residual_guidance.json")
        write_json({**oof_preview, "symbolic_vocabulary": extractor.summary()},
                   cell_dir / "oof_symbolic_residual_guidance_preview.json")
        log.info("[%s] OOF residual rules: alpha=%.3f rules=%d coverage=%.3f folds=%d", variant,
                 main["selected_alpha"], main["n_rules"], main["test_rule_coverage"], len(oof_folds))
    if use_selective_guidance:
        # Phase 8: symbols never enter the frozen UniTS backbone.  They can
        # only add a sparse, validation-selected correction after the base
        # prediction is made.  The validation block is split before mining:
        # early half discovers rule consequents; later half chooses alpha.
        val_base, _, _ = predict(runner, "val", ecfg.get("batch_size", 64))
        n_val = len(data.val)
        split = max(1, min(n_val - 1, n_val // 2))
        if task == "classification":
            val_y = data.val.target(np.arange(n_val)).reshape(-1)
            test_y = data.test.target(np.arange(len(data.test))).reshape(-1)
            guidance = ClassificationRuleGuidance(
                ordered_words=guidancecfg.get("ordered_sax_words", 24),
                ngram=guidancecfg.get("ngram", 3),
                miner_events_per_channel=guidancecfg.get("miner_events_per_channel", 2),
                activation_quantile=guidancecfg.get("miner_activation_quantile", 0.75),
                min_support=guidancecfg.get("min_support", 4),
                support_shrinkage=guidancecfg.get("support_shrinkage", 8.0),
                max_rules_per_query=guidancecfg.get("max_rules_per_query", 8),
                alphas=guidancecfg.get("alpha_grid", [0.0, 0.05, 0.1, 0.2, 0.5]), seed=seed)
            out, selective_guidance_summary, guidance_preview, guidance_outputs = guidance.fit_apply(
                sym["train"], sym["val"][:split], val_y[:split],
                sym["val"][split:], val_base[split:], val_y[split:],
                sym["test"], out, test_y)
            main = selective_guidance_summary["main"]
            gates = np.full(len(data.test), main["selected_alpha"], dtype=np.float32)
            selective_guidance_summary.update({
                "method": variant, "task": task,
                "foundation_output_modified": bool(main["selected_alpha"] > 0),
                "active_on_test": bool(main["selected_alpha"] > 0),
                "ordered_sax_words": int(guidancecfg.get("ordered_sax_words", 24)),
                "validation_split_index": int(split),
            })
            log.info("[%s] selective class rules: alpha=%.3f rules=%d coverage=%.3f", variant,
                     main["selected_alpha"], main["n_rules"], main["test_rule_coverage"])
        elif task == "forecasting":
            val_y = data.val.target(np.arange(n_val))
            test_y = data.test.target(np.arange(len(data.test)))
            guidance = HorizonResidualRules(
                ordered_words=guidancecfg.get("ordered_sax_words", 24),
                ngram=guidancecfg.get("ngram", 3),
                miner_events_per_channel=guidancecfg.get("miner_events_per_channel", 2),
                activation_quantile=guidancecfg.get("miner_activation_quantile", 0.75),
                min_support=guidancecfg.get("min_support", 32),
                support_shrinkage=guidancecfg.get("support_shrinkage", 24.0),
                max_rules_per_query=guidancecfg.get("max_rules_per_query", 8),
                weights=guidancecfg.get("alpha_grid", [0.0, 0.05, 0.1, 0.2]),
                mae_relative_tolerance=guidancecfg.get("validation_mae_relative_tolerance", 0.01), seed=seed)
            out, residual_summary, guidance_preview = guidance.fit_apply(
                sym["train"], sym["val"][:split], val_base[:split], val_y[:split],
                sym["val"][split:], val_base[split:], val_y[split:],
                sym["test"], out, test_y)
            main = residual_summary["main"]
            selective_guidance_summary = {
                "policy": "validation_gated_event_conditioned_horizon_residual_rules",
                "discovery_split": "early_chronological_validation_block",
                "selection_split": "later_chronological_validation_block",
                "test_targets_never_used_for_rule_mining_or_alpha_selection": True,
                "method": variant, "task": task, "main": main,
                "controls": residual_summary["controls"],
                "active_on_test": bool(main["selected_alpha"] > 0),
                "foundation_output_modified": bool(main["selected_alpha"] > 0),
                "ordered_sax_words": int(guidancecfg.get("ordered_sax_words", 24)),
                "validation_split_index": int(split),
            }
            gates = np.full(len(data.test), main["selected_alpha"], dtype=np.float32)
            log.info("[%s] selective forecast rules: alpha=%.3f rules=%d coverage=%.3f", variant,
                     main["selected_alpha"], main["n_rules"], main["test_rule_coverage"])
        else:
            # A score-changing anomaly correction requires labelled validation
            # anomalies.  Our standard SMAP split has no such calibration
            # labels, so applying a rule would be pseudo-supervised leakage.
            # Record an auditable abstention rather than fabricate guidance.
            selective_guidance_summary = {
                "policy": "validation_gated_event_conditioned_anomaly_score_rules",
                "method": variant, "task": task, "active_on_test": False,
                "foundation_output_modified": False,
                "abstained": True,
                "reason": "standard anomaly validation split has no labelled anomalies for correction calibration",
                "required_for_active_guidance": "labelled validation anomaly targets or a separately pre-registered synthetic calibration protocol",
            }
            guidance_preview = {"query_split": "test", "matches": []}
            log.info("[%s] selective anomaly guidance abstained: no labelled validation anomalies", variant)
        write_json(selective_guidance_summary, cell_dir / "selective_rule_guidance.json")
        write_json({**guidance_preview, "symbolic_vocabulary": extractor.summary()},
                   cell_dir / "selective_rule_guidance_preview.json")
    if use_behavior_audit:
        # Symbols never enter UniTS and `out` is never changed below.  We mine
        # only validation behaviour and use held-out targets once for scoring.
        val_base, _, _ = predict(runner, "val", ecfg.get("batch_size", 64))
        if task == "classification":
            val_y = data.val.target(np.arange(len(data.val))).reshape(-1)
            test_y = data.test.target(np.arange(len(data.test))).reshape(-1)
            discovery_target = (val_base.argmax(1) == val_y).astype(np.int8)
            test_target = (out.argmax(1) == test_y).astype(np.int8)
            behavior = "correct_foundation_model_classification"
        elif task == "forecasting":
            val_y = data.val.target(np.arange(len(data.val)))
            test_y = data.test.target(np.arange(len(data.test)))
            axes = tuple(range(1, val_base.ndim))
            val_error = ((val_base - val_y) ** 2).mean(axis=axes)
            high_q = float(behaviorcfg.get("forecast_high_error_quantile", 0.75))
            error_threshold = float(np.quantile(val_error, high_q))
            test_error = ((out - test_y) ** 2).mean(axis=axes)
            discovery_target = (val_error >= error_threshold).astype(np.int8)
            # Forecast loss scale can differ materially between a validation
            # block and a later test block.  The predeclared *behaviour* is a
            # relative high-error regime (top quartile), so the held-out label
            # is ranked within the held-out error distribution.  This uses test
            # targets only for final scoring; it never affects rules/threshold.
            test_target = (test_error >= np.quantile(test_error, high_q)).astype(np.int8)
            behavior = "high_foundation_model_forecast_error"
        else:
            val_score = anomaly_window_scores(data.val, val_base, model).mean(1)
            test_score = anomaly_window_scores(data.test, out, model).mean(1)
            score_threshold = float(np.quantile(val_score, behaviorcfg.get("anomaly_high_score_quantile", 0.90)))
            discovery_target = (val_score >= score_threshold).astype(np.int8)
            test_target = (test_score >= score_threshold).astype(np.int8)
            behavior = "high_foundation_model_anomaly_score"
        audit = ContrastiveBehaviorRules(
            ordered_words=behaviorcfg.get("ordered_sax_words", 24),
            ngram=behaviorcfg.get("ngram", 3),
            miner_events_per_channel=behaviorcfg.get("miner_events_per_channel", 2),
            activation_quantile=behaviorcfg.get("miner_activation_quantile", 0.75),
            min_support=behaviorcfg.get("min_support", 4),
            min_confidence=behaviorcfg.get("min_confidence", 0.55),
            min_growth=behaviorcfg.get("min_growth", 1.25),
            support_shrinkage=behaviorcfg.get("support_shrinkage", 8.0),
            max_rules_per_query=behaviorcfg.get("max_rules_per_query", 8), seed=seed)
        audit_result = audit.fit_apply(sym["train"], sym["val"], discovery_target,
                                       sym["test"], test_target, behavior=behavior)
        behavior_audit_summary = audit_result["summary"]
        behavior_audit_summary.update({"method": variant, "prediction_preserving": True,
                                       "foundation_output_modified": False})
        if task == "forecasting":
            behavior_audit_summary["held_out_target_protocol"] = (
                "top_quantile_within_held_out_per_window_mse; used_only_for_final_evaluation")
        write_json(behavior_audit_summary, cell_dir / "contrastive_behavior_audit.json")
        write_json({**audit_result["preview"], "symbolic_vocabulary": extractor.summary()},
                   cell_dir / "contrastive_behavior_audit_preview.json")
        # Save model inputs/outputs plus the highest-scoring held-out rule
        # matches.  PNGs are rendered by a separate script after the run;
        # this keeps training and visualisation independent and inspectable.
        test_events = audit.events(sym["test"])
        test_rule_score, test_matches = audit._score(test_events)
        write_behavior_visualization_payload(
            cell_dir=cell_dir, task=task, variant=variant, data=data, out=out,
            test_target=test_target, test_rule_score=test_rule_score,
            test_matches=test_matches, behavior_target=test_target,
            sym_meta=sym_meta, extractor=extractor, behavior=behavior, cfg=cfg,
        )
        held = behavior_audit_summary["held_out_behavior_metrics"]
        control = behavior_audit_summary["permuted_behavior_target_control"]["held_out_metrics"]
        log.info("[%s] contrastive audit: %s rules=%d heldout_pr_auc=%.4f permuted=%.4f", variant,
                 behavior, behavior_audit_summary["n_rules"], held["pr_auc"], control["pr_auc"])
    if external_window_scores is None:
        metrics, window_scores = task_metrics(data, task, out, model)
    else:
        metrics, window_scores = anomaly_metrics_from_window_scores(data.test, external_window_scores), external_window_scores
    if use_rule_evidence:
        # Evaluation-only perturbation metrics. The held-out targets here are
        # never used by the rule miner or validation weight selection.
        faithfulness_metrics = {}
        for name, counterfactual in evidence_result.counterfactuals.items():
            counter_metrics, _ = task_metrics(data, task, counterfactual, model)
            faithfulness_metrics[name] = {key: float(value) for key, value in counter_metrics.items()}
        rule_evidence_summary["faithfulness"]["counterfactual_test_metrics"] = faithfulness_metrics
        write_json(rule_evidence_summary, cell_dir / "symbolic_rule_evidence.json")
    # A non-zero learned gate is not itself proof of guidance.  For every
    # trainable symbolic arm, compare its normal output with the exact same
    # trained model evaluated with its symbolic gate forced to zero.  This does
    # not compare against the separately trained baseline arm; it answers the
    # narrower causal question: did the symbolic branch change this model's
    # held-out output at all?
    symbolic_ablation = None
    if variant != "baseline" and continuation_memory is None and not use_retrieval_mixer and not use_task_evidence and not use_rule_evidence and not use_behavior_audit and not use_selective_guidance and not use_oof_residual_guidance:
        zero_out, _, _ = predict(runner, "test", ecfg.get("batch_size", 64), force_gate=0.0)
        zero_metrics, _ = task_metrics(data, task, zero_out, model)
        delta = np.asarray(out, dtype=np.float64) - np.asarray(zero_out, dtype=np.float64)
        symbolic_ablation = {
            "protocol": "same_trained_model_force_gate_zero",
            "zero_gate_metrics": {k: float(v) for k, v in zero_metrics.items()},
            "mean_absolute_output_change": float(np.abs(delta).mean()),
            "max_absolute_output_change": float(np.abs(delta).max()),
        }
        write_json(symbolic_ablation, cell_dir / "symbolic_gate_ablation.json")
        log.info("[%s] symbolic gate ablation: mean_abs_output_change=%.6g", variant,
                 symbolic_ablation["mean_absolute_output_change"])
    infer = head_s + (timings.get("path_a_encode_test_s", 0.0)) + timings.get("symbolic_transform_test_s", 0.0)
    timings["infer_s"] = infer
    timings["infer_ms_per_sample"] = 1000 * infer / max(1, len(data.test))
    gsum = gate_summary(None if gates is None else torch.as_tensor(gates))
    # ---------------- explanations
    expl = {}
    do_explain = continuation_memory is None and not use_retrieval_mixer and not use_task_evidence and not use_rule_evidence and not use_behavior_audit and not use_selective_guidance and not use_oof_residual_guidance and variant != "baseline" and ecfg.get("explain", True) and (
        task in ("classification", "anomaly") or ecfg.get("explain_forecasting", False))
    if do_explain:
        records, problems = explain(runner, extractor, sym_meta, task, variant, out, window_scores, ecfg,
                                    ecfg.get("explain_batch_size", 32))
        (cell_dir / "explanations.jsonl").unlink(missing_ok=True)
        write_jsonl(records, cell_dir / "explanations.jsonl", append=False)
        write_json(records[: ecfg.get("preview", 10)], cell_dir / "explanations_preview.json")
        expl = {"n_records": len(records), "n_verified": len(records), "n_problems": len(problems),
                "problems_sample": problems[:5], **placeholder_check(records)}
        log.info("[%s] explanations: %d records, %d with problems", variant, len(records), len(problems))
    result = {
        "timestamp": dt.datetime.now().isoformat(), "phase": cfg["experiment"].get("phase"),
        "experiment": cfg["experiment"].get("name"), "backbone": backbone, "model_name": mcfg["name"],
        "regime": regime, "task": task, "dataset": data.name, "variant": variant, "seed": seed,
        "metrics": {k: float(v) for k, v in metrics.items()}, "gate": gsum, "timings": timings,
        "continuation_memory": continuation,
        "retrieval_mixer": retrieval_mixer_summary,
        "task_conditioned_evidence": task_evidence_summary,
        "symbolic_rule_evidence": rule_evidence_summary,
        "selective_rule_guidance": selective_guidance_summary,
        "oof_symbolic_residual_guidance": oof_residual_guidance_summary,
        "contrastive_behavior_audit": behavior_audit_summary,
        "symbolic_ablation": symbolic_ablation,
        "zero_shot_metrics": zero_shot, "explanations": expl,
        "train_state": {k: st[k] for k in ("epoch", "global_step", "best", "stopped")},
        "n_train": len(data.train), "n_val": len(data.val), "n_test": len(data.test),
        "symbolic": None if extractor is None else extractor.summary(),
        "anomaly_threshold_protocol": ANOMALY_THRESHOLD_PROTOCOL if task == "anomaly" else None,
        "config_hash": cfg_hash(cfg), "cell_dir": str(cell_dir), "device": str(device),
    }
    append_jsonl(result, out_dir / "results.jsonl")
    write_json({"finished": result["timestamp"], "metrics": result["metrics"]}, cell_dir / "done.json")
    write_report(out_dir)
    log.info("[%s] %s | %s", variant, data.name, "  ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items()
                                                         if isinstance(v, float)))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True, help="persistent storage (Drive / volume)")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt / skip finished cells")
    ap.add_argument("--fresh", action="store_true", help="delete existing cell checkpoints and restart")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides (recorded)")
    ap.add_argument("--data-dir", default=None, help="dataset cache (default <output-dir>/data_cache)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-ephemeral", action="store_true")
    ap.add_argument("--debug-exit-after-steps", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out_dir / "logs" / "run.log", encoding="utf-8")], force=True)
    check_persistent(out_dir, args.allow_ephemeral)
    cfg = load_config(args.config, args.set)
    cfg["_config_file"] = str(args.config)
    exp = cfg["experiment"]
    if exp["regime"] not in ("R1", "R1.5", "R2"):
        raise SystemExit("experiment.regime must be R1, R1.5, or R2")
    data_dir = Path(args.data_dir or out_dir / "data_cache")
    log.info("config %s (hash %s) regime %s -> %s", args.config, cfg_hash(cfg), exp["regime"], out_dir)
    for cell in exp["cells"]:
        task, ds = cell["task"], cell["dataset"]
        dkw = {**cfg.get("data", {}).get(task, {}), **cell.get("data", {})}
        seq_len = cfg["model"].get("seq_len", 512)
        if ds == "synthetic":
            data = synthetic_task(task, **dkw)
        else:
            data = load_task(task, ds, data_dir, seq_len=seq_len, seed=exp.get("seed", 0), **dkw)
        log.info("cell %s/%s: train %d val %d test %d channels %d", task, ds, len(data.train), len(data.val),
                 len(data.test), data.n_channels)
        cache = PathACache()
        for variant in cell.get("variants", exp.get("variants", VARIANTS)):
            if variant not in VARIANTS:
                raise SystemExit(f"unknown variant {variant}")
            run_variant(cfg, data, task, variant, out_dir, cache, args)
    write_report(out_dir)
    log.info("all cells done; report in %s", out_dir / "report")


if __name__ == "__main__":
    main()
