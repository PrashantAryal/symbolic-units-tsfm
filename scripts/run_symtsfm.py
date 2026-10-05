#!/usr/bin/env python3
"""Phase 10: task-specific symbolic evidence heads around frozen UniTS (R1).

For every (task, dataset, seed) this script
  1. trains the ordinary UniTS R1 model once (prompt tokens + task head; the
     backbone stays frozen) and saves its per-sample outputs;
  2. runs the symbolic evidence head for each miner (FastShapelets, BOSS-ST):
       anomaly          -> normal-dictionary symbolic novelty fused with the
                           UniTS reconstruction score (changes the anomaly score)
       classification   -> sparse symbolic reliability head, selective prediction
       forecasting      -> sparse symbolic reliability head, selective prediction
     The UniTS prediction path is never modified by a reliability head.

Example:
  python scripts/run_symtsfm.py --config configs/evidence_heads/units_anomaly_r1_symbolic_novelty.yaml \
      --output-dir /data/ts_runs/phase10_symbolic_evidence_heads/anomaly \
      --data-dir /data/ts_cache --units-checkpoint /data/checkpoints/units_x128_pretrain_checkpoint.pth \
      --seeds 0 1 2 3 4 --device cuda
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_suite  # noqa: E402  (puts src/ on sys.path; core-runner patches)
from symtsfm.evaluation import anomaly_protocol as rb_metrics  # noqa: E402
from symtsfm.data.datasets import load_task  # noqa: E402

log = logging.getLogger("phase10")
VARIANTS = ("fastshapelets", "bossst")


# =========================================================================== io helpers
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(obj, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=_jsonable) + "\n")


# =========================================================================== metrics
def anomaly_metric_fn(vus_buffer: int):
    """anomaly protocol plus VUS-PR from the same single vus call."""
    from sklearn.metrics import average_precision_score

    try:
        from vus.metrics import get_metrics
    except ImportError as exc:
        raise RuntimeError("Phase 10 anomaly metrics need the vus package") from exc

    def fn(scores_by_entity: dict, labels_by_entity: dict) -> dict:
        rows = []
        for entity, scores in scores_by_entity.items():
            labels = np.asarray(labels_by_entity[entity], dtype=np.int8)
            scores = np.asarray(scores, dtype=np.float64)
            if labels.min() == labels.max():
                continue
            f1, precision, recall, threshold = rb_metrics.best_f1_threshold(scores, labels)
            lo, hi = scores.min(), scores.max()
            norm = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
            vus = get_metrics(norm, labels.astype(np.int64), metric="vus",
                              slidingWindow=min(int(vus_buffer), len(scores) - 1))
            rows.append({"f1": f1, "precision": precision, "recall": recall,
                         "pr_auc": float(average_precision_score(labels, scores)),
                         "fpr": rb_metrics._raw_fpr(scores >= threshold, labels),
                         "adjusted_best_f1": rb_metrics.adjusted_best_f1(scores, labels),
                         "vus_roc": float(vus.get("VUS_ROC", np.nan)),
                         "vus_pr": float(vus.get("VUS_PR", np.nan))})
        names = ("f1", "precision", "recall", "pr_auc", "fpr", "adjusted_best_f1", "vus_roc", "vus_pr")
        if not rows:
            return {n: float("nan") for n in names} | {"n_entities_scored": 0}
        return {n: float(np.nanmean([r[n] for r in rows])) for n in names} | {"n_entities_scored": len(rows)}

    return fn


# =========================================================================== foundation model
def _predict_embed(core_run, runner, split, bs):
    try:
        out, embed, _ = core_run.predict_with_pooled(runner, split, bs)
        return out, embed
    except RuntimeError:
        out, _, _ = core_run.predict(runner, split, bs)
        return out, None


def train_foundation_model(cfg, data, task, seed, fm_dir: Path, device, resume: bool) -> dict:
    """Ordinary UniTS R1 training (sym_dim=0); returns the arrays the heads need."""
    from symtsfm.experiments import run as core_run

    npz = fm_dir / "fm_outputs.npz"
    if resume and npz.exists():
        z = np.load(npz)
        log.info("loaded saved foundation-model outputs %s", npz)
        return {k: z[k] for k in z.files}
    fm_dir.mkdir(parents=True, exist_ok=True)
    regime = cfg["experiment"]["regime"]
    tcfg, bs = cfg["train"], cfg.get("eval", {}).get("batch_size", 64)
    core_run.set_seed(seed)
    model = core_run.build_model(cfg, data, task, 0, seed).to(device)
    model.set_regime(regime)
    runner = core_run.Runner(model, data, None, None, device, task)
    t0 = time.perf_counter()
    state = core_run.train(runner, tcfg, fm_dir, seed, resume, None)
    out = {}
    if task == "classification":
        out["test_logits"], test_embed = _predict_embed(core_run, runner, "test", bs)
        _, fit_embed = _predict_embed(core_run, runner, "train", bs)
        if test_embed is not None and fit_embed is not None:
            out["test_embed"], out["fit_embed"] = test_embed, fit_embed
        oof_folds = int(cfg.get("reliability", {}).get("oof_folds", 5))
        out["fit_logits"], folds = core_run.out_of_fold_classification_logits(
            cfg, data, task, regime, device, tcfg, fm_dir, seed, oof_folds)
        write_json(folds, fm_dir / "oof_folds.json")
    elif task == "forecasting":
        out["test_out"], test_embed = _predict_embed(core_run, runner, "test", bs)
        out["fit_out"], fit_embed = _predict_embed(core_run, runner, "val", bs)
        if test_embed is not None and fit_embed is not None:
            out["test_embed"], out["fit_embed"] = test_embed, fit_embed
    else:
        for split in ("val", "test"):
            o, _, _ = core_run.predict(runner, split, bs)
            out[f"{split}_window_scores"] = core_run.anomaly_window_scores(getattr(data, split), o, model)
    out = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    np.savez_compressed(npz, **out)
    write_json({"train_state": {k: state[k] for k in ("epoch", "global_step", "best", "stopped")},
                "train_s": time.perf_counter() - t0, "regime": regime, "seed": seed,
                "saved_arrays": {k: list(v.shape) for k, v in out.items()}}, fm_dir / "fm_meta.json")
    del runner, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def foundation_metrics(task, data, fm) -> dict:
    from symtsfm.evaluation.metrics import classification_metrics, forecasting_metrics
    from symtsfm.symbolic.evidence_heads import softmax

    test = data.test
    idx = np.arange(len(test))
    if task == "classification":
        return classification_metrics(np.asarray(test.target(idx)).reshape(-1), softmax(fm["test_logits"]))
    if task == "forecasting":
        return forecasting_metrics(fm["test_out"], test.target(idx))
    return {}


def normal_training_series(dataset: str, data, data_dir: Path) -> dict | None:
    """Contiguous normal training series per entity, [C, T], for the novelty head.

    Same source and scaling as the loader that produced the test split; only
    the training prefix (all normal by construction of each benchmark) is read.
    """
    if hasattr(data.train, "series"):  # MSL / SMAP / SMD / PSM lazy route (already z-scored per entity)
        return {str(e): np.asarray(s, dtype=np.float32) for e, s in data.train.series.items()}
    from symtsfm.data.datasets import _UAD_RE, MOMENT_UAD_FILES

    if dataset in MOMENT_UAD_FILES:
        name = MOMENT_UAD_FILES[dataset]
        raw = np.loadtxt(Path(data_dir) / "moment_uad" / name, delimiter=",", dtype=np.float32, ndmin=2)
        train_end = int(_UAD_RE.match(name).group(2))
        return {dataset: raw[:train_end, 0][None, :]}
    from symtsfm.data.loaders import UCR_ANOMALY_DATASETS, _ucr_anomaly_entities

    if dataset in UCR_ANOMALY_DATASETS:
        return {str(eid): np.asarray(xtr, dtype=np.float32)[None, :]
                for eid, (xtr, _), _ in _ucr_anomaly_entities(dataset, data_dir)}
    return None


# =========================================================================== main loop
def compact_row(task, res) -> dict:
    """The few numbers the summary tables need; the full record is in *_readout.json."""
    if task == "anomaly":
        return {"metrics": res["metrics"], "lambda_sensitivity": res["lambda_sensitivity_diagnostic"]}
    rc = res["risk_coverage"]
    return {"risk_coverage": {name: {k: v for k, v in r.items() if k != "curve"} for name, r in rc.items()},
            "symbolic_permutation_p": res["symbolic_permutation_control"]["p_value"],
            "symbolic_null_mean_aurc": res["symbolic_permutation_control"]["null_mean"],
            "head_sparsity": res["head_sparsity"], "task_info": res["task_info"]}


def run_cell(cfg, cell, seed, args, out_root: Path, metric_fn):
    from symtsfm.symbolic.evidence_heads import anomaly_readout, reliability_readout

    task, ds = cell["task"], cell["dataset"]
    dkw = {**cfg.get("data", {}).get(task, {}), **cell.get("data", {})}
    data = load_task(task, ds, args.data_dir, seq_len=cfg["model"].get("seq_len", 512), seed=seed, **dkw)
    log.info("=== %s/%s seed %d: train %d val %d test %d channels %d", task, ds, seed, len(data.train),
             len(data.val), len(data.test), data.n_channels)
    cell_dir = out_root / task / ds / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    fm = train_foundation_model(cfg, data, task, seed, cell_dir / "fm", args.device, args.resume)
    fm_metrics = {k: float(v) for k, v in foundation_metrics(task, data, fm).items()}
    fm_anomaly_metrics = None
    normal_series = normal_training_series(ds, data, args.data_dir) if task == "anomaly" else None
    for variant in cell.get("variants", cfg["experiment"].get("variants", VARIANTS)):
        path = cell_dir / f"{variant}_readout.json"
        if args.resume and path.exists() and not args.redo_readouts:
            log.info("skip finished %s", path)
            continue
        t0 = time.perf_counter()
        if task == "anomaly":
            res = anomaly_readout(data=data, variant=variant, ncfg=cfg["novelty"], seed=seed, fm=fm,
                                  metric_fn=metric_fn, fm_metrics=fm_anomaly_metrics,
                                  normal_series=normal_series)
            fm_anomaly_metrics = {k: v for k, v in res["metrics"]["fm_reconstruction"].items()
                                  if k != "top1_hit_rate"}
        else:
            res = reliability_readout(task=task, data=data, variant=variant, scfg=cfg["symbolic"],
                                      rcfg=cfg.get("reliability", {}), seed=seed, fm=fm)
        res.update({"task": task, "dataset": ds, "seed": seed, "fm_test_metrics": fm_metrics,
                    "readout_s": time.perf_counter() - t0, "config_name": cfg["experiment"]["name"],
                    "finished": dt.datetime.now().isoformat()})
        write_json(res, path)
        append_jsonl({"task": task, "dataset": ds, "seed": seed, "variant": variant,
                      "fm_test_metrics": fm_metrics, **compact_row(task, res)}, out_root / "results.jsonl")
        if task == "anomaly":
            m = res["metrics"]
            log.info("[%s] %s seed %d  F1 fm %.4f -> fused %.4f | VUS-PR fm %.4f -> fused %.4f | top1 hit %.2f -> %.2f",
                     variant, ds, seed, m["fm_reconstruction"]["f1"], m["fused"]["f1"],
                     m["fm_reconstruction"]["vus_pr"], m["fused"]["vus_pr"],
                     m["fm_reconstruction"]["top1_hit_rate"], m["fused"]["top1_hit_rate"])
        else:
            rc = res["risk_coverage"]
            base = "fm_confidence" if task == "classification" else "context_volatility"
            log.info("[%s] %s seed %d  AURC random %.4f | %s %.4f | symbolic %.4f (perm p %.3f) | fm+symbolic %.4f",
                     variant, ds, seed, rc["random"]["aurc"], base, rc[base]["aurc"], rc["symbolic"]["aurc"],
                     res["symbolic_permutation_control"]["p_value"], rc["fm_signal+symbolic"]["aurc"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--units-checkpoint", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="*", default=None, help="optional subset of the config's cells")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--redo-readouts", action="store_true",
                    help="recompute symbolic heads for finished cells; saved UniTS outputs are reused")
    args = ap.parse_args()
    args.config = args.config.resolve()
    out_root = args.output_dir.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out_root / "phase10.log", encoding="utf-8")], force=True)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["model"]["checkpoint"] = args.units_checkpoint
    vus_buffer = int(cfg.get("research", {}).get("vus_max_buffer", 512))
    run_suite.patch_core_runner(vus_buffer)  # chdir to the repository root, deterministic seeds, metric patches
    run_suite.write_environment(out_root)
    (out_root / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    metric_fn = anomaly_metric_fn(vus_buffer)
    cells = [c for c in cfg["experiment"]["cells"] if not args.datasets or c["dataset"] in args.datasets]
    for cell in cells:
        for seed in args.seeds:
            cfg["experiment"]["seed"] = seed
            run_cell(cfg, cell, seed, args, out_root, metric_fn)
    log.info("all cells done: %s", out_root)


if __name__ == "__main__":
    main()
