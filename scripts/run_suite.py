#!/usr/bin/env python3
"""Run the full-data, five-seed UniTS symbolic-adapter suite.

Examples:
  python scripts/run_suite.py --backbone units --output-dir /data/runs/units \
      --units-checkpoint /data/checkpoints/units_x128_pretrain_checkpoint.pth
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets the scripts run without installing the package

from symtsfm.data.datasets import load_task
from symtsfm.evaluation.anomaly_protocol import ANOMALY_THRESHOLD_PROTOCOL, VUS_PROTOCOL, anomaly_metrics


CONFIGS = {
    "units": {
        "classification": ROOT / "configs" / "units_classification_full.yaml",
        "forecasting": ROOT / "configs" / "units_forecasting_h96_full.yaml",
        "anomaly": ROOT / "configs" / "units_anomaly_full.yaml",
    },
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_environment(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {name: package_version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "torch", "transformers", "vus")},
        "anomaly_threshold_protocol": ANOMALY_THRESHOLD_PROTOCOL,
        "vus_protocol": VUS_PROTOCOL,
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    try:
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=60)
        (output_dir / "pip_freeze.txt").write_text(frozen, encoding="utf-8")
    except Exception as exc:  # metadata still makes the run usable if pip is unavailable
        (output_dir / "pip_freeze.txt").write_text(f"Unable to capture pip freeze: {exc}\n", encoding="utf-8")


def deterministic_seed(seed: int) -> None:
    """Paired seeds: same seed and deterministic data order for all three arms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_config(path: Path) -> int:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    research = config.get("research", {})
    if research.get("fast_mode", False):
        raise SystemExit(f"{path.name} enables FAST_MODE; full-data suite refuses to run it")
    fusion_mode = research.get("fusion_mode", "gated")
    if fusion_mode not in {"gated", "cross_attention", "horizon_cross_attention_residual",
                           "horizon_prehead_cross_attention", "horizon_location_prehead_cross_attention",
                           "continuation_memory", "embedding_retrieval_mixer", "task_conditioned_evidence",
                           "symbolic_rule_evidence", "prediction_preserving_symbolic_audit",
                           "selective_rule_guidance", "oof_symbolic_residual_guidance"}:
        raise SystemExit("fusion_mode must be 'gated', 'cross_attention', or "
                         "'horizon_cross_attention_residual', 'horizon_prehead_cross_attention', "
                         "'horizon_location_prehead_cross_attention', or "
                         "'continuation_memory', 'embedding_retrieval_mixer', or "
                         "'task_conditioned_evidence', 'symbolic_rule_evidence', or "
                         "'prediction_preserving_symbolic_audit', or "
                         "'selective_rule_guidance', or 'oof_symbolic_residual_guidance'.")
    for forbidden in ("max_train", "max_eval", "max_entities", "max_fit_series"):
        def contains(value):
            if isinstance(value, dict):
                return any(key == forbidden and item is not None or contains(item) for key, item in value.items())
            if isinstance(value, list):
                return any(contains(item) for item in value)
            return False
        if contains(config):
            raise SystemExit(f"{path.name} contains {forbidden}; this full-data runner refuses capped data")
    return int(research.get("vus_max_buffer", 512))


def patch_core_runner(vus_buffer: int) -> None:
    """Reuse core training code while replacing only data dispatch and anomaly metrics."""
    os.chdir(ROOT)  # relative paths in the configuration files (third_party/UniTS) are relative to the repository root
    from symtsfm.experiments import run as core_run
    import symtsfm.evaluation.metrics as core_metrics
    import symtsfm.evaluation.report as core_report

    core_run.load_task = load_task
    core_run.set_seed = deterministic_seed
    # The legacy runner passes scaler metadata positionally, whereas its metric
    # helper only declares two positional arguments. Forecast metrics are already
    # calculated on the standardised arrays, so ignore that unused metadata.
    core_run.forecasting_metrics = lambda prediction, target, *_unused: core_metrics.forecasting_metrics(
        prediction, target)
    core_run.anomaly_metrics = lambda scores, labels: anomaly_metrics(scores, labels, vus_max_buffer=vus_buffer)
    core_run.ANOMALY_THRESHOLD_PROTOCOL = ANOMALY_THRESHOLD_PROTOCOL
    core_metrics.ANOMALY_THRESHOLD_PROTOCOL = ANOMALY_THRESHOLD_PROTOCOL
    core_metrics.VUS_PROTOCOL = VUS_PROTOCOL
    core_report.ANOMALY_THRESHOLD_PROTOCOL = ANOMALY_THRESHOLD_PROTOCOL
    core_report.VUS_PROTOCOL = VUS_PROTOCOL


def run_one(config: Path, output_dir: Path, data_dir: Path, seed: int, device: str,
            resume: bool, checkpoint: str | None, extra_sets: list[str]) -> None:
    vus_buffer = validate_config(config)
    patch_core_runner(vus_buffer)
    from symtsfm.experiments import run as core_run

    overrides = [f"experiment.seed={seed}", *extra_sets]
    if checkpoint is not None:
        overrides.append(f"model.checkpoint={checkpoint}")
    args = ["--config", str(config), "--output-dir", str(output_dir), "--data-dir", str(data_dir),
            "--device", device, "--set", *overrides]
    if resume:
        args.insert(-len(overrides) - 1, "--resume")
    core_run.main(args)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--backbone", choices=CONFIGS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="persistent server volume")
    parser.add_argument("--data-dir", type=Path, default=None, help="shared persistent data cache")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--task", choices=("classification", "forecasting", "anomaly"), action="append")
    parser.add_argument("--config", type=Path, default=None,
                        help="custom config for exactly one --task; keeps V2 pilots separate from defaults")
    parser.add_argument("--units-checkpoint", default=None, help="required for the UniTS suite")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="additional core-config overrides; saved in each run metadata file")
    args = parser.parse_args()

    if args.backbone == "units" and not args.units_checkpoint:
        parser.error("--units-checkpoint is required for the UniTS suite")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but torch.cuda.is_available() is false")
    if args.config is not None:
        if not args.config.is_file():
            parser.error(f"custom config does not exist: {args.config}")
        # ``patch_core_runner`` later changes CWD to the sibling core project;
        # keep an absolute config path so custom pilots remain readable there.
        args.config = args.config.resolve()
        if not args.task or len(args.task) != 1:
            parser.error("--config requires exactly one --task")
        custom = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if custom.get("model", {}).get("backbone") != args.backbone:
            parser.error("custom config model.backbone must match --backbone")

    root = args.output_dir.expanduser().resolve()
    data_dir = (args.data_dir or root / "data_cache").expanduser().resolve()
    write_environment(root)
    tasks = args.task or list(CONFIGS[args.backbone])
    for task in tasks:
        config = args.config if args.config is not None else CONFIGS[args.backbone][task]
        for seed in args.seeds:
            seed_dir = root / task / f"seed_{seed}"
            print(f"\n=== {args.backbone} | {task} | seed {seed} -> {seed_dir} ===", flush=True)
            run_one(config, seed_dir, data_dir, seed, args.device, args.resume, args.units_checkpoint, args.set)


if __name__ == "__main__":
    main()
