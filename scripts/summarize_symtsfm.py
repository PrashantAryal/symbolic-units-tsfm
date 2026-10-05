#!/usr/bin/env python3
"""Aggregate Phase 10 results.jsonl files into paper tables (mean +/- std over seeds).

  python scripts/summarize_symtsfm.py --root /data/ts_runs/phase10_symbolic_evidence_heads

Writes <root>/summary.md and <root>/summary_<task>.csv.  Paired statistics:
per-seed sign counts and, across datasets, a one-sided Wilcoxon signed-rank
test of the per-dataset means (n = number of datasets; report it as such).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANOMALY_METRICS = ("f1", "pr_auc", "vus_pr", "vus_roc", "adjusted_best_f1", "top1_hit_rate")
STREAMS = ("fm_reconstruction", "symbolic_novelty", "fused", "fused_calibrated")
FUSIONS = {"fused": "fused z (λ=1)", "fused_calibrated": "fused calibrated"}
_COMMON = ("statistics", "embedding", "symbolic", "rank:fm_signal+statistics", "rank:fm_signal+symbolic",
           "rank:embedding+statistics", "rank:embedding+symbolic", "fm_signal+symbolic")
SCORERS = {
    "classification": ("random", "fm_confidence") + _COMMON,
    "forecasting": ("random", "context_volatility") + _COMMON,
}
BASE_SIGNAL = {"classification": "fm_confidence", "forecasting": "context_volatility"}


def load_rows(root: Path) -> list[dict]:
    rows, seen = [], {}
    for path in sorted(root.rglob("results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                seen[(r["task"], r["dataset"], r["variant"], r["seed"])] = r  # last write wins on reruns
    rows = list(seen.values())
    return rows


def ms(values, scale=1.0, digits=4):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float) * scale
    if not len(v):
        return "-"
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}" if len(v) > 1 else f"{v.mean():.{digits}f}"


def wilcoxon_greater(diffs):
    d = np.asarray([x for x in diffs if np.isfinite(x)])
    if len(d) < 3 or np.allclose(d, 0):
        return float("nan")
    from scipy.stats import wilcoxon
    # Exact signed-rank distribution (n = number of datasets is small); exact zeros are dropped.
    d = d[np.abs(d) > 1e-12]
    if len(d) < 3:
        return float("nan")
    try:
        return float(wilcoxon(d, alternative="greater", method="exact").pvalue)
    except ValueError:  # tied magnitudes: fall back to the normal approximation
        return float(wilcoxon(d, alternative="greater").pvalue)


def anomaly_section(rows, out_dir: Path) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        if r["task"] == "anomaly":
            groups[(r["dataset"], r["variant"])].append(r)
    if not groups:
        return []
    lines = ["## Anomaly detection: UniTS reconstruction vs symbolic novelty fusion", "",
             "Mean ± std over seeds. Δ = fusion − UniTS; `k/n` = seeds where the fusion beats UniTS. "
             "`fused z (λ=1)` is the pre-specified fusion; `fused calibrated` sums −log tail probabilities "
             "under held-out normal scores (added after a scale diagnostic). "
             "Best-F1 is the oracle-threshold diagnostic used in all earlier tables.", ""]
    csv_rows = []
    per_variant_means = defaultdict(lambda: defaultdict(list))
    for metric in ANOMALY_METRICS:
        lines += [f"### {metric}", "",
                  "| Dataset | Miner | UniTS | Symbolic only | fused z (λ=1) | Δ | wins | fused calibrated | Δ | wins |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for (ds, var), rs in sorted(groups.items()):
            vals = {s: [r["metrics"].get(s, {}).get(metric) for r in rs] for s in STREAMS}
            cells, row = [], {}
            for s in FUSIONS:
                d = [f - b for f, b in zip(vals[s], vals["fm_reconstruction"]) if f is not None and b is not None]
                wins = sum(x > 1e-12 for x in d)
                cells.append(f"{ms(vals[s])} | {ms(d)} | {wins}/{len(d)}")
                if d:
                    per_variant_means[(var, metric, s)]["diff"].append(float(np.nanmean(d)))
                row.update({f"{s}_delta_mean": float(np.nanmean(d)) if d else float("nan"), f"{s}_seeds_improved": wins})
            lines.append(f"| {ds} | {var} | {ms(vals['fm_reconstruction'])} | {ms(vals['symbolic_novelty'])} | "
                         + " | ".join(cells) + " |")
            csv_rows.append({"dataset": ds, "miner": var, "metric": metric, "n_seeds": len(rs),
                             **{f"{s}_mean": float(np.nanmean([v for v in vals[s] if v is not None] or [np.nan]))
                                for s in STREAMS},
                             **{f"{s}_std": float(np.nanstd([v for v in vals[s] if v is not None] or [np.nan]))
                                for s in STREAMS}, **row})
        lines.append("")
    lines += ["### Across datasets (one-sided exact Wilcoxon on per-dataset mean Δ, fusion > UniTS)", "",
              "| Miner | Fusion | Metric | datasets improved | p |", "|---|---|---|---:|---:|"]
    for (var, metric, s), dd in sorted(per_variant_means.items()):
        diffs = dd["diff"]
        lines.append(f"| {var} | {FUSIONS[s]} | {metric} | {sum(x > 0 for x in diffs)}/{len(diffs)} | "
                     f"{wilcoxon_greater(diffs):.4f} |")
    lines.append("")
    _write_csv(csv_rows, out_dir / "summary_anomaly.csv")
    return lines


def reliability_section(rows, task, out_dir: Path) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        if r["task"] == task:
            groups[(r["dataset"], r["variant"])].append(r)
    if not groups:
        return []
    scale = 100.0 if task == "classification" else 1.0
    unit = "error-rate × 100" if task == "classification" else "MSE"
    base = BASE_SIGNAL[task]
    lines = [f"## {task.capitalize()}: selective prediction with symbolic reliability heads", "",
             f"AURC ({unit}; lower is better), mean ± std over seeds. A random ranking has expected AURC = "
             "full-coverage risk. `perm p<.05` counts seeds whose symbolic head beats its permuted-target null.", ""]
    present = [s for s in SCORERS[task] if any(s in r["risk_coverage"] for rs in groups.values() for r in rs)]
    lines += ["| Dataset | Miner | " + " | ".join(present) + " | perm p<.05 |",
              "|---|---|" + "---:|" * len(present) + "---:|"]
    csv_rows = []
    across = defaultdict(list)
    for (ds, var), rs in sorted(groups.items()):
        cells = []
        for s in present:
            vals = [r["risk_coverage"].get(s, {}).get("aurc") for r in rs]
            cells.append(ms(vals, scale, 3 if task == "classification" else 4))
            csv_rows.append({"dataset": ds, "miner": var, "scorer": s, "n_seeds": len(rs),
                             "aurc_mean": float(np.nanmean([v for v in vals if v is not None])),
                             "aurc_std": float(np.nanstd([v for v in vals if v is not None])),
                             "risk_at_80_mean": float(np.nanmean([r["risk_coverage"].get(s, {}).get("risk_at_80", np.nan)
                                                                  for r in rs]))})
        sig = sum(r["symbolic_permutation_p"] < 0.05 for r in rs)
        lines.append(f"| {ds} | {var} | " + " | ".join(cells) + f" | {sig}/{len(rs)} |")

        def mean_aurc(s):
            return float(np.nanmean([r["risk_coverage"][s]["aurc"] for r in rs if s in r["risk_coverage"]]))

        def add(name, worse, better):
            if all(any(s in r["risk_coverage"] for r in rs) for s in (worse, better)):
                across[(var, name)].append(mean_aurc(worse) - mean_aurc(better))

        add("symbolic < random", "random", "symbolic")
        add("symbolic < statistics", "statistics", "symbolic")
        add("symbolic < embedding (black box)", "embedding", "symbolic")
        add(f"rank:fm_signal+symbolic < {base}", base, "rank:fm_signal+symbolic")
        add("rank:fm_signal+symbolic < rank:fm_signal+statistics", "rank:fm_signal+statistics", "rank:fm_signal+symbolic")
        add("rank:embedding+symbolic < embedding", "embedding", "rank:embedding+symbolic")
        add("rank:embedding+symbolic < rank:embedding+statistics", "rank:embedding+statistics", "rank:embedding+symbolic")
    lines += ["", "### Across datasets (one-sided Wilcoxon on per-dataset mean AURC reduction)", "",
              "| Miner | Comparison | datasets improved | p |", "|---|---|---:|---:|"]
    for (var, name), diffs in sorted(across.items()):
        lines.append(f"| {var} | {name} | {sum(x > 0 for x in diffs)}/{len(diffs)} | {wilcoxon_greater(diffs):.4f} |")
    lines.append("")
    _write_csv(csv_rows, out_dir / f"summary_{task}.csv")
    return lines


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"no results.jsonl rows under {args.root}")
    lines = ["# Phase 10: symbolic evidence heads for UniTS", "",
             f"{len(rows)} (task, dataset, miner, seed) records.", ""]
    lines += anomaly_section(rows, args.root)
    lines += reliability_section(rows, "classification", args.root)
    lines += reliability_section(rows, "forecasting", args.root)
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
