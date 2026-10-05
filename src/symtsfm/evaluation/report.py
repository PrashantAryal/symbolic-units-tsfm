"""Aggregate ``results.jsonl`` into baseline-vs-guided comparison tables and plots.

Regenerated after every completed cell by ``experiments.run`` (and runnable on its
own: ``python -m symtsfm.evaluation.report --output-dir <dir>``). Comparisons are grouped by
(backbone, regime, task, dataset): a guided model is only ever compared with the
baseline trained under the *same* regime, and every table states that regime.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from symtsfm.evaluation.metrics import ANOMALY_THRESHOLD_PROTOCOL, VUS_PROTOCOL

METRICS = {
    "classification": ["accuracy", "macro_f1", "precision", "recall", "pr_auc"],
    "forecasting": ["mse", "mae"],
    "anomaly": ["f1", "precision", "recall", "pr_auc", "fpr", "adjusted_best_f1", "vus_roc"],
}
LOWER_IS_BETTER = {"mse", "mae", "fpr"}
REGIME_TEXT = {"R1": "R1 - frozen backbone", "R1.5": "R1.5 - final-block partial fine-tune",
               "R2": "R2 - full fine-tune"}
VARIANT_ORDER = ["baseline", "fastshapelets", "bossst"]
TIMING_KEYS = ["symbolic_fit_s", "symbolic_transform_s", "path_a_encode_s", "train_s", "infer_s", "infer_ms_per_sample"]


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    latest = {}
    for r in rows:  # last record wins per cell/variant (re-runs overwrite)
        latest[(r["backbone"], r["regime"], r["task"], r["dataset"], r["variant"])] = r
    return list(latest.values())


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_groups(rows: list[dict]) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault((r["backbone"], r["regime"], r["task"], r["dataset"]), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99)
    return dict(sorted(groups.items()))


def table_rows(groups: dict) -> list[dict]:
    flat = []
    for (bb, regime, task, ds), rs in groups.items():
        regimes = {r["regime"] for r in rs}
        assert regimes == {regime}, f"mixed regimes inside one comparison: {regimes}"
        base = next((r for r in rs if r["variant"] == "baseline"), None)
        for r in rs:
            row = {"backbone": bb, "regime": regime, "task": task, "dataset": ds, "variant": r["variant"],
                   "model_name": r.get("model_name")}
            for m in METRICS[task]:
                v = r["metrics"].get(m)
                row[m] = v
                if base is not None and r is not base and v is not None and base["metrics"].get(m) is not None:
                    row[f"delta_{m}"] = v - base["metrics"][m]
            row["gate_mean"] = r.get("gate", {}).get("gate_mean")
            row["gate_std"] = r.get("gate", {}).get("gate_std")
            continuation = r.get("continuation_memory") or {}
            rule_evidence = r.get("symbolic_rule_evidence") or {}
            row["continuation_alpha"] = continuation.get("selected_alpha")
            row["continuation_val_mse"] = continuation.get("validation_selected_mse")
            row["rule_weight"] = rule_evidence.get("selected_weight")
            row["rule_coverage"] = rule_evidence.get("test_rule_coverage")
            for t in TIMING_KEYS:
                row[t] = r.get("timings", {}).get(t)
            row["explanations_checked"] = r.get("explanations", {}).get("n_verified")
            row["explanation_problems"] = r.get("explanations", {}).get("n_problems")
            flat.append(row)
    return flat


def to_markdown(groups: dict) -> str:
    out = ["# Baseline vs symbolic-guided comparison", "",
           f"Anomaly threshold protocol: {ANOMALY_THRESHOLD_PROTOCOL}.",
           f"Anomaly VUS protocol: {VUS_PROTOCOL}.",
           "Deltas are guided minus baseline under the same regime; for mse/mae/fpr lower is better.", ""]
    for (bb, regime, task, ds), rs in groups.items():
        ms = METRICS[task]
        base = next((r for r in rs if r["variant"] == "baseline"), None)
        out.append(f"## {bb} | {task} | {ds} | regime: {REGIME_TEXT.get(regime, regime)}")
        out.append(f"model: `{rs[0].get('model_name')}`" + ("" if base else "  **(no baseline in this regime yet: no deltas)**"))
        out.append("")
        hdr = ["variant"] + ms + ["gate mean", "memory alpha", "rule weight", "rule coverage", "sym fit s", "train s", "infer ms/sample"]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * len(hdr))
        for r in rs:
            cells = [r["variant"]]
            for m in ms:
                v = r["metrics"].get(m)
                s = _fmt(v)
                if base is not None and r is not base and v is not None and base["metrics"].get(m) is not None:
                    d = v - base["metrics"][m]
                    better = (d < 0) if m in LOWER_IS_BETTER else (d > 0)
                    s += f" ({'+' if d >= 0 else ''}{d:.4f}{' better' if better and abs(d) > 1e-12 else ''})"
                cells.append(s)
            t = r.get("timings", {})
            continuation = r.get("continuation_memory") or {}
            rule_evidence = r.get("symbolic_rule_evidence") or {}
            cells += [_fmt(r.get("gate", {}).get("gate_mean")),
                      _fmt(continuation.get("selected_alpha")),
                      _fmt(rule_evidence.get("selected_weight")),
                      _fmt(rule_evidence.get("test_rule_coverage")),
                      _fmt(t.get("symbolic_fit_s"), 1),
                      _fmt(t.get("train_s"), 1), _fmt(t.get("infer_ms_per_sample"), 2)]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def plot_groups(groups: dict, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#6b7280", "fastshapelets": "#2563eb", "bossst": "#d97706"}
    for (bb, regime, task, ds), rs in groups.items():
        ms = METRICS[task] + ["gate_mean"]
        fig, axes = plt.subplots(1, len(ms), figsize=(2.6 * len(ms), 2.8))
        for ax, m in zip(np.atleast_1d(axes), ms):
            vals = [(r.get("gate", {}).get("gate_mean") if m == "gate_mean" else r["metrics"].get(m)) for r in rs]
            names = [r["variant"] for r in rs]
            v = [np.nan if x is None else x for x in vals]
            ax.bar(names, v, color=[colors.get(n, "#999") for n in names])
            ax.set_title(m + (" (lower better)" if m in LOWER_IS_BETTER else ""), fontsize=9)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{bb} | {task} | {ds} | {REGIME_TEXT.get(regime, regime)}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{bb}_{regime}_{task}_{ds}.png", dpi=120)
        plt.close(fig)


def write_report(output_dir) -> Path:
    output_dir = Path(output_dir)
    rows = load_results(output_dir / "results.jsonl")
    rep = output_dir / "report"
    rep.mkdir(parents=True, exist_ok=True)
    groups = build_groups(rows)
    flat = table_rows(groups)
    if flat:
        import pandas as pd

        pd.DataFrame(flat).to_csv(rep / "results_table.csv", index=False)
    (rep / "results_table.md").write_text(to_markdown(groups), encoding="utf-8")
    plot_groups(groups, rep / "plots")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    print(write_report(ap.parse_args().output_dir))
