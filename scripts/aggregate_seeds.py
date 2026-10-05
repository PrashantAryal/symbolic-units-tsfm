#!/usr/bin/env python3
"""Aggregate seed-level result JSONL files into mean +/- standard-deviation tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

LOWER_IS_BETTER = {"mse", "mae", "fpr"}
METRICS = {
    "classification": ["accuracy", "precision", "recall", "macro_f1", "pr_auc"],
    "forecasting": ["mse", "mae"],
    "anomaly": ["precision", "recall", "f1", "pr_auc", "fpr", "adjusted_best_f1", "vus_roc"],
}


def read_rows(root: Path) -> list[dict]:
    latest = {}
    for path in root.rglob("results.jsonl"):
        seed = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (seed, row["backbone"], row["regime"], row["task"], row["dataset"], row["variant"])
            latest[key] = row
    return list(latest.values())


def flatten(rows: list[dict]) -> pd.DataFrame:
    flat = []
    for row in rows:
        item = {key: row.get(key) for key in ("backbone", "model_name", "regime", "task", "dataset", "variant", "seed")}
        item["gate_mean"] = row.get("gate", {}).get("gate_mean")
        item["gate_std"] = row.get("gate", {}).get("gate_std")
        for key, value in row.get("metrics", {}).items():
            item[key] = value
        flat.append(item)
    return pd.DataFrame(flat)


def summary(seed_level: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["backbone", "model_name", "regime", "task", "dataset", "variant"]
    numeric = [column for column in seed_level.columns if column not in id_cols + ["seed"]]
    grouped = seed_level.groupby(id_cols, dropna=False)
    mean = grouped[numeric].mean().add_suffix("_mean")
    std = grouped[numeric].std(ddof=1).add_suffix("_std")
    count = grouped.size().rename("n_seeds")
    return pd.concat([count, mean, std], axis=1).reset_index()


def markdown(table: pd.DataFrame) -> str:
    lines = ["# Five-seed aggregate", "", "Values are mean +/- sample standard deviation.", ""]
    for (backbone, task), group in table.groupby(["backbone", "task"], dropna=False):
        lines += [f"## {backbone} | {task}", ""]
        metrics = METRICS.get(task, [])
        header = ["dataset", "variant"] + metrics + ["gate_mean", "n_seeds"]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for _, row in group.sort_values(["dataset", "variant"]).iterrows():
            values = [str(row["dataset"]), str(row["variant"])]
            for metric in metrics + ["gate_mean"]:
                mean, std = row.get(metric + "_mean"), row.get(metric + "_std")
                if pd.isna(mean):
                    values.append("-")
                elif pd.isna(std):
                    values.append(f"{mean:.4f}")
                else:
                    values.append(f"{mean:.4f} +/- {std:.4f}")
            values.append(str(int(row["n_seeds"])))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input_dir)
    if not rows:
        raise SystemExit(f"No results.jsonl files below {args.input_dir}")
    out = args.input_dir / "aggregate"
    out.mkdir(exist_ok=True)
    seed_level = flatten(rows)
    aggregate = summary(seed_level)
    seed_level.to_csv(out / "seed_level.csv", index=False)
    aggregate.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(markdown(aggregate), encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()

