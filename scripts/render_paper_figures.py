#!/usr/bin/env python3
"""Clean, paper-ready interpretability figures (PNG only; --pdf adds PDF copies).

  python scripts/render_paper_figures.py --data-dir /data/ts_cache \
      --phase10-root /data/ts_runs/phase10_symbolic_evidence_heads \
      --gated-root   /data/ts_runs/paperA_gated_fusion \
      --out /data/ts_runs/paper_figures

Writes three folders under --out:
  anomaly/         series with the true anomaly and the rare pattern, plus the two scores
  classification/  input series with the matched words and their push toward the predicted class
  forecasting/     input window with matched words, what usually followed each word, forecast vs truth
No model is retrained: anomaly and classification read saved records; forecasting
refits the symbolic vocabulary on the training split and reads saved UniTS forecasts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MINER = {"fastshapelets": "FastShapelets", "bossst": "BOSS-ST"}
WORD_COLORS = ["#d62728", "#9467bd", "#2ca02c"]  # never blue: blue is UniTS
LABEL_BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
DIRECTION = {0: "fall", 1: "flat", 2: "rise"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, path: Path, pdf: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def label_spans(labels):
    lab = np.asarray(labels) > 0
    edges = np.diff(np.r_[0, lab.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


# =========================================================================== anomaly
def plot_anomaly(ex: dict, dataset: str, miner: str, path: Path, pdf: bool = False):
    """Top: the signal, the true anomaly and the rare pattern.  Bottom: the two scores."""
    a = int(ex["segment_start"])
    sig = np.asarray(ex["series"])[0]
    x = np.arange(a, a + len(sig))
    fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    for s, e in label_spans(ex["labels_segment"]):
        for ax in (top, bot):
            ax.axvspan(x[s], x[e - 1], color="#ffbf00", alpha=0.25, lw=0)
    top.plot(x, sig, color="black", lw=1.2)
    top.axvspan(ex["start"], ex["end"] - 1, facecolor="none", edgecolor="#d62728", lw=2)
    seen = "never appeared" if ex["train_count"] == 0 else f"appeared only {ex['train_count']} times"
    top.set_title(f"{dataset}: the red box is the rare pattern '{ex['word']}', which {seen} "
                  f"in {ex['train_total']:,} normal windows", fontsize=9.5, loc="left")
    top.set_ylabel("value")
    top.plot([], [], color="#ffbf00", alpha=0.5, lw=8, label="true anomaly")
    top.plot([], [], color="#d62728", lw=2, label="rare pattern")
    top.legend(fontsize=8, loc="upper right", frameon=False)
    fm = ex.get("fm_q_segment", ex.get("fm_z_segment"))
    sym = ex.get("symbolic_q_segment", ex.get("symbolic_z_segment"))
    bot.plot(x, fm, color="#1f77b4", lw=1, label="UniTS (reconstruction error)")
    bot.plot(x, sym, color="#d62728", lw=1.6, label=f"Symbolic (rare pattern, {MINER.get(miner, miner)})")
    bot.set_ylabel("anomaly score")
    bot.set_xlabel("time")
    bot.legend(fontsize=8, loc="upper right", frameon=False)
    save(fig, path, pdf)


def render_anomaly(root: Path, out: Path, seed: int, n: int, pdf: bool):
    count = 0
    for f in sorted(root.rglob(f"anomaly/*/seed_{seed}/*_readout.json")):
        res = json.loads(f.read_text(encoding="utf-8"))
        for k, ex in enumerate(res.get("examples", [])[:n]):
            plot_anomaly(ex, res["dataset"], res["variant"],
                         out / "anomaly" / f"{res['dataset']}_{res['variant']}_{k + 1}", pdf)
            count += 1
    print(f"anomaly: {count} figures")


# =========================================================================== classification
def plot_classification(x: np.ndarray, rec: dict, dataset: str, miner: str, class_names, path: Path,
                        pdf: bool = False, max_words: int = 3):
    words = [w for w in rec["top_words"] if w.get("location_informative", True)][:max_words]
    chans = list(dict.fromkeys(w["channel"] for w in words)) or [0]
    fig, axes = plt.subplots(len(chans), 1, figsize=(8, 2.4 * len(chans) + 0.6), sharex=True, squeeze=False)
    name = lambda c: class_names[c] if class_names and c < len(class_names) else str(c)  # noqa: E731
    pred, true = int(rec["prediction"]), int(rec["target"])
    verdict = "correct" if pred == true else f"wrong, true class {name(true)}"
    for ax, c in zip(axes[:, 0], chans):
        ax.plot(x[c], color="black", lw=1.1)
        ax.set_ylabel(f"channel {c}")
        line = 0
        for i, w in enumerate(words):
            if w["channel"] != c:
                continue
            s, e = w["location"]["start"], w["location"]["end"]
            col = WORD_COLORS[i % len(WORD_COLORS)]
            ax.axvspan(s, e - 1, color=col, alpha=0.18, lw=0)
            ax.plot(np.arange(s, e), x[c][s:e], color=col, lw=2.2)
            sign = "supports" if w["attribution"] >= 0 else "argues against"
            ax.annotate(f"'{w['word']}' {sign} class {name(pred)}", xy=(0.01, 0.97 - 0.12 * line),
                        xycoords="axes fraction", fontsize=8, color=col, va="top", bbox=LABEL_BOX)
            line += 1
    axes[0, 0].set_title(f"{dataset} ({MINER.get(miner, miner)}): predicted class {name(pred)} ({verdict}). "
                         "Coloured parts are the matched patterns.", fontsize=9.5, loc="left")
    axes[-1, 0].set_xlabel("time")
    save(fig, path, pdf)


def render_classification(gated_root: Path, data_dir: Path, out: Path, seed: int, n: int, pdf: bool):
    from symtsfm.data.datasets import load_task

    cache, count = {}, 0
    for f in sorted(gated_root.rglob(f"seed_{seed}/units/R1/classification_*/*/explanations.jsonl")):
        variant, dataset = f.parent.name, f.parent.parent.name.split("_", 1)[1]
        if dataset not in cache:
            cache[dataset] = load_task("classification", dataset, data_dir, seq_len=512, val_fraction=0.1, seed=seed)
        data = cache[dataset]
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Fixed rule: correctly classified samples whose evidence is located, strongest evidence first.
        good = [r for r in recs if int(r["prediction"]) == int(r["target"])
                and r["top_words"] and r["top_words"][0].get("location_informative", True)]
        good.sort(key=lambda r: -r["top_words"][0]["attribution_abs"])
        for k, rec in enumerate(good[:n]):
            x = data.test.x_raw(np.array([int(rec["sample_id"])]))[0]
            plot_classification(x, rec, dataset, variant, data.info.get("classes"),
                                out / "classification" / f"{dataset}_{variant}_{k + 1}", pdf)
            count += 1
    print(f"classification: {count} figures")


# =========================================================================== forecasting
def word_tendencies(presence: np.ndarray, labels: np.ndarray):
    """presence [n, C, K] (bool), labels [n, C] in {0,1,2} -> counts [C, K, 3]."""
    n, C, K = presence.shape
    out = np.zeros((C, K, 3), dtype=np.int64)
    for c in range(C):
        for d in range(3):
            out[c, :, d] = (presence[:, c, :] & (labels[:, c, None] == d)).sum(0)
    return out


def plot_forecast(x: np.ndarray, truth: np.ndarray, forecast: np.ndarray, words: list[dict], dataset: str,
                  channel_name: str, miner: str, actual: str, path: Path, pdf: bool = False):
    L, H = len(x), len(truth)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(np.arange(L), x, color="black", lw=1.1, label="input (history)")
    ax.plot(np.arange(L, L + H), truth, color="#7f7f7f", lw=1.1, label="what actually happened")
    ax.plot(np.arange(L, L + H), forecast, color="#1f77b4", lw=1.4, label="UniTS forecast")
    ax.axvline(L - 0.5, color="black", lw=0.6, ls=":")
    for i, w in enumerate(words):
        col = WORD_COLORS[i % len(WORD_COLORS)]
        s, e = w["start"], w["end"]
        ax.axvspan(s, e - 1, color=col, alpha=0.18, lw=0)
        ax.plot(np.arange(s, e), x[s:e], color=col, lw=2.2)
        ax.annotate(f"'{w['word']}': in training, followed by a {w['direction']} "
                    f"{w['share']:.0%} of the time (n={w['n']:,})",
                    xy=(0.01, 0.97 - 0.09 * i), xycoords="axes fraction", fontsize=8, color=col, va="top",
                    bbox=LABEL_BOX)
    ax.set_title(f"{dataset}, {channel_name} ({MINER.get(miner, miner)}): matched patterns in the history. "
                 f"Actual next {H} steps: {actual}", fontsize=9.5, loc="left")
    ax.set_xlabel("time")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    save(fig, path, pdf)


def render_forecasting(root: Path, data_dir: Path, config: Path, out: Path, seed: int, n: int, pdf: bool,
                       min_support: int = 30, n_candidates: int = 300):
    import yaml

    from symtsfm.data.datasets import load_task
    from symtsfm.symbolic.evidence_heads import fit_extractor

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    count = 0
    for f in sorted(root.rglob(f"forecasting/*/seed_{seed}/fm/fm_outputs.npz")):
        dataset = f.parent.parent.parent.name
        data = load_task("forecasting", dataset, data_dir, seq_len=cfg["model"]["seq_len"], seed=seed,
                         **cfg["data"]["forecasting"])
        forecast_all = np.load(f)["test_out"]
        names = data.info.get("channel_names") or [f"channel {c}" for c in range(data.n_channels)]
        for miner in cfg["experiment"].get("variants", ["fastshapelets", "bossst"]):
            ext = fit_extractor(data, miner, cfg["symbolic"], seed)
            # What followed each word in training: evenly spaced training windows (training data only).
            tr_idx = np.linspace(0, len(data.train) - 1, min(len(data.train), 4000)).round().astype(int)
            _, meta_tr = ext.transform(data.train.x_raw(tr_idx))
            counts = word_tendencies(meta_tr["presence"] > 0, data.train.fit_labels(tr_idx))
            # Fixed rule for examples: evenly spaced test windows that contain a well-supported word.
            te_idx = np.linspace(0, len(data.test) - 1, min(len(data.test), n_candidates)).round().astype(int)
            _, meta_te = ext.transform(data.test.x_raw(te_idx))
            candidates = []
            for j, i in enumerate(te_idx):
                best = []
                for c in range(data.n_channels):
                    words = ext.words(c)
                    for k in range(ext.K):
                        tot = counts[c, k].sum()
                        if words[k]["word"] == "<none>" or tot < min_support or not meta_te["presence"][j, c, k]:
                            continue
                        d = int(counts[c, k].argmax())
                        start = int(meta_te["location"][j, c, k])
                        best.append({"channel": c, "word": words[k]["word"], "direction": DIRECTION[d],
                                     "share": counts[c, k, d] / tot, "n": int(tot),
                                     "start": start, "end": start + ext.window_})
                if best:
                    best.sort(key=lambda w: -w["share"])
                    c = best[0]["channel"]
                    candidates.append((int(i), c, [w for w in best if w["channel"] == c][:2]))
            if not candidates:
                continue
            for k, pos in enumerate(np.linspace(0, len(candidates) - 1, n).round().astype(int)):
                i, c, words = candidates[pos]
                actual = DIRECTION[int(data.test.fit_labels(np.array([i]))[0, c])]
                plot_forecast(data.test.x_raw(np.array([i]))[0, c], data.test.target(np.array([i]))[0, c],
                              forecast_all[i, c], words, dataset, str(names[c]), miner, actual,
                              out / "forecasting" / f"{dataset}_{miner}_{k + 1}", pdf)
                count += 1
    print(f"forecasting: {count} figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--phase10-root", type=Path, required=True)
    ap.add_argument("--gated-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--forecast-config", type=Path, default=None)
    ap.add_argument("--tasks", nargs="+", default=["anomaly", "classification", "forecasting"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2, help="examples per dataset and miner")
    ap.add_argument("--pdf", action="store_true", help="also write PDF copies (for the paper)")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_suite  # noqa: F401  (puts src/ on sys.path and installs the core-runner patches)

    fcfg = args.forecast_config or (Path(__file__).resolve().parents[1] / "configs" / "evidence_heads"
                                    / "units_forecasting_h96_r1_symbolic_reliability.yaml")
    if "anomaly" in args.tasks:
        render_anomaly(args.phase10_root, args.out, args.seed, args.n, args.pdf)
    if "classification" in args.tasks:
        render_classification(args.gated_root, args.data_dir, args.out, args.seed, args.n, args.pdf)
    if "forecasting" in args.tasks:
        render_forecasting(args.phase10_root, args.data_dir, fcfg, args.out, args.seed, args.n, args.pdf)


if __name__ == "__main__":
    main()
