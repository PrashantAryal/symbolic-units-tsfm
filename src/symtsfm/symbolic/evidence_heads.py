"""Phase 10: task-specific symbolic evidence heads around a frozen foundation model.

One symbolic dictionary family (FastShapelets/SAX or BOSS-ST/SFA) is fitted on
training data only and read out through a head matched to each task:

* anomaly detection  -> :func:`anomaly_readout`
      normal-dictionary novelty fused with the UniTS reconstruction score;
* classification / forecasting -> :func:`reliability_readout`
      sparse symbolic reliability head for selective prediction.

This module is NumPy/scikit-learn only.  The foundation-model outputs are
passed in as arrays (see ``scripts/run_symtsfm.py``), which keeps
the evidence heads testable without a GPU and guarantees that the UniTS
prediction path itself is never modified here.
"""
from __future__ import annotations

import time

import numpy as np

from symtsfm.symbolic.features import SymbolicFeatureExtractor
from symtsfm.symbolic.novelty import SymbolicNoveltyScorer, estimate_period, tail_surprisal, top_events
from symtsfm.symbolic.reliability import (
    SparseReliability,
    context_volatility,
    permutation_null,
    permutation_p_value,
    risk_coverage,
    summary_statistic_names,
    summary_statistics,
    symbolic_feature_names,
)

REPRESENTATION = {"fastshapelets": "sax", "bossst": "sfa"}


# =========================================================================== shared helpers
def _all(split):
    return np.arange(len(split))


def fit_extractor(data, variant: str, scfg: dict, seed: int) -> SymbolicFeatureExtractor:
    """Same fitting protocol as the earlier phases: training split only."""
    tr = data.train
    rng = np.random.default_rng(seed)
    n_fit = scfg.get("max_fit_series") or len(tr)
    idx = np.sort(rng.choice(len(tr), min(n_fit, len(tr)), replace=False))
    ext = SymbolicFeatureExtractor(
        method=variant, window=scfg["window"], word_len=scfg["word_len"], alphabet_size=scfg["alphabet_size"],
        top_k=scfg["top_k"], step=scfg.get("step", 1), max_fit_series=scfg.get("max_fit_series"), seed=seed,
        method_kwargs=scfg.get(variant, {}), chunk=scfg.get("chunk", 256))
    ext.fit_transform(tr.x_raw(idx), tr.fit_labels(idx))
    return ext


def transform(split, ext: SymbolicFeatureExtractor, chunk: int = 256, keep_meta: bool = False):
    feats, metas = [], []
    n = len(split)
    for s in range(0, n, chunk):
        f, m = ext.transform(split.x_raw(np.arange(s, min(n, s + chunk))))
        feats.append(f)
        if keep_meta:
            metas.append(m)
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]} if keep_meta else None
    return np.concatenate(feats), meta


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _pct_rank(s: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(np.asarray(s, dtype=np.float64)) / max(1, len(s))


def _jsonable_array(a, decimals: int = 4):
    return np.round(np.asarray(a, dtype=np.float64), decimals).tolist()


# =========================================================================== reliability head
def reliability_readout(*, task: str, data, variant: str, scfg: dict, rcfg: dict, seed: int, fm: dict) -> dict:
    """Selective classification / forecasting with a sparse symbolic reliability head.

    ``fm`` keys -- classification: ``fit_logits`` (out-of-fold train logits),
    ``test_logits``; forecasting: ``fit_out`` (validation forecasts),
    ``test_out``; optional ``fit_embed`` / ``test_embed`` (UniTS pooled states).
    Reliability heads are fitted on train-OOF (classification) or validation
    (forecasting) rows only and evaluated once on the test split.
    """
    if task not in ("classification", "forecasting"):
        raise ValueError(task)
    timings = {}
    t0 = time.perf_counter()
    ext = fit_extractor(data, variant, scfg, seed)
    timings["symbolic_fit_s"] = time.perf_counter() - t0
    fit_split = data.train if task == "classification" else data.val
    test = data.test
    chunk = scfg.get("chunk", 256)
    t0 = time.perf_counter()
    sym_fit, _ = transform(fit_split, ext, chunk)
    sym_test, meta_test = transform(test, ext, chunk, keep_meta=True)
    timings["symbolic_transform_s"] = time.perf_counter() - t0
    F_fit, F_test = sym_fit.reshape(len(sym_fit), -1), sym_test.reshape(len(sym_test), -1)
    X_fit, X_test = fit_split.x_raw(_all(fit_split)), test.x_raw(_all(test))
    S_fit, S_test = summary_statistics(X_fit), summary_statistics(X_test)

    if task == "classification":
        kind = "classification"
        y_fit_true = np.asarray(fit_split.target(_all(fit_split))).reshape(-1)
        y_test_true = np.asarray(test.target(_all(test))).reshape(-1)
        p_fit, p_test = softmax(fm["fit_logits"]), softmax(fm["test_logits"])
        y_fit = (p_fit.argmax(1) != y_fit_true).astype(int)
        loss_test = (p_test.argmax(1) != y_test_true).astype(np.float64)

        def conf(p):
            ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
            return np.stack([-p.max(1), ent], 1)

        C_fit, C_test = conf(p_fit), conf(p_test)
        conf_names = [{"channel": None, "kind": "fm_negative_max_probability", "word": None},
                      {"channel": None, "kind": "fm_predictive_entropy", "word": None}]
        scores = {"fm_confidence": -p_test.max(1)}
        task_info = {"fit_rows": "out_of_fold_train", "fit_error_rate": float(y_fit.mean()),
                     "test_error_rate": float(loss_test.mean())}
    else:
        kind = "regression"
        tgt_fit, tgt_test = fit_split.target(_all(fit_split)), test.target(_all(test))
        loss_fit = ((np.asarray(fm["fit_out"]) - tgt_fit) ** 2).mean((1, 2))
        loss_test = ((np.asarray(fm["test_out"]) - tgt_test) ** 2).mean((1, 2))
        y_fit = np.log(loss_fit + 1e-8)
        C_fit, C_test = context_volatility(X_fit)[:, None], context_volatility(X_test)[:, None]
        conf_names = [{"channel": None, "kind": "context_volatility", "word": None}]
        scores = {"context_volatility": C_test[:, 0]}
        task_info = {"fit_rows": "validation_windows", "fit_mean_mse": float(loss_fit.mean()),
                     "test_mean_mse": float(loss_test.mean())}

    time_ordered = task == "forecasting"
    n_folds = int(rcfg.get("n_folds", 5))
    heads = {}

    def head(name, A_fit, A_test):
        t = time.perf_counter()
        h = SparseReliability(kind, n_folds=n_folds, seed=seed, time_ordered=time_ordered).fit(A_fit, y_fit)
        heads[name] = h
        scores[name] = h.score(A_test)
        timings[f"head_{name}_s"] = time.perf_counter() - t

    head("statistics", S_fit, S_test)
    head("symbolic", F_fit, F_test)
    head("fm_signal+statistics", np.hstack([C_fit, S_fit]), np.hstack([C_test, S_test]))
    head("fm_signal+symbolic", np.hstack([C_fit, F_fit]), np.hstack([C_test, F_test]))
    E_fit, E_test = fm.get("fit_embed"), fm.get("test_embed")
    if E_fit is not None and E_test is not None:
        head("embedding", E_fit, E_test)
        head("embedding+symbolic", np.hstack([E_fit, F_fit]), np.hstack([E_test, F_test]))
    # Fit-free combinations: mean percentile rank of two risk signals.  Nothing is
    # learned, so the shift between out-of-fold (fit) and final-model (test)
    # confidence cannot distort them, unlike the fitted "+"-heads above.
    base = "fm_confidence" if task == "classification" else "context_volatility"
    pairs = [(base, "symbolic"), (base, "statistics")]
    if "embedding" in scores:
        pairs += [("embedding", "symbolic"), ("embedding", "statistics")]
    for a, b in pairs:
        name = f"rank:{'fm_signal' if a == base else a}+{b}"
        scores[name] = 0.5 * (_pct_rank(scores[a]) + _pct_rank(scores[b]))
    scores["random"] = np.random.default_rng(seed + 17).random(len(loss_test))
    evaluation = {name: risk_coverage(s, loss_test, seed=seed) for name, s in scores.items()}

    t = time.perf_counter()
    null = permutation_null(F_fit, y_fit, F_test, loss_test, kind, rcfg.get("n_permutations", 20), seed,
                            time_ordered=time_ordered, n_folds=n_folds)
    timings["permutation_null_s"] = time.perf_counter() - t
    sym_names = symbolic_feature_names(ext)
    stat_names = summary_statistic_names(data.n_channels)
    top_k = int(rcfg.get("rule_cards", 10))
    rule_cards = {
        "symbolic": heads["symbolic"].top_terms(sym_names, top_k),
        "fm_signal+symbolic": heads["fm_signal+symbolic"].top_terms(conf_names + sym_names, top_k),
        "statistics": heads["statistics"].top_terms(stat_names, top_k),
    }
    examples = _reliability_examples(task, heads["symbolic"], scores["symbolic"], F_test, sym_names, meta_test,
                                     ext, X_test, loss_test, fm, test, rcfg.get("n_examples", 6))
    sym_eval = evaluation["symbolic"]
    return {
        "readout": "symbolic_reliability",
        "variant": variant,
        "task_info": task_info,
        "n_fit": int(len(y_fit)), "n_test": int(len(loss_test)),
        "risk_coverage": evaluation,
        "symbolic_permutation_control": {**null, "observed_aurc": sym_eval["aurc"],
                                         "p_value": permutation_p_value(sym_eval["aurc"], null)},
        "head_sparsity": {name: {"n_nonzero": h.n_nonzero, "n_features": int(h.coef_.size),
                                 "abstained_constant": h.constant_ is not None} for name, h in heads.items()},
        "rule_cards": rule_cards,
        "examples": examples,
        "symbolic_vocabulary": ext.summary(),
        "timings": timings,
    }


def _term_location(name: dict, meta: dict, row: int, ext) -> dict | None:
    c, k = name.get("channel"), name.get("word_index")
    if c is None or meta is None:
        return None
    if name["kind"] == "rarity":
        start = int(meta["rarity_location"][row, c])
    elif k is not None:
        if not bool(meta["informative"][row, c, k]):
            return None
        start = int(meta["location"][row, c, k])
    else:
        return None
    return {"channel": int(c), "start": start, "end": start + int(ext.window_)}


def _reliability_examples(task, head, risk, F_test, names, meta, ext, X_test, loss_test, fm, test, n_examples):
    """Highest-risk test cases with their exact additive symbolic evidence terms."""
    order = np.argsort(-np.asarray(risk), kind="stable")[: int(n_examples)]
    if not len(order):
        return []
    contrib = head.contributions(F_test[order])
    out = []
    for r, i in enumerate(order):
        terms = []
        for j in np.argsort(-contrib[r], kind="stable")[:3]:
            if contrib[r, j] <= 0:
                break
            terms.append({**names[j], "contribution": float(contrib[r, j]),
                          "location": _term_location(names[j], meta, int(i), ext)})
        item = {"test_index": int(i), "symbolic_risk": float(risk[i]), "observed_loss": float(loss_test[i]),
                "evidence_terms": terms, "input": _jsonable_array(X_test[i])}
        if task == "classification":
            p = softmax(fm["test_logits"][i:i + 1])[0]
            item.update({"true_class": int(np.asarray(test.target(np.array([i]))).reshape(-1)[0]),
                         "predicted_class": int(p.argmax()), "predicted_probability": float(p.max())})
        else:
            item.update({"target": _jsonable_array(test.target(np.array([i]))[0]),
                         "forecast": _jsonable_array(np.asarray(fm["test_out"])[i])})
        out.append(item)
    return out


# =========================================================================== anomaly head
def entity_series(split) -> dict:
    """Full test series per entity, [C, T], rebuilt from the split's windows."""
    ents = split.meta["entities"]
    if hasattr(split, "series"):
        return {str(e): np.asarray(split.series[str(e)], dtype=np.float32) for e in ents}
    X = split.x_raw(_all(split))
    out = {str(e): np.zeros((X.shape[1], v["length"]), dtype=np.float32) for e, v in ents.items()}
    for x, e, s in zip(X, split.meta["entity"], split.meta["start"]):
        e, s = str(e), int(s)
        if s < 0:  # left-padded short series
            out[e][:] = x[:, -s:-s + out[e].shape[1]]
        else:
            out[e][:, s:s + x.shape[-1]] = x
    return out


def assemble(window_scores: np.ndarray, split) -> dict:
    """Per-window per-point scores [n, L] -> full per-entity arrays (eval.metrics convention)."""
    ents = split.meta["entities"]
    out = {str(e): np.zeros(v["length"], dtype=np.float64) for e, v in ents.items()}
    L = window_scores.shape[1]
    for w, e, s in zip(window_scores, split.meta["entity"], split.meta["start"]):
        e, s = str(e), int(s)
        if s < 0:
            out[e][:] = w[-s:-s + len(out[e])]
        else:
            out[e][s:s + L] = w
    return out


def cheap_anomaly_metrics(scores: dict, labels: dict) -> dict:
    """Best-F1 (oracle) and PR-AUC, entity mean: used for the lambda sensitivity table only."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    f1s, aps = [], []
    for e, s in scores.items():
        y = np.asarray(labels[e]).astype(int)
        if y.min() == y.max():
            continue
        p, r, _ = precision_recall_curve(y, s)
        f1s.append(float(np.nanmax(2 * p * r / np.maximum(p + r, 1e-12))))
        aps.append(float(average_precision_score(y, s)))
    return {"f1": float(np.mean(f1s)) if f1s else float("nan"),
            "pr_auc": float(np.mean(aps)) if aps else float("nan")}


def top1_hit_rate(scores: dict, labels: dict, tolerance: int) -> float:
    """UCR-archive style: is each entity's single highest-scoring point within
    ``tolerance`` points of a labelled anomaly?  Entity mean."""
    hits = []
    for e, s in scores.items():
        y = np.asarray(labels[e]) > 0
        if not y.any():
            continue
        t = int(np.argmax(s))
        hits.append(bool(y[max(0, t - tolerance): t + tolerance + 1].any()))
    return float(np.mean(hits)) if hits else float("nan")


def _robust(values: np.ndarray):
    v = np.asarray(values, dtype=np.float64).ravel()
    loc = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(v.std()) if v.std() > 1e-12 else 1.0
    return loc, scale


def choose_window(ncfg: dict, n_channels: int, normal_series: list | None):
    """Label-free subsequence length.

    ``window: auto`` -> univariate data: the dominant training period (one cycle,
    the usual discord choice), clipped to ``period_bounds``; multivariate data or
    no clear period: ``window_multivariate`` (24, the setting of earlier phases).
    """
    w = ncfg.get("window", 24)
    if w != "auto":
        return int(w), {"rule": "fixed", "window": int(w)}
    fallback = int(ncfg.get("window_multivariate", 24))
    if n_channels > 1 or not normal_series:
        return fallback, {"rule": "multivariate_fixed", "window": fallback}
    period = estimate_period(normal_series, max_lag=int(ncfg.get("period_max_lag", 512)),
                             min_acf=float(ncfg.get("period_min_acf", 0.3)))
    if period is None:
        return fallback, {"rule": "no_clear_period_fallback", "window": fallback}
    lo, hi = ncfg.get("period_bounds", [16, 256])
    w = int(np.clip(period, lo, hi))
    return w, {"rule": "training_period", "estimated_period": int(period), "window": w}


def anomaly_readout(*, data, variant: str, ncfg: dict, seed: int, fm: dict, metric_fn,
                    fm_metrics: dict | None = None, normal_series: dict | None = None) -> dict:
    """Fuse UniTS reconstruction scores with normal-dictionary symbolic novelty.

    ``fm`` keys: ``val_window_scores`` [n_val, L] and ``test_window_scores``
    [n_test, L] (pointwise reconstruction errors).  ``normal_series`` (optional):
    the contiguous normal training series per entity, [C, T].  When given, the
    vocabulary is fitted on the first (1 - holdout) of each series and the
    symbolic normaliser on the held-out tail; otherwise on the training /
    validation windows.  No test label is used for fitting, normalising, the
    window rule or the fixed weight ``lambda``.
    """
    timings = {}
    rep = REPRESENTATION[variant]
    t0 = time.perf_counter()
    series_list = list(normal_series.values()) if normal_series else None
    window, window_info = choose_window(ncfg, data.n_channels, series_list)
    scorer = SymbolicNoveltyScorer(
        rep, window=window, word_len=ncfg.get("word_len", 8),
        alphabet_size=ncfg.get("alphabet_size", 4), step=ncfg.get("step", 1),
        distance_weight=ncfg.get("distance_weight", 4.0), flat_std=ncfg.get("flat_std", 1e-3),
        max_fit_windows=ncfg.get("max_fit_windows", 200_000), scale_floor=ncfg.get("scale_floor", 0.5),
        channel_top_k=ncfg.get("channel_top_k"), max_nearest_vocab=ncfg.get("max_nearest_vocab", 2048), seed=seed)
    holdout = float(ncfg.get("normalizer_holdout", 0.1))
    usable = [S for S in (series_list or []) if S.shape[1] >= 4 * window]
    if usable:
        cuts = [int(round((1.0 - holdout) * S.shape[1])) for S in usable]
        scorer.fit([S[:, :c] for S, c in zip(usable, cuts)])
        timings["novelty_fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scorer.fit_normalizer_heldout(list(zip(usable, cuts)))
        normalizer_source = f"held-out last {holdout:.0%} of each normal training series"
    else:
        scorer.fit(data.train.x_raw(_all(data.train)))
        timings["novelty_fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scorer.fit_normalizer(data.val.x_raw(_all(data.val)))
        normalizer_source = "validation windows"
    timings["novelty_normalizer_s"] = time.perf_counter() - t0

    fm_loc, fm_scale = _robust(fm["val_window_scores"])
    fm_full = assemble(np.asarray(fm["test_window_scores"]), data.test)
    labels = {str(e): np.asarray(v["labels"]) for e, v in data.test.meta["entities"].items()}
    series = entity_series(data.test)
    lam = float(ncfg.get("lambda", 1.0))
    z_fm, z_sym, z_ch = {}, {}, {}
    t0 = time.perf_counter()
    for e, S in series.items():
        z_fm[e] = (fm_full[e] - fm_loc) / fm_scale
        z_ch[e], z_sym[e] = scorer.standardized(S)
    timings["novelty_test_s"] = time.perf_counter() - t0
    fused = {e: z_fm[e] + lam * z_sym[e] for e in series}

    t0 = time.perf_counter()
    # Calibrated fusion: each stream -> -log tail probability under its own held-out
    # normal scores (UniTS: validation windows; symbolic: held-out normal tail), then
    # summed (Fisher-style).  Scale-free, no weight.  Added after the equal-weight z
    # fusion proved dominated by the heavy-tailed reconstruction error; both are reported.
    fm_ref = np.asarray(fm["val_window_scores"], dtype=np.float64).ravel()
    if len(fm_ref) > 1_000_000:
        fm_ref = fm_ref[np.random.default_rng(seed).choice(len(fm_ref), 1_000_000, replace=False)]
    q_fm = {e: tail_surprisal(fm_full[e], fm_ref) for e in series}
    q_sym = {e: tail_surprisal(z_sym[e], scorer.reference_) for e in series}
    fused_cal = {e: q_fm[e] + q_sym[e] for e in series}
    streams = {"fm_reconstruction": fm_full, "symbolic_novelty": z_sym, "fused": fused,
               "fused_calibrated": fused_cal}
    # The UniTS-only stream is identical for both miners: callers may pass its metrics once.
    metrics = {name: (dict(fm_metrics) if name == "fm_reconstruction" and fm_metrics else metric_fn(s, labels))
               for name, s in streams.items()}
    timings["metrics_s"] = time.perf_counter() - t0
    tol = int(ncfg.get("top1_tolerance", max(100, scorer.window_)))
    for name, s in streams.items():
        metrics[name]["top1_hit_rate"] = top1_hit_rate(s, labels, tol)
    sensitivity = {}
    for lv in ncfg.get("lambda_sensitivity", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]):
        sensitivity[str(lv)] = cheap_anomaly_metrics({e: z_fm[e] + lv * z_sym[e] for e in series}, labels)

    explanations, examples = _anomaly_explanations(scorer, series, z_fm, z_sym, z_ch, fused, labels, ncfg,
                                                   q_fm=q_fm, q_sym=q_sym, fused_cal=fused_cal)
    return {
        "readout": "symbolic_novelty_fusion",
        "variant": variant,
        "lambda": lam,
        "calibrated_fusion": "q_UniTS + q_symbolic, q = -log tail probability under held-out normal scores",
        "window_selection": window_info,
        "symbolic_normalizer_source": normalizer_source,
        "fm_normalizer": {"median": fm_loc, "iqr_or_std": fm_scale, "source": "validation_normal_windows"},
        "metrics": metrics,
        "lambda_sensitivity_diagnostic": sensitivity,
        "top1_tolerance": tol,
        "explanations": explanations,
        "examples": examples,
        "symbolic_dictionary": scorer.summary(),
        "timings": timings,
    }


def _anomaly_explanations(scorer, series, z_fm, z_sym, z_ch, fused, labels, ncfg, *, q_fm, q_sym, fused_cal):
    """Top events of the calibrated fused score, each with its symbolic evidence."""
    w = scorer.window_
    per_entity = int(ncfg.get("events_per_entity", 3))
    records = []
    for e, S in series.items():
        for t in top_events(fused_cal[e], per_entity, min_separation=4 * w):
            ch = int(np.argmax(z_ch[e][:, t]))
            ev = scorer.explain_point(S, t, ch)
            span = labels[e][ev["start"]:ev["end"]]
            records.append({"entity": e, "t": int(t), "fused_calibrated_score": float(fused_cal[e][t]),
                            "fused_score": float(fused[e][t]),
                            "fm_q": float(q_fm[e][t]), "symbolic_q": float(q_sym[e][t]),
                            "fm_z": float(z_fm[e][t]), "symbolic_z": float(z_sym[e][t]),
                            "channel_z": float(z_ch[e][ch, t]),
                            # Evaluation-only fields: test labels never enter scoring.
                            "label_at_t": int(labels[e][t] > 0),
                            "labelled_points_in_span": int((span > 0).sum()),
                            **ev})
    records.sort(key=lambda r: -r["fused_calibrated_score"])
    examples = []
    for r in records[: int(ncfg.get("n_examples", 6))]:
        e, t = r["entity"], r["t"]
        T = series[e].shape[1]
        a, b = max(0, t - 4 * w), min(T, t + 4 * w)
        ch = r["channel"]
        others = [int(c) for c in np.argsort(-z_ch[e][:, t])[:3] if int(c) != ch][:2]
        chans = [ch] + others
        examples.append({**r, "segment_start": int(a), "segment_end": int(b), "channels": chans,
                         "series": _jsonable_array(series[e][chans, a:b]),
                         "fm_z_segment": _jsonable_array(z_fm[e][a:b]),
                         "symbolic_z_segment": _jsonable_array(z_sym[e][a:b]),
                         "fused_segment": _jsonable_array(fused[e][a:b]),
                         "fm_q_segment": _jsonable_array(q_fm[e][a:b]),
                         "symbolic_q_segment": _jsonable_array(q_sym[e][a:b]),
                         "fused_calibrated_segment": _jsonable_array(fused_cal[e][a:b]),
                         "labels_segment": np.asarray(labels[e][a:b]).astype(int).tolist()})
    return records, examples
