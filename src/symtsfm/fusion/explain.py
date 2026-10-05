"""Per-prediction explanation artefacts for the dual-path models.

Every classification / anomaly prediction of a guided model is paired with a
JSON record holding
  * the numeric prediction (and ground truth when known),
  * the fusion gate weight for that sample,
  * the top-N contributing SAX/SFA words, each with its raw subsequence location
    (start/end index in the model input and, when an ``offset`` is given, in
    the original series), the matched raw values, match distance, IG score and
    the training class the word is associated with.

"Contribution" is gradient x input of the task output w.r.t. the standardised
symbolic features (baseline = training mean = 0), summed in absolute value over
a word's presence/frequency/distance features. It is model-agnostic: the
caller passes a function mapping symbolic features -> one scalar per sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from symtsfm.symbolic.sax import min_znorm_distance


def symbolic_attributions(forward_fn, sym: torch.Tensor) -> torch.Tensor:
    """Gradient x input of ``forward_fn(sym) -> [B]`` with respect to ``sym``."""
    sym = sym.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        target = forward_fn(sym)
        (grad,) = torch.autograd.grad(target.sum(), sym)
    return (grad * sym).detach()


def _round(v, nd=4):
    return [round(float(a), nd) for a in np.asarray(v).ravel()]


def build_explanations(
    *,
    extractor,
    attributions: np.ndarray,
    meta: dict,
    x_raw: np.ndarray,
    gate: np.ndarray,
    predictions: list,
    targets: list | None = None,
    sample_ids: list | None = None,
    offsets: np.ndarray | None = None,
    top_n: int = 3,
    extra: list[dict] | None = None,
    task: str = "",
    variant: str = "",
) -> list[dict]:
    """Build one JSON-serialisable record per sample.

    attributions [B, C, 3K+1] (or [B, C*(3K+1)]); meta: the extractor's match
    metadata for the same samples (``location``/``distance``/``presence`` [B, C, K],
    ``rarity_location`` [B, C]); x_raw [B, C, T] the raw input the symbolic path saw.
    """
    B = len(x_raw)
    C, K, F = extractor.n_channels, extractor.K, extractor.dim_per_channel
    full_attr = np.asarray(attributions).reshape(B, C, -1)
    if full_attr.shape[-1] < F:
        raise ValueError(f"expected at least {F} symbolic attribution features, got {full_attr.shape[-1]}")
    attr = full_attr[..., :F]
    word_attr = np.abs(attr[..., :-1].reshape(B, C, K, 3)).sum(-1)  # [B, C, K]
    signed = attr[..., :-1].reshape(B, C, K, 3).sum(-1)
    # Location-aware forecast adapters append K normalised match locations
    # after the original [3*K+1] extractor layout.  Attribute each location
    # to its own word rather than discarding that temporal evidence.
    if full_attr.shape[-1] >= F + K:
        loc_attr = full_attr[..., F : F + K]
        word_attr += np.abs(loc_attr)
        signed += loc_attr
    w = extractor.window_
    out = []
    real = np.array([[w["word"] != "<none>" for w in extractor.words(c)] for c in range(C)])
    for b in range(B):
        flat = np.where(real.ravel(), word_attr[b].ravel(), -np.inf)  # skip padding slots
        order = [i for i in np.argsort(-flat, kind="mergesort")[:top_n] if np.isfinite(flat[i])]
        words = []
        for rank, idx in enumerate(order, 1):
            c, k = divmod(int(idx), K)
            info = extractor.selectors_[c].describe()[k]
            start = int(meta["location"][b, c, k])
            rec = {
                "rank": rank,
                "channel": c,
                "word": info["word"],
                "method": extractor.method,
                "attribution": round(float(signed[b, c, k]), 6),
                "attribution_abs": round(float(word_attr[b, c, k]), 6),
                "info_gain": round(info["info_gain"], 4),
                "associated_class": info["class"],
                "train_support": round(info["support"], 4),
                "present_exactly": bool(meta["presence"][b, c, k] > 0),
                "match_distance": round(float(meta["distance"][b, c, k]), 4),
                "location": {"start": start, "end": start + w},
                # False when every window of the input is equidistant from the word
                # (e.g. an all-flat input): the location is then arbitrary, not evidence.
                "location_informative": bool(meta["informative"][b, c, k]),
                "subsequence": _round(x_raw[b, c, start : start + w]),
            }
            if offsets is not None:
                rec["location"]["absolute_start"] = int(offsets[b]) + start
                rec["location"]["absolute_end"] = int(offsets[b]) + start + w
            words.append(rec)
        rc = int(np.argmax(np.abs(attr[b, :, -1])))
        rl = int(meta["rarity_location"][b, rc])
        record = {
            "task": task,
            "variant": variant,
            "sample_id": sample_ids[b] if sample_ids is not None else b,
            "prediction": predictions[b],
            "target": None if targets is None else targets[b],
            "gate_weight": round(float(np.asarray(gate[b]).mean()), 6),
            "top_words": words,
            "rarest_window": {
                "channel": rc,
                "attribution": round(float(attr[b, rc, -1]), 6),
                "location": {"start": rl, "end": rl + w},
            },
        }
        if extra is not None:
            record.update(extra[b])
        out.append(record)
    return out


def write_jsonl(records: list[dict], path: str | Path, append: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# =========================================================================== validation
def verify_record(record: dict, extractor, x_raw: np.ndarray, atol: float = 1e-3) -> list[str]:
    """Re-derive each reported word match from the raw input; return a list of problems.

    Checks: location in bounds, subsequence equals the raw input at that location,
    and the reported distance is reproduced by an
    independent recomputation (real z-norm distance for FastShapelets, symbolic
    distance for BOSS-ST).
    """
    problems = []
    w = extractor.window_
    for wd in record["top_words"]:
        c, s, e = wd["channel"], wd["location"]["start"], wd["location"]["end"]
        series = x_raw[c]
        if not (0 <= s < e <= series.shape[-1]) or e - s != w:
            problems.append(f"out of bounds: {wd['word']} [{s},{e}) for length {series.shape[-1]}")
            continue
        sub = series[s:e]
        if not np.allclose(sub, wd["subsequence"], atol=1e-3):
            problems.append(f"subsequence mismatch for {wd['word']} at {s}")
        sel = extractor.selectors_[c]
        k = sel.words_.index(wd["word"])
        if extractor.method == "fastshapelets":
            d, _ = min_znorm_distance(sub[None], sel.shapelets_[k : k + 1])
            d = float(d[0, 0])
        else:
            syms = sel.sfa.symbols(sub[None])[0]
            d = float(np.abs(syms - sel._syms_sel[k]).mean() / max(1, sel.alphabet_size - 1))
        if abs(d - wd["match_distance"]) > atol:
            problems.append(f"distance {wd['match_distance']} not reproduced ({d:.4f}) for {wd['word']}")
    return problems


def placeholder_check(records: list[dict]) -> dict:
    """Aggregate sanity: are locations/words/gates varying across predictions (not placeholders)?"""
    starts = [w["location"]["start"] for r in records for w in r["top_words"]]
    words = [w["word"] for r in records for w in r["top_words"]]
    gates = [r["gate_weight"] for r in records]
    flat_subseq = sum(np.ptp(w["subsequence"]) < 1e-8 for r in records for w in r["top_words"])
    return {
        "n_records": len(records),
        "n_distinct_starts": len(set(starts)),
        "n_distinct_words": len(set(words)),
        "gate_range": [min(gates), max(gates)] if gates else None,
        "n_flat_subsequences": int(flat_subseq),
        "n_uninformative_locations": int(sum(not w.get("location_informative", True)
                                             for r in records for w in r["top_words"])),
        "locations_vary": len(set(starts)) > 1,
    }
