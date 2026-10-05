"""Multivariate anomaly-detection loaders (MSL, SMAP, SMD, PSM).

The univariate loaders in ``loaders.py`` use the TSB-UAD per-channel form of MSL/SMAP.
This module loads the *multivariate* form that the MOMENT and UniTS papers use, where
one sample is ``[n_channels, window]``.

Sources, and what needs credentials:

  MSL / SMAP  NASA telemanom .npy files, one per channel, shape (timesteps, n_inputs)
              where column 0 is the telemetry value and the remaining columns are
              one-hot encoded command flags (25 inputs for SMAP, 55 for MSL).
              The original S3 bucket now returns 403, so the data comes from Kaggle
              (``patrickfleith/nasa-anomaly-detection-dataset-smap-msl``) and needs a
              kaggle.json API key. The *labels* (labeled_anomalies.csv) are on GitHub
              and need no credentials.
  SMD         OmniAnomaly ServerMachineDataset, 38 channels per machine. Direct HTTP.
  PSM         eBay RANSynCoders Pooled Server Metrics, 25 channels. Direct HTTP.

``load_anomaly_mv`` returns the same ``TaskData`` structure as the univariate loader,
so the training loop, metrics and explanation code are unchanged.
"""
from __future__ import annotations

import ast
import io
import logging
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from symtsfm.data.loaders import ArraySplit, TaskData, _download, _windows

log = logging.getLogger(__name__)

TELEMANOM_LABELS = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
KAGGLE_DATASET = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
SMD_RAW = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
PSM_RAW = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"
SMD_DEFAULT_MACHINES = ["machine-1-1", "machine-2-1", "machine-3-2"]


# =========================================================================== NASA (MSL / SMAP)
def fetch_nasa(data_dir) -> Path | None:
    """Download the telemanom .npy archive from Kaggle if an API key is configured.

    Returns the directory holding ``train/`` and ``test/``, or None if unavailable
    (no kaggle.json, no network, or the Kaggle CLI is missing). Never raises: the
    caller decides whether to fall back to another dataset.
    """
    root = Path(data_dir) / "anomaly_mv" / "nasa"
    for cand in (root, root / "data", root / "data" / "data"):
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    root.mkdir(parents=True, exist_ok=True)
    key = Path.home() / ".kaggle" / "kaggle.json"
    if not key.exists():
        log.warning("No %s: cannot download MSL/SMAP telemetry from Kaggle", key)
        return None
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(root)],
                       check=True, capture_output=True, timeout=1800)
        zips = list(root.glob("*.zip"))
        if not zips:
            return None
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(root)
        zips[0].unlink()
    except Exception as e:  # noqa: BLE001
        log.warning("Kaggle download of %s failed: %s", KAGGLE_DATASET, e)
        return None
    for cand in (root, root / "data", root / "data" / "data"):
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    return None


def nasa_labels(data_dir, spacecraft: str) -> dict[str, list]:
    """chan_id -> [[start, end], ...] for one spacecraft ('MSL' or 'SMAP')."""
    import pandas as pd

    path = _download(TELEMANOM_LABELS, Path(data_dir) / "anomaly_mv" / "labeled_anomalies.csv")
    df = pd.read_csv(path)
    df = df[df["spacecraft"].str.upper() == spacecraft.upper()]
    return {r.chan_id: ast.literal_eval(r.anomaly_sequences) for r in df.itertuples()}


def _nasa_entities(name, data_dir, max_entities=None, nasa_root=None):
    root = Path(nasa_root) if nasa_root else fetch_nasa(data_dir)
    if root is None:
        raise FileNotFoundError(
            f"{name} telemetry not available. Upload a kaggle.json API key (Kaggle -> Account -> "
            f"Create New API Token) so '{KAGGLE_DATASET}' can be downloaded, or pass nasa_root=... "
            "pointing at a folder that contains train/ and test/ .npy files."
        )
    labels = nasa_labels(data_dir, name)
    ents = []
    for chan in sorted(labels):
        tr_f, te_f = root / "train" / f"{chan}.npy", root / "test" / f"{chan}.npy"
        if not (tr_f.exists() and te_f.exists()):
            continue
        tr = np.load(tr_f).astype(np.float32)  # [T, n_inputs]: col 0 telemetry, rest one-hot commands
        te = np.load(te_f).astype(np.float32)
        lab = np.zeros(len(te), dtype=np.int64)
        for s, e in labels[chan]:
            lab[int(s) : int(e) + 1] = 1
        ents.append((chan, tr.T, np.zeros(len(tr), np.int64), te.T, lab))  # channels-first
    if not ents:
        raise FileNotFoundError(f"no {name} channels found under {root}")
    return ents[:max_entities] if max_entities else ents


# =========================================================================== SMD / PSM
def _smd_entities(data_dir, machines=None, max_entities=None):
    root = Path(data_dir) / "anomaly_mv" / "SMD"
    machines = machines or SMD_DEFAULT_MACHINES
    ents = []
    for m in machines:
        tr = np.loadtxt(_download(f"{SMD_RAW}/train/{m}.txt", root / "train" / f"{m}.txt"), delimiter=",", dtype=np.float32)
        te = np.loadtxt(_download(f"{SMD_RAW}/test/{m}.txt", root / "test" / f"{m}.txt"), delimiter=",", dtype=np.float32)
        lab = np.loadtxt(_download(f"{SMD_RAW}/test_label/{m}.txt", root / "test_label" / f"{m}.txt"), dtype=np.int64)
        ents.append((m, tr.T, np.zeros(len(tr), np.int64), te.T, lab))
    return ents[:max_entities] if max_entities else ents


def _psm_entities(data_dir, max_entities=None):
    import pandas as pd

    root = Path(data_dir) / "anomaly_mv" / "PSM"
    tr = pd.read_csv(_download(f"{PSM_RAW}/train.csv", root / "train.csv")).drop(columns=["timestamp_(min)"], errors="ignore")
    te = pd.read_csv(_download(f"{PSM_RAW}/test.csv", root / "test.csv")).drop(columns=["timestamp_(min)"], errors="ignore")
    lab = pd.read_csv(_download(f"{PSM_RAW}/test_label.csv", root / "test_label.csv"))
    lab = lab["label"].to_numpy(np.int64)
    tr = np.nan_to_num(tr.to_numpy(np.float32)).T  # PSM has a few NaNs
    te = np.nan_to_num(te.to_numpy(np.float32)).T
    return [("psm", tr, np.zeros(tr.shape[1], np.int64), te, lab)][:max_entities or 1]


# =========================================================================== assembly
def load_anomaly_mv(name, data_dir, seq_len=512, train_stride=None, val_fraction=0.1, seed=0,
                    max_entities=None, smd_machines=None, nasa_root=None, scale=True, **_) -> TaskData:
    """One sample = [n_channels, seq_len]. Train split is normal-only; test carries point labels."""
    if name.upper() in ("MSL", "SMAP"):
        ents = _nasa_entities(name.upper(), data_dir, max_entities, nasa_root)
    elif name.upper() == "SMD":
        ents = _smd_entities(data_dir, smd_machines, max_entities)
    elif name.upper() == "PSM":
        ents = _psm_entities(data_dir, max_entities)
    else:
        raise ValueError(f"unknown multivariate anomaly dataset {name!r}")

    L = seq_len
    stride = train_stride or L
    tr_X, tr_pad, te_X, te_lab, te_pad, te_ent, te_start, tests = [], [], [], [], [], [], [], {}
    n_channels = ents[0][1].shape[0]
    for eid, xtr, ltr, xte, lte in ents:
        if xtr.shape[0] != n_channels:
            raise ValueError(f"entity {eid} has {xtr.shape[0]} channels, expected {n_channels}")
        if scale:  # per-entity standardisation fitted on the TRAIN part only
            mu = xtr.mean(1, keepdims=True)
            sd = xtr.std(1, keepdims=True) + 1e-8
            xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        w, _, _, pad = _windows_mv(xtr, ltr, L, stride)
        tr_X.append(w), tr_pad.append(pad)
        w, lab, st, pad = _windows_mv(xte, lte, L, L)
        te_X.append(w), te_lab.append(lab), te_start.append(st), te_pad.append(pad)
        te_ent += [eid] * len(w)
        tests[eid] = {"length": xte.shape[1], "labels": lte}

    Xtr = np.concatenate(tr_X)
    Ptr = np.concatenate(tr_pad)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    n_va = max(1, int(round(len(Xtr) * val_fraction)))
    va, tr = np.sort(perm[:n_va]), np.sort(perm[n_va:])
    mk = lambda i: ArraySplit(Xtr[i], None, L, pad=Ptr[i], meta={"window_labels": np.zeros(len(i), int)})  # noqa: E731
    test = ArraySplit(np.concatenate(te_X), None, L, pad=np.concatenate(te_pad),
                      meta={"entity": np.array(te_ent), "start": np.concatenate(te_start),
                            "point_labels": np.concatenate(te_lab), "entities": tests})
    info = {"n_entities": len(ents), "n_channels": n_channels,
            "channel_names": [f"channel_{i}" for i in range(n_channels)], "supervised_words": False,
            "entity_ids": [e[0] for e in ents],
            "anomaly_ratio": float((np.concatenate(te_lab) > 0).mean())}
    return TaskData("anomaly", name, n_channels, 0, mk(tr), mk(va), test, info)


def _windows_mv(x, labels, L, stride):
    """x [C, T] -> windows [n, C, L] (left edge-padded if the series is shorter than L)."""
    C, T = x.shape
    if T < L:
        p = L - T
        w = np.concatenate([np.repeat(x[:, :1], p, axis=1), x], axis=1)[None]
        lab = np.concatenate([np.full(p, -1), labels])[None]
        return w, lab, np.array([-p]), np.array([p])
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] != T - L:
        starts.append(T - L)
    starts = np.array(starts)
    idx = starts[:, None] + np.arange(L)[None]
    return x[:, idx].transpose(1, 0, 2), labels[idx], starts, np.zeros(len(starts), dtype=int)
