"""Dataset loaders for the 3 tasks x 3 datasets matrix.

  classification  UCR/UEA archive (.ts)      FordA, UWaveGestureLibrary, EthanolConcentration
  forecasting     Autoformer-format CSVs      ETTh1, Weather, Electricity
  anomaly         TSB-UAD / OmniAnomaly       MSL, SMAP, SMD

Every loader returns a ``TaskData`` with ``train`` / ``val`` / ``test`` splits
exposing the same interface:

  len(split)               number of samples
  split.x_raw(idx)         [n, C, T]  raw input as seen by Path B (symbolic)
  split.x_model(idx)       ([n, C, seq_len], [n, seq_len] mask)  stock-MOMENT input
  split.target(idx)        labels [n] | future [n, C, H] | reconstruction target
  split.fit_labels()       labels Path B may use for word scoring (train split only)

Files are downloaded once into ``data_dir`` (put it on persistent storage).
Downloads happen only here; nothing at train/eval time touches the network.
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

UCR_URL = "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
# The aeon/TSML archive mirrors these datasets on Zenodo.  The primary archive
# occasionally returns HTTP 503, so use this official mirror automatically for
# the classification datasets selected by the notebook.
TSC_ZENODO_RECORDS = {
    "NATOPS": 11206248,
    "ArticularyWordRecognition": 11204924,
    "SelfRegulationSCP1": 11206265,
    "ECG5000": 11186692,
    "FordA": 11191164,
}
PILE = "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main"
PILE_API = "https://huggingface.co/api/datasets/AutonLab/Timeseries-PILE/tree/main"
FORECAST_FILES = {"ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv",
                  "Weather": "weather.csv", "Electricity": "electricity.csv",
                  # Autoformer-format multivariate benchmark available from the
                  # MOMENT Time-series Pile.  It uses the standard custom split
                  # in ``load_forecasting`` below.
                  "ExchangeRate": "exchange_rate.csv"}
TSB_DIRS = {"MSL": "NASA-MSL", "SMAP": "NASA-SMAP"}
SMD_RAW = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
SMD_API = "https://api.github.com/repos/NetManAIOps/OmniAnomaly/contents/ServerMachineDataset/train"
# Official UCR 2021 anomaly archive.  The archive is large, but ``_ucr_anomaly_file``
# extracts only the requested member and caches it on Drive.
UCR_ANOMALY_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
UCR_ANOMALY_DATASETS = {"BIDMC1", "PowerDemand1"}


# =========================================================================== download
def _download(url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            log.info("downloading %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "symbolic-fm/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"download failed for {url}: {e}") from e
            time.sleep(2 * (attempt + 1))
    return dest


def _download_classification_archive(name: str, root: Path) -> Path:
    """Fetch a UCR/UEA archive, falling back to its aeon Zenodo mirror.

    The fallback keeps a transient failure of timeseriesclassification.com from
    stopping the complete dataset-inspection cell in the notebook.
    """
    dest = root / f"{name}.zip"
    try:
        return _download(UCR_URL.format(name=name), dest)
    except RuntimeError as primary_error:
        record = TSC_ZENODO_RECORDS.get(name)
        if record is None:
            raise RuntimeError(
                f"Could not download {name} from the UCR/UEA archive. "
                "The primary server is unavailable and no Zenodo fallback is "
                f"configured for this dataset. Original error: {primary_error}"
            ) from primary_error
        log.warning("primary classification archive failed for %s; using Zenodo mirror", name)
        # Zenodo stores the two source ``.ts`` files, rather than a ZIP named
        # ``{name}.zip``.  Cache both files and create the small local archive
        # expected by the rest of this loader.
        train = _download(
            f"https://zenodo.org/records/{record}/files/{name}_TRAIN.ts?download=1",
            root / f"{name}_TRAIN.ts", retries=4,
        )
        test = _download(
            f"https://zenodo.org/records/{record}/files/{name}_TEST.ts?download=1",
            root / f"{name}_TEST.ts", retries=4,
        )
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(train, arcname=train.name)
            archive.write(test, arcname=test.name)
        return dest


def _json(url: str, cache: Path):
    return json.loads(_download(url, cache).read_text(encoding="utf-8"))


# =========================================================================== model input
def prepare_model_input(x: np.ndarray, seq_len: int = 512, long_policy: str = "interpolate",
                        pad: np.ndarray | None = None):
    """Stock MOMENT input convention: left zero-pad short series with input_mask=0;
    series longer than ``seq_len`` are linearly resampled (or truncated to the last
    ``seq_len`` points). ``pad`` optionally marks already-padded leading points."""
    x = np.asarray(x, dtype=np.float32)
    n, C, T = x.shape
    mask = np.ones((n, seq_len), dtype=np.float32)
    if T > seq_len:
        if long_policy == "truncate":
            x = x[..., -seq_len:]
        elif long_policy == "interpolate":
            src = np.linspace(0, T - 1, seq_len)
            lo = np.floor(src).astype(int)
            hi = np.minimum(lo + 1, T - 1)
            w = (src - lo).astype(np.float32)
            x = x[..., lo] * (1 - w) + x[..., hi] * w
        else:
            raise ValueError(long_policy)
    elif T < seq_len:
        out = np.zeros((n, C, seq_len), dtype=np.float32)
        out[..., seq_len - T :] = x
        mask[:, : seq_len - T] = 0
        x = out
    if pad is not None:  # pre-padded samples (anomaly windows shorter than seq_len)
        for i, p in enumerate(pad):
            if p:
                x[i, :, :p] = 0
                mask[i, :p] = 0
    return np.nan_to_num(x), mask


# =========================================================================== splits
class ArraySplit:
    def __init__(self, X, y, seq_len=512, long_policy="interpolate", pad=None, meta=None, fit_y=None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = y
        self.seq_len, self.long_policy = seq_len, long_policy
        self.pad = pad
        self.meta = meta or {}
        self._fit_y = fit_y

    def __len__(self):
        return len(self.X)

    def x_raw(self, idx=None):
        return self.X if idx is None else self.X[idx]

    def x_model(self, idx):
        return prepare_model_input(self.X[idx], self.seq_len, self.long_policy,
                                   None if self.pad is None else self.pad[idx])

    def target(self, idx):
        return self.X[idx] if self.y is None else self.y[idx]

    def fit_labels(self, idx=None):
        y = self._fit_y if self._fit_y is not None else self.y
        return y if (y is None or idx is None) else y[idx]


class ForecastSplit:
    """Lazy sliding windows over a [T, C] array (Electricity would not fit in RAM otherwise)."""

    def __init__(self, data, seq_len, horizon, stride=1, max_samples=None, direction_edges=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.L, self.H = seq_len, horizon
        starts = np.arange(0, len(self.data) - seq_len - horizon + 1, stride)
        if max_samples and len(starts) > max_samples:  # evenly spaced subsample (CPU configs)
            starts = starts[np.linspace(0, len(starts) - 1, max_samples).round().astype(int)]
        self.starts = starts
        self.direction_edges = direction_edges
        self.meta = {"starts": starts}

    def __len__(self):
        return len(self.starts)

    def _win(self, idx, off, length):
        s = self.starts[idx if idx is not None else slice(None)]
        s = np.atleast_1d(s)
        ix = s[:, None] + off + np.arange(length)[None, :]
        return self.data[ix].transpose(0, 2, 1)  # [n, C, length]

    def x_raw(self, idx=None):
        return self._win(idx, 0, self.L)

    def x_model(self, idx):
        x = self.x_raw(idx)
        return x, np.ones((len(x), self.L), dtype=np.float32)

    def target(self, idx):
        return self._win(idx, self.L, self.H)

    def fit_labels(self, idx=None):
        """Future-direction pseudo-classes per channel (see symbolic.features), computed
        with cumulative sums so no window tensor is materialised. First call fixes the
        tertile edges (on the training split)."""
        s = self.starts if idx is None else self.starts[idx]
        d = self.data.astype(np.float64)
        c1 = np.concatenate([np.zeros((1, d.shape[1])), d.cumsum(0)])
        c2 = np.concatenate([np.zeros((1, d.shape[1])), (d**2).cumsum(0)])
        L, H, tail = self.L, self.H, 24
        mean_ctx = (c1[s + L] - c1[s]) / L
        sd = np.sqrt(np.maximum((c2[s + L] - c2[s]) / L - mean_ctx**2, 0)) + 1e-8
        tail_m = (c1[s + L] - c1[s + L - tail]) / tail
        fut = (c1[s + L + H] - c1[s + L]) / H
        delta = (fut - tail_m) / sd  # [n, C]
        if self.direction_edges is None:
            self.direction_edges = np.quantile(delta, [1 / 3, 2 / 3])
        return np.digitize(delta, self.direction_edges)


@dataclass
class TaskData:
    task: str
    name: str
    n_channels: int
    out_dim: int
    train: object
    val: object
    test: object
    info: dict = field(default_factory=dict)


def _stratified_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = int(round(len(idx) * frac))
        val.extend(rng.choice(idx, k, replace=False))
    val = np.sort(np.array(val, dtype=int))
    tr = np.setdiff1d(np.arange(len(y)), val)
    return tr, val


# =========================================================================== classification
def parse_ts(text: str):
    """Minimal parser for the sktime/aeon ``.ts`` format (equal-length series)."""
    X, y, in_data = [], [], False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            if line.lower().startswith("@data"):
                in_data = True
            continue
        *dims, label = line.split(":")
        X.append([[np.nan if v == "?" else float(v) for v in d.split(",")] for d in dims])
        y.append(label.strip())
    lengths = {len(d) for s in X for d in s}
    if len(lengths) != 1:
        raise ValueError(f"unequal-length series not supported: lengths {sorted(lengths)[:5]}")
    X = np.array(X, dtype=np.float32)
    if np.isnan(X).any():  # linear interpolation along time, edges filled
        for i, c in zip(*np.where(np.isnan(X).any(-1))):
            s = X[i, c]
            ok = ~np.isnan(s)
            X[i, c] = np.interp(np.arange(len(s)), np.flatnonzero(ok), s[ok]) if ok.any() else 0
    return X, np.array(y)


def load_classification(name, data_dir, seq_len=512, val_fraction=0.1, seed=0,
                        long_policy="interpolate", max_train=None, max_test=None, **_):
    root = Path(data_dir) / "ucr_uea"
    z = _download_classification_archive(name, root)
    with zipfile.ZipFile(z) as zf:
        names = {Path(n).name.lower(): n for n in zf.namelist()}
        read = lambda s: zf.read(names[f"{name}_{s}.ts".lower()]).decode("utf-8", errors="replace")  # noqa: E731
        Xtr, ytr = parse_ts(read("TRAIN"))
        Xte, yte = parse_ts(read("TEST"))
    classes = np.unique(np.concatenate([ytr, yte]))
    ytr, yte = np.searchsorted(classes, ytr), np.searchsorted(classes, yte)
    rng = np.random.default_rng(seed)
    if max_train and len(Xtr) > max_train:
        keep, _ = _stratified_split(ytr, 1 - max_train / len(Xtr), seed)
        Xtr, ytr = Xtr[keep], ytr[keep]
    if max_test and len(Xte) > max_test:
        keep = np.sort(rng.choice(len(Xte), max_test, replace=False))
        Xte, yte = Xte[keep], yte[keep]
    tr, va = _stratified_split(ytr, val_fraction, seed)
    mk = lambda X, y: ArraySplit(X, y, seq_len, long_policy)  # noqa: E731
    return TaskData("classification", name, Xtr.shape[1], len(classes),
                    mk(Xtr[tr], ytr[tr]), mk(Xtr[va], ytr[va]), mk(Xte, yte),
                    {"classes": classes.tolist(), "length": int(Xtr.shape[-1]),
                     "channel_names": [f"channel_{i}" for i in range(Xtr.shape[1])]})


# =========================================================================== forecasting
def load_forecasting(name, data_dir, seq_len=512, horizon=96, train_stride=1, eval_stride=1,
                     max_train=None, max_eval=None, target_column=None, **_):
    import pandas as pd

    path = _download(f"{PILE}/forecasting/autoformer/{FORECAST_FILES[name]}",
                     Path(data_dir) / "forecasting" / FORECAST_FILES[name])
    df = pd.read_csv(path)
    value_df = df.drop(columns=[c for c in df.columns if c.lower() == "date"])
    if target_column is not None:
        if target_column not in value_df.columns:
            raise ValueError(f"{name}: target column {target_column!r} not found; "
                             f"available columns: {value_df.columns.tolist()}")
        value_df = value_df[[target_column]]
    values = value_df.to_numpy(np.float32)
    n = len(values)
    if name.startswith("ETTh"):
        b1 = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif name.startswith("ETTm"):
        b1 = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        b2 = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:  # Autoformer "custom" split 70/10/20
        n_tr, n_te = int(n * 0.7), int(n * 0.2)
        n_va = n - n_tr - n_te
        b1 = [0, n_tr - seq_len, n - n_te - seq_len]
        b2 = [n_tr, n_tr + n_va, n]
    mu = values[b1[0] : b2[0]].mean(0)
    sd = values[b1[0] : b2[0]].std(0) + 1e-8
    scaled = (values - mu) / sd
    train = ForecastSplit(scaled[b1[0] : b2[0]], seq_len, horizon, train_stride, max_train)
    train.fit_labels()  # fixes direction-bin edges on the training split
    val = ForecastSplit(scaled[b1[1] : b2[1]], seq_len, horizon, eval_stride, max_eval, train.direction_edges)
    test = ForecastSplit(scaled[b1[2] : b2[2]], seq_len, horizon, eval_stride, max_eval, train.direction_edges)
    return TaskData("forecasting", name, values.shape[1], horizon, train, val, test,
                    {"scaler_mean": mu.tolist(), "scaler_std": sd.tolist(), "borders": [b1, b2],
                     "channel_names": value_df.columns.tolist()})


# =========================================================================== anomaly
_UCR_ANOMALY_RE = re.compile(r"^\d+_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")


def _ucr_anomaly_file(name: str, data_dir) -> Path:
    """Return one official UCR 2021 anomaly-series file, downloading it if needed.

    UCR filenames encode the normal training-prefix length and the inclusive anomaly
    interval.  We intentionally extract just the selected member rather than unpacking
    the complete archive.
    """
    root = Path(data_dir) / "anomaly_ucr"
    found = sorted(root.rglob(f"*_UCR_Anomaly_{name}_*.txt")) if root.exists() else []
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(f"multiple UCR anomaly files found for {name}: {found}")

    archive = _download(UCR_ANOMALY_URL, root / "UCR_TimeSeriesAnomalyDatasets2021.zip")
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist()
                   if not Path(m).name.startswith("_")
                   and re.match(rf"^\d+_UCR_Anomaly_{re.escape(name)}_\d+_\d+_\d+\.txt$", Path(m).name)]
        if len(members) != 1:
            raise FileNotFoundError(f"expected exactly one UCR anomaly member for {name}, found {members}")
        dest = root / Path(members[0]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(members[0]) as src, open(dest, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    return dest


def _ucr_anomaly_entities(name: str, data_dir):
    """One UCR univariate series as ``(id, train, test)`` entity data.

    The archive's integer locations are one-based and inclusive.  Its prefix before
    ``train_end`` is normal training data; labels are generated only for the test tail.
    """
    path = _ucr_anomaly_file(name, data_dir)
    match = _UCR_ANOMALY_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected UCR anomaly filename: {path.name}")
    parsed_name, train_end, anomaly_start, anomaly_end = match.groups()
    if parsed_name != name:
        raise ValueError(f"UCR filename {path.name} does not match requested dataset {name}")
    x = np.loadtxt(path, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"{path.name}: expected one value per timestamp, got shape {x.shape}")
    train_end, anomaly_start, anomaly_end = map(int, (train_end, anomaly_start, anomaly_end))
    if not (0 < train_end < anomaly_start <= anomaly_end <= len(x)):
        raise ValueError(f"invalid UCR split/anomaly positions in {path.name} (length={len(x)})")
    labels = np.zeros(len(x), dtype=np.int64)
    labels[anomaly_start - 1 : anomaly_end] = 1  # UCR locations are one-based and inclusive
    return [(name, (x[:train_end], np.zeros(train_end, dtype=np.int64)),
             (x[train_end:], labels[train_end:]))]


def _tsb_entities(name, data_dir, max_entities=None):
    root = Path(data_dir) / "anomaly" / name
    listing = _json(f"{PILE_API}/anomaly_detection/TSB-UAD-Public/{TSB_DIRS[name]}", root / "_listing.json")
    files = sorted(x["path"] for x in listing if x["path"].endswith(".out"))
    ids = sorted({Path(f).name.split(".")[0] for f in files})
    if max_entities:
        ids = ids[:max_entities]
    ents = []
    for e in ids:
        parts = {}
        for split in ("train", "test"):
            rel = f"anomaly_detection/TSB-UAD-Public/{TSB_DIRS[name]}/{e}.{split}.out"
            arr = np.loadtxt(_download(f"{PILE}/{rel}", root / f"{e}.{split}.out"), delimiter=",", ndmin=2)
            parts[split] = (arr[:, 0].astype(np.float32), arr[:, 1].astype(np.int64))
        ents.append((e, parts["train"], parts["test"]))
    return ents


def _smd_entities(data_dir, machines=None, channels=None, max_entities=None):
    root = Path(data_dir) / "anomaly" / "SMD"
    if not machines:
        listing = _json(SMD_API, root / "_listing.json")
        machines = sorted(Path(x["name"]).stem for x in listing if x["name"].endswith(".txt"))
    ents = []
    for m in machines:
        tr = np.loadtxt(_download(f"{SMD_RAW}/train/{m}.txt", root / "train" / f"{m}.txt"), delimiter=",")
        te = np.loadtxt(_download(f"{SMD_RAW}/test/{m}.txt", root / "test" / f"{m}.txt"), delimiter=",")
        lab = np.loadtxt(_download(f"{SMD_RAW}/test_label/{m}.txt", root / "test_label" / f"{m}.txt")).astype(np.int64)
        chans = range(tr.shape[1]) if channels in (None, "all") else channels
        for c in chans:
            if np.ptp(tr[:, c]) < 1e-8 and np.ptp(te[:, c]) < 1e-8:
                continue  # constant channel: nothing to reconstruct or explain
            ents.append((f"{m}@{c}", (tr[:, c].astype(np.float32), np.zeros(len(tr), np.int64)),
                         (te[:, c].astype(np.float32), lab)))
    return ents[:max_entities] if max_entities else ents


def _windows(series, labels, L, stride, cover_end=True):
    """Cut a 1-D series into length-L windows. Short series are edge-padded on the left
    (``pad`` records how many leading points are padding; MOMENT masks them)."""
    T = len(series)
    if T < L:
        p = L - T
        return (np.concatenate([np.full(p, series[0]), series])[None], np.concatenate([np.full(p, -1), labels])[None],
                np.array([-p]), np.array([p]))
    starts = list(range(0, T - L + 1, stride))
    if cover_end and starts[-1] != T - L:
        starts.append(T - L)
    starts = np.array(starts)
    idx = starts[:, None] + np.arange(L)[None]
    return series[idx], labels[idx], starts, np.zeros(len(starts), dtype=int)


def load_anomaly(name, data_dir, seq_len=512, train_stride=None, val_fraction=0.1, seed=0,
                 max_entities=None, smd_machines=None, smd_channels=None, **_):
    if name in UCR_ANOMALY_DATASETS:
        ents = _ucr_anomaly_entities(name, data_dir)
    elif name in TSB_DIRS:
        ents = _tsb_entities(name, data_dir, max_entities)
    elif name == "SMD":
        ents = _smd_entities(data_dir, smd_machines, smd_channels, max_entities)
    else:
        raise ValueError(f"unknown anomaly dataset {name}")
    L = seq_len
    tr_X, tr_lab, tr_pad, te_X, te_lab, te_pad, te_ent, te_start = [], [], [], [], [], [], [], []
    tests = {}
    for eid, (xtr, ltr), (xte, lte) in ents:
        w, lab, _, pad = _windows(xtr, ltr, L, train_stride or L)
        tr_X.append(w), tr_lab.append(lab), tr_pad.append(pad)
        w, lab, st, pad = _windows(xte, lte, L, L)
        te_X.append(w), te_lab.append(lab), te_pad.append(pad)
        te_ent += [eid] * len(w)
        te_start.append(st)
        tests[eid] = {"length": len(xte), "labels": lte}
    Xtr = np.concatenate(tr_X)[:, None, :]
    Ltr = np.concatenate(tr_lab)
    Ptr = np.concatenate(tr_pad)
    wl = (Ltr > 0).any(1).astype(int)  # window label for Path B: any anomalous point
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    n_va = max(1, int(round(len(Xtr) * val_fraction)))
    va, tr = np.sort(perm[:n_va]), np.sort(perm[n_va:])
    fit_y = wl[tr] if len(np.unique(wl[tr])) > 1 else None  # all-normal train -> unsupervised words
    mk = lambda i: ArraySplit(Xtr[i], None, L, pad=Ptr[i], meta={"window_labels": wl[i]})  # noqa: E731
    train = mk(tr)
    train._fit_y = fit_y
    test = ArraySplit(np.concatenate(te_X)[:, None, :], None, L, pad=np.concatenate(te_pad),
                      meta={"entity": np.array(te_ent), "start": np.concatenate(te_start),
                            "point_labels": np.concatenate(te_lab), "entities": tests})
    return TaskData("anomaly", name, 1, 0, train, mk(va), test,
                    {"n_entities": len(ents), "supervised_words": fit_y is not None})


# =========================================================================== dispatch
def load_task(task: str, dataset: str, data_dir, seq_len: int = 512, **kw) -> TaskData:
    fn = {"classification": load_classification, "forecasting": load_forecasting, "anomaly": load_anomaly}[task]
    return fn(dataset, data_dir, seq_len=seq_len, **kw)


def synthetic_task(task: str, n_train=48, n_test=24, length=128, n_channels=1, horizon=16, seed=0) -> TaskData:
    """Tiny synthetic data for tests and smoke runs (no network)."""
    rng = np.random.default_rng(seed)

    def cls_data(n):
        y = rng.integers(0, 2, n)
        X = rng.normal(0, 0.3, (n, n_channels, length)).astype(np.float32)
        for i in np.flatnonzero(y == 1):
            p = rng.integers(0, length - 16)
            X[i, :, p : p + 16] += np.hanning(16) * 4
        return X, y

    if task == "classification":
        (Xa, ya), (Xb, yb) = cls_data(n_train), cls_data(n_test)
        tr, va = _stratified_split(ya, 0.2, seed)
        return TaskData(task, "synthetic", n_channels, 2, ArraySplit(Xa[tr], ya[tr]), ArraySplit(Xa[va], ya[va]),
                        ArraySplit(Xb, yb), {"classes": ["0", "1"]})
    if task == "forecasting":
        t = np.arange(4000)
        data = np.stack([np.sin(t / (8 + c)) + rng.normal(0, 0.1, len(t)) for c in range(n_channels)], 1)
        tr = ForecastSplit(data[:2400], 512, horizon, 32)
        tr.fit_labels()
        return TaskData(task, "synthetic", n_channels, horizon, tr,
                        ForecastSplit(data[2400 - 512 : 3200], 512, horizon, 64, None, tr.direction_edges),
                        ForecastSplit(data[3200 - 512 :], 512, horizon, 64, None, tr.direction_edges))
    if task == "anomaly":
        def series(n, anomalous):
            x = np.sin(np.arange(n) / 6) + rng.normal(0, 0.05, n)
            lab = np.zeros(n, np.int64)
            if anomalous:
                for p in rng.integers(100, n - 100, 3):
                    x[p : p + 20] += 3
                    lab[p : p + 20] = 1
            return x.astype(np.float32), lab
        xtr, ltr = series(512 * n_train // 8, False)
        xte, lte = series(512 * 4, True)
        w, lab, _, pad = _windows(xtr, ltr, 512, 512)
        tw, tl, ts, tp = _windows(xte, lte, 512, 512)
        tr = ArraySplit(w[1:, None], None, pad=pad[1:], meta={"window_labels": (lab[1:] > 0).any(1).astype(int)})
        va = ArraySplit(w[:1, None], None, pad=pad[:1], meta={"window_labels": np.zeros(1, int)})
        te = ArraySplit(tw[:, None], None, pad=tp, meta={"entity": np.array(["syn"] * len(tw)), "start": ts,
                                                         "point_labels": tl,
                                                         "entities": {"syn": {"length": len(xte), "labels": lte}}})
        return TaskData(task, "synthetic", 1, 0, tr, va, te, {"n_entities": 1, "supervised_words": False})
    raise ValueError(task)
