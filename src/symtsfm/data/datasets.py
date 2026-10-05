"""Additional dataset routes used by the run scripts.

The existing project loader is reused for UCR/UEA, forecasting, and its normal
anomaly routes. This module adds three univariate TSB-UAD members explicitly
reported in MOMENT's Time-series Pile table and routes MSL/SMAP/SMD to the
multivariate loader.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MOMENT_UAD_FILES = {
    "1sddb40": "109_UCR_Anomaly_1sddb40_35000_52000_52620.out",
    "CIMIS44AirTemperature3": "115_UCR_Anomaly_CIMIS44AirTemperature3_4000_6520_6544.out",
    "ECG2": "120_UCR_Anomaly_ECG2_15000_16000_16100.out",
}
_UAD_RE = re.compile(r"^\d+_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.out$")
PILE = "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main"
CONSTANT_STD = 1e-4  # training std below this = constant channel (sensor never moved)


def _download(url: str, destination: Path, retries: int = 3) -> Path:
    """Download once to persistent cache without accepting a partial file."""
    if destination.exists() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "research-base1/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as stream:
                while block := response.read(1 << 20):
                    stream.write(block)
            partial.replace(destination)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def load_moment_uad(name: str, data_dir, seq_len: int = 512, train_stride: int | None = None,
                    val_fraction: float = 0.1, seed: int = 0, **_):
    """Load one MOMENT-paper univariate anomaly series into the shared TaskData API."""
    from symtsfm.data.loaders import ArraySplit, TaskData, _windows

    filename = MOMENT_UAD_FILES[name]
    parsed = _UAD_RE.match(filename)
    if parsed is None:
        raise ValueError(f"unexpected UAD filename: {filename}")
    parsed_name, train_end, anomaly_start, anomaly_end = parsed.groups()
    if parsed_name != name:
        raise ValueError(f"dataset/file mismatch: {name} vs {filename}")
    path = _download(f"{PILE}/anomaly_detection/TSB-UAD-Public/KDD21/{filename}",
                     Path(data_dir) / "moment_uad" / filename)
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if raw.shape[1] != 2:
        raise ValueError(f"{filename} must have value,label columns; found {raw.shape}")
    values, labels = raw[:, 0], raw[:, 1].astype(np.int64)
    train_end, anomaly_start, anomaly_end = map(int, (train_end, anomaly_start, anomaly_end))
    if not (0 < train_end < anomaly_start <= anomaly_end <= len(values)):
        raise ValueError(f"invalid split positions encoded by {filename}")
    if labels.sum() == 0:
        labels[anomaly_start - 1:anomaly_end] = 1

    normal_train = values[:train_end]
    test_values, test_labels = values[train_end:], labels[train_end:]
    length = int(seq_len)
    train_x, train_lab, _, train_pad = _windows(normal_train, np.zeros(len(normal_train), dtype=np.int64),
                                                 length, train_stride or length)
    test_x, test_lab, test_start, test_pad = _windows(test_values, test_labels, length, length)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_x))
    n_val = max(1, int(round(len(train_x) * val_fraction)))
    val_i, train_i = np.sort(order[:n_val]), np.sort(order[n_val:])

    def split(indices):
        return ArraySplit(train_x[indices, None, :], None, length, pad=train_pad[indices],
                          meta={"window_labels": train_lab[indices].max(axis=1)})

    test = ArraySplit(test_x[:, None, :], None, length, pad=test_pad,
                      meta={"entity": np.asarray([name] * len(test_x)), "start": test_start,
                            "point_labels": test_lab,
                            "entities": {name: {"length": len(test_values), "labels": test_labels}}})
    return TaskData("anomaly", name, 1, 0, split(train_i), split(val_i), test,
                    {"n_entities": 1, "anomaly_ratio": float(test_labels.mean()),
                     "channel_names": ["value"], "source": "MOMENT Time-series Pile / TSB-UAD"})


class _LazyMVAnomalySplit:
    """Window-index view over multivariate anomaly series.

    Unlike the legacy loader, this class does not allocate every overlapping
    stride-1 train window. That distinction is essential for full SMD runs.
    """

    def __init__(self, series: dict, references: list[tuple[str, int]], seq_len: int,
                 entities: dict | None = None, point_labels: list[np.ndarray] | None = None):
        self.series = series
        self.references = np.asarray(references, dtype=object)
        self.seq_len = seq_len
        self.entities = entities
        self._point_labels = point_labels
        self.meta = {"window_labels": np.zeros(len(self.references), dtype=np.int64)}
        if entities is not None:
            self.meta.update({
                "entity": self.references[:, 0],
                "start": self.references[:, 1].astype(int),
                "entities": entities,
                "point_labels": np.stack(point_labels).astype(np.int64),
            })

    def __len__(self):
        return len(self.references)

    def x_raw(self, idx=None):
        selected = self.references if idx is None else self.references[np.asarray(idx)]
        selected = np.atleast_2d(selected)
        return np.stack([self.series[str(entity)][:, int(start): int(start) + self.seq_len]
                         for entity, start in selected]).astype(np.float32, copy=False)

    def x_model(self, idx):
        x = self.x_raw(idx)
        return x, np.ones((len(x), self.seq_len), dtype=np.float32)

    def target(self, idx):
        return self.x_raw(idx)

    def fit_labels(self, idx=None):
        # Training contains only normal samples; supervised information gain is undefined.
        return None


def _starts(length: int, window: int, stride: int) -> np.ndarray:
    if length < window:
        raise ValueError("anomaly evaluation requires anomaly series at least as long as seq_len")
    result = np.arange(0, length - window + 1, stride, dtype=int)
    if result[-1] != length - window:
        result = np.r_[result, length - window]
    return result


def load_multivariate_anomaly_lazy(name: str, data_dir, seq_len: int = 96, train_stride: int | None = None,
                                   val_fraction: float = 0.1, seed: int = 0, max_entities=None,
                                   smd_machines=None, nasa_root=None, scale: bool = True, **_):
    """Full-data MSL/SMAP/SMD route with lazy training windows and full entities."""
    from symtsfm.data.anomaly_mv import _nasa_entities, _psm_entities, _smd_entities
    from symtsfm.data.loaders import TaskData

    upper = name.upper()
    if upper in {"MSL", "SMAP"}:
        raw_entities = _nasa_entities(upper, data_dir, max_entities, nasa_root)
    elif upper == "SMD":
        raw_entities = _smd_entities(data_dir, smd_machines, max_entities)
    elif upper == "PSM":
        raw_entities = _psm_entities(data_dir, max_entities)
    else:
        raise ValueError(f"unknown multivariate anomaly dataset {name!r}")
    if not raw_entities:
        raise RuntimeError(f"no entities were loaded for {name}")

    # NASA MSL includes T-9, whose 439-step training segment is shorter than
    # the fixed 512-step context used by this experiment.  It cannot produce a
    # valid training (or validation) window.  Exclude such entities explicitly
    # rather than padding or silently changing their temporal content.
    eligible_entities, skipped_entities = [], []
    for entity, raw_train, train_labels, raw_test, labels in raw_entities:
        if raw_train.ndim != 2 or raw_test.ndim != 2:
            raise ValueError(f"{entity}: expected channels-first 2-D arrays")
        if raw_train.shape[1] < seq_len or raw_test.shape[1] < seq_len:
            skipped_entities.append(str(entity))
            continue
        eligible_entities.append((entity, raw_train, train_labels, raw_test, labels))
    if skipped_entities:
        log.warning("%s: excluding %d entities shorter than seq_len=%d: %s",
                    name, len(skipped_entities), seq_len, ", ".join(skipped_entities))
    raw_entities = eligible_entities
    if not raw_entities:
        raise ValueError(f"{name}: no entities have both train/test length >= seq_len={seq_len}")

    train_series, test_series, test_entities = {}, {}, {}
    train_refs, test_refs, test_windows = [], [], []
    channel_count = raw_entities[0][1].shape[0]
    train_stride = int(train_stride or seq_len)
    n_constant_channels = 0
    for entity, raw_train, _train_labels, raw_test, labels in raw_entities:
        if raw_train.shape[0] != channel_count:
            raise ValueError(f"{entity}: expected {channel_count} channels, got {raw_train.shape[0]}")
        if scale:
            mean = raw_train.mean(axis=1, keepdims=True)
            std = raw_train.std(axis=1, keepdims=True)
            # A channel that is (almost) constant in training must not be divided by its
            # ~0 std: any test deviation would explode to ~1e7-1e8.  Like scikit-learn's
            # StandardScaler, such channels are only centred (scale 1).
            constant = std < CONSTANT_STD
            n_constant_channels += int(constant.sum())
            std = np.where(constant, 1.0, std)
            raw_train, raw_test = (raw_train - mean) / std, (raw_test - mean) / std
        entity = str(entity)
        train_series[entity], test_series[entity] = raw_train.astype(np.float32), raw_test.astype(np.float32)
        test_entities[entity] = {"length": raw_test.shape[1], "labels": np.asarray(labels, dtype=np.int64)}
        train_refs.extend((entity, int(start)) for start in _starts(raw_train.shape[1], seq_len, train_stride))
        for start in _starts(raw_test.shape[1], seq_len, seq_len):
            test_refs.append((entity, int(start)))
            test_windows.append(np.asarray(labels[start:start + seq_len], dtype=np.int64))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_refs))
    n_val = max(1, int(round(len(train_refs) * val_fraction)))
    val_refs = [train_refs[i] for i in np.sort(order[:n_val])]
    train_refs = [train_refs[i] for i in np.sort(order[n_val:])]
    train = _LazyMVAnomalySplit(train_series, train_refs, seq_len)
    val = _LazyMVAnomalySplit(train_series, val_refs, seq_len)
    test = _LazyMVAnomalySplit(test_series, test_refs, seq_len, test_entities, test_windows)
    anomaly_ratio = float(np.concatenate([v["labels"] for v in test_entities.values()]).mean())
    log.info("%s: %d of %d entity-channels are constant in training (centred, not rescaled)",
             name, n_constant_channels, channel_count * len(raw_entities))
    return TaskData("anomaly", name, channel_count, 0, train, val, test,
                    {"n_entities": len(raw_entities), "n_channels": channel_count,
                     "channel_names": [f"channel_{i}" for i in range(channel_count)],
                     "entity_ids": list(test_entities), "anomaly_ratio": anomaly_ratio,
                     "supervised_words": False, "lazy_train_windows": True,
                     "constant_train_channels": n_constant_channels,
                     "skipped_short_entities": skipped_entities})


def load_task(task: str, dataset: str, data_dir, seq_len: int = 512, **kwargs):
    """Dispatch to the correct research route while keeping existing loaders untouched."""
    if task == "anomaly" and dataset in MOMENT_UAD_FILES:
        return load_moment_uad(dataset, data_dir, seq_len=seq_len, **kwargs)
    if task == "anomaly" and dataset.upper() in {"MSL", "SMAP", "SMD", "PSM"}:
        return load_multivariate_anomaly_lazy(dataset, data_dir, seq_len=seq_len, **kwargs)
    from symtsfm.data.loaders import load_task as stock_load_task
    return stock_load_task(task, dataset, data_dir, seq_len=seq_len, **kwargs)
