"""Dual-path wrapper around stock UniTS (Phase 2).

=============================================================================
STEP 1 ANALYSIS -- UniTS task tokens (mims-harvard/UniTS @ 0e02814, models/UniTS.py)
=============================================================================
Input x [B, L, V] -> ``tokenize``: per-sample instance norm over time, right-pad to a
multiple of patch_len (16), non-overlapping patch embedding -> [B*V, Ltok, d] (d=128
for the x128 checkpoint). ``prepare_prompt`` then builds, per variable:

  classification : [prompt_tokens(10) | patches + pos-emb | CLS token]           (CLS appended LAST)
                   CLS token = ``cls_tokens[task_data_name]`` [1, V, 1, d]
  forecasting    : [prompt_tokens(10) | patches | GEN tokens] + pos-emb (non-prompt part)
                   GEN tokens = ``mask_tokens[dataset]`` repeated pred_token_len times and
                   passed through ``prompt2forecat`` (DynamicLinear) -> [1, V, P, d]
  anomaly        : [prompt_tokens(10) | patches + pos-emb]  (no task token; reconstruction)

``backbone`` (3 BasicBlocks: sequence attention + variable attention + MLP) keeps the
token layout. Heads, both SHARED across all tasks of their family:
  ``cls_head`` (CLSHead): proj_in -> cross-attention from the LAST token (CLS) over all
      tokens -> MLP -> CLS feature [B, V, 1, d]; logits = mean_V <CLS feature,
      category_tokens[task] [1, V, n_class, d]>.
  ``forecast_head`` (ForecastHead): all tokens [B, V, 10+Ltok+P, d] -> proj_in ->
      DynamicLinear over the token axis -> MLP -> proj_out -> fold back to time.

DECISION -- where Path B is fused: (b) the same late-fusion point as Phase 1
(post-encoder, pre-head), NOT (a) an extra input token.
  * (a) would put a foreign token into every attention layer: the "deep/early fusion"
    the technical documentation rejects, and it would change what the frozen blocks see.
  * UniTS already exposes a clean pre-head representation per task: the CLS feature
    for classification, the backbone token states for GEN tasks. Fusing there keeps
    Phase 1's validated design and the "gate -> 0 recovers the baseline" guarantee.
  Concretely, the V1 adapter wraps the stock head:
      fused, g = LateFusionGate(z_a, sym)          # [z_a ; g * p]  (fusion/gate.py as-is)
      z_a'     = fused[:d] + W_s fused[d:]          # W_s zero-initialised, d -> d
  V4 instead treats the individual mined words as a small symbolic memory:
      C = CrossAttention(query=z_a, key/value=pattern_tokens(sym))
      z_a' = z_a + W_o (g * C)                      # W_o zero-initialised
  The query is therefore task-conditioned: CLS features for classification,
  post-encoder token states (including GEN tokens) for forecasting, and
  reconstruction token states for anomaly detection.
  so the pretrained ``cls_head`` / ``forecast_head`` keep their input width, and at
  initialisation (or with g = 0) the output is exactly stock UniTS. Symbolic features
  are per variable ([B, V, F]) and match UniTS's per-variable token layout.
  Classification fuses the CLS feature; GEN tasks broadcast the window's vector to every token.

  V5 is forecasting-only. It keeps the upstream UniTS forecast head unchanged,
  lets only its final GEN tokens query symbolic word tokens, and adds a distinct
  symbolic residual to each output forecast patch. The residual starts at zero,
  so V5 is exactly stock UniTS before training.

Adapters are keyed by task and act only when that task's symbolic input is set, so
adding a fusion hook for one task cannot change another task's outputs (see
tests/test_units_wrapper.py::test_cross_task_non_interference). Note that UniTS
shares ``forecast_head`` between forecasting, imputation and anomaly detection; with
R1 "task head" training in a *multi-task* model, training it for one GEN task would
move the others. Every cell here is a single-task model, so that does not arise.

R1 for UniTS: frozen = patch embedding, positional embedding, transformer blocks,
prompt2forecat. Trainable = this dataset's task tokens (prompt / mask / CLS /
category, UniTS's own prompt-tuning parameters), the task head, and fusion. Baseline
and guided train exactly the same UniTS parameters.
=============================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

from symtsfm.fusion.gate import LateFusionGate

log = logging.getLogger(__name__)

UNITS_REPO = "https://github.com/mims-harvard/UniTS.git"
UNITS_COMMIT = "0e0281482864017cac8832b2651906ff5375a34e"
CKPT_URL = "https://github.com/mims-harvard/UniTS/releases/download/ckpt/{name}"
TASK_NAMES = {"classification": "classification", "forecasting": "long_term_forecast", "anomaly": "anomaly_detection"}
FROZEN_PREFIXES = ("patch_embeddings.", "position_embedding.", "blocks.", "prompt2forecat.")
DEBUG_ARGS = {"d_model": 32, "e_layers": 1, "n_heads": 4, "patch_len": 16, "stride": 16, "prompt_num": 10, "dropout": 0.1}


# =========================================================================== setup
def ensure_units(repo_dir: str | Path, checkpoint: str | Path | None) -> tuple[Path, Path | None]:
    """Clone UniTS at the pinned commit / download the checkpoint if missing (setup-time only)."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / "models" / "UniTS.py").exists():
        log.info("cloning UniTS into %s", repo_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", UNITS_REPO, str(repo_dir)], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", "-q", UNITS_COMMIT], check=True)
    if checkpoint is None:
        return repo_dir, None
    ck = Path(checkpoint)
    if not ck.exists():
        ck.parent.mkdir(parents=True, exist_ok=True)
        url = CKPT_URL.format(name=ck.name)
        try:
            log.info("downloading %s", url)
            urllib.request.urlretrieve(url, ck.with_suffix(".part"))
            ck.with_suffix(".part").replace(ck)
        except Exception as e:  # never fall back to random weights
            raise RuntimeError(f"UniTS checkpoint {url} not reachable ({e}); refusing to continue") from e
    return repo_dir, ck


def load_units_module(repo_dir: Path):
    spec = importlib.util.spec_from_file_location("units_upstream", Path(repo_dir) / "models" / "UniTS.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def task_config(task, dataset, n_channels, seq_len, out_dim):
    name = {"classification": f"CLS_{dataset}", "forecasting": f"LTF_{dataset}_p{out_dim}",
            "anomaly": f"AD_{dataset}"}[task]
    cfg = {"task_name": TASK_NAMES[task], "dataset": dataset, "enc_in": n_channels, "seq_len": seq_len,
           "pred_len": out_dim if task == "forecasting" else 0, "label_len": 0}
    if task == "classification":
        cfg["num_class"] = out_dim
    return [name, cfg]


# =========================================================================== fusion adapters
class ResidualFusion(nn.Module):
    """[z ; g p] -> z + W_s (g p), W_s zero-initialised (identity at init)."""

    def __init__(self, d: int, sym_dim: int, fusion_cfg: dict | None = None):
        super().__init__()
        self.gate = LateFusionGate(d, sym_dim, **(fusion_cfg or {}))
        self.out = nn.Linear(self.gate.proj_dim, d, bias=False)
        nn.init.zeros_(self.out.weight)
        self.d = d

    def forward(self, z, sym, force_gate=None):
        fused, g = self.gate(z, sym.to(z.dtype), force_gate)
        return fused[..., : self.d] + self.out(fused[..., self.d :]), g


def symbolic_pattern_tokens(sym: torch.Tensor, token_dim: int) -> torch.Tensor:
    """Convert the existing [B,V,3*K+1] layout into K word tokens plus rarity."""
    if sym.ndim == 4:  # legacy callers may broadcast along an encoder-token axis
        sym = sym[:, :, 0, :]
    if sym.ndim != 3:
        raise ValueError(f"expected symbolic features [B,V,F], got {tuple(sym.shape)}")
    core, rarity = sym[..., :-1], sym[..., -1:]
    pad = (-core.shape[-1]) % token_dim
    if pad:
        core = torch.nn.functional.pad(core, (0, pad))
    words = core.reshape(*core.shape[:-1], -1, token_dim)
    rare = torch.zeros(*rarity.shape[:-1], 1, token_dim, device=sym.device, dtype=sym.dtype)
    rare[..., 0] = rarity
    return torch.cat([words, rare], dim=-2)


def location_aware_symbolic_pattern_tokens(sym: torch.Tensor) -> torch.Tensor:
    """Build one token per mined word while retaining its matched location.

    The layout is [presence, frequency, distance] * K, rarity, then K
    normalised locations.  This lets a forecast query distinguish a recent
    motif from the same motif early in the lookback window.
    """
    if sym.ndim == 4:
        sym = sym[:, :, 0, :]
    if sym.ndim != 3 or (sym.shape[-1] - 1) % 4:
        raise ValueError(
            "location-aware symbolic tokens require [B,V,4*K+1] features; "
            f"got {tuple(sym.shape)}"
        )
    k = (sym.shape[-1] - 1) // 4
    core = sym[..., : 3 * k].reshape(*sym.shape[:-1], k, 3)
    location = sym[..., 3 * k : 4 * k].unsqueeze(-1)
    words = torch.cat([core, location], dim=-1)
    rarity = sym[..., 4 * k : 4 * k + 1]
    rare = torch.zeros(*rarity.shape[:-1], 1, 4, device=sym.device, dtype=sym.dtype)
    rare[..., 0] = rarity
    return torch.cat([words, rare], dim=-2)


class SymbolicCrossAttentionFusion(nn.Module):
    """Task-conditioned attention from UniTS representations to mined words.

    ``sym`` has the existing symbolic feature layout ``[..., 3*K + 1]``:
    three values (presence, distance, frequency) for each of K selected words,
    followed by a rarity value.  This module converts those groups into K+1
    symbolic *tokens*, rather than collapsing them to one global feature
    vector.  Every UniTS representation token then queries the pattern tokens.

    ``out`` is exactly zero-initialised. Consequently this adapter is an exact
    identity at initialisation, regardless of attention or gate values.
    """

    def __init__(self, d: int, sym_dim: int, *, num_heads: int = 4,
                 gate_init_bias: float = 0.0, attention_dropout: float = 0.0,
                 token_dim: int = 3):
        super().__init__()
        if d % num_heads:
            raise ValueError(f"d={d} must be divisible by num_heads={num_heads}")
        if token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.d, self.sym_dim, self.token_dim = d, sym_dim, token_dim
        self.pattern_proj = nn.Linear(token_dim, d)
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=attention_dropout, batch_first=True)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)
        self.out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.out.weight)
        self.last_attention = None

    def _pattern_tokens(self, sym: torch.Tensor) -> torch.Tensor:
        """Convert [B,V,F] (or broadcast [B,V,T,F]) to [B,V,K+1,token_dim]."""
        return symbolic_pattern_tokens(sym, self.token_dim)

    def forward(self, z, sym, force_gate=None):
        """``z`` [B,V,T,d] -> per-token cross-attended residual and gate."""
        if z.ndim != 4:
            raise ValueError(f"expected UniTS token states [B,V,T,d], got {tuple(z.shape)}")
        B, V, T, D = z.shape
        if D != self.d:
            raise ValueError(f"expected embedding size {self.d}, got {D}")
        pattern = self._pattern_tokens(sym.to(z.dtype))
        query = z.reshape(B * V, T, D)
        key_value = self.pattern_proj(pattern).reshape(B * V, pattern.shape[-2], D)
        context, weights = self.attention(query, key_value, key_value, need_weights=True,
                                          average_attn_weights=True)
        if force_gate is None:
            g = torch.sigmoid(self.gate(query))
        else:
            g = torch.full(query.shape[:-1] + (1,), float(force_gate), device=z.device, dtype=z.dtype)
        self.last_attention = weights.detach().reshape(B, V, T, pattern.shape[-2])
        out = query + self.out(g * context)
        return out.reshape(B, V, T, D), g.reshape(B, V, T, 1)


class LocationAwareSymbolicCrossAttentionFusion(SymbolicCrossAttentionFusion):
    """Cross-attention whose word tokens retain match location and recency."""

    def __init__(self, d: int, sym_dim: int, **kwargs):
        kwargs.pop("token_dim", None)
        super().__init__(d, sym_dim, token_dim=4, **kwargs)

    def _pattern_tokens(self, sym: torch.Tensor) -> torch.Tensor:
        return location_aware_symbolic_pattern_tokens(sym)


class HorizonSymbolicResidualFusion(nn.Module):
    """V5 forecast-only adapter: GEN-token pattern retrieval -> output-patch residual.

    The final ``ceil(H / patch_len)`` backbone tokens are the UniTS generated
    forecast-query tokens for horizon ``H``. Only those tokens attend to the
    symbolic word memory. Their contexts are mapped to additive output patches
    *after* the upstream forecast head. ``residual_out`` is zero-initialised,
    which makes this adapter an exact identity at initialization.
    """

    def __init__(self, d: int, sym_dim: int, *, patch_len: int, num_heads: int = 4,
                 gate_init_bias: float = -2.0, attention_dropout: float = 0.0,
                 token_dim: int = 3):
        super().__init__()
        if d % num_heads:
            raise ValueError(f"d={d} must be divisible by num_heads={num_heads}")
        if patch_len < 1 or token_dim < 1:
            raise ValueError("patch_len and token_dim must be positive")
        self.d, self.sym_dim, self.patch_len, self.token_dim = d, sym_dim, patch_len, token_dim
        self.pattern_proj = nn.Linear(token_dim, d)
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=attention_dropout, batch_first=True)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)
        self.residual_out = nn.Linear(d, patch_len)
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)
        self.last_attention = None

    def forward(self, x_full, sym, horizon: int, force_gate=None):
        """Return a [B,V,H] residual plus [B,V,P,1] gates for forecast horizon H."""
        if x_full.ndim != 4:
            raise ValueError(f"expected UniTS token states [B,V,T,d], got {tuple(x_full.shape)}")
        B, V, T, D = x_full.shape
        if D != self.d:
            raise ValueError(f"expected embedding size {self.d}, got {D}")
        n_gen = math.ceil(horizon / self.patch_len)
        if n_gen > T:
            raise ValueError(f"forecast needs {n_gen} GEN tokens but only {T} are available")
        pattern = symbolic_pattern_tokens(sym.to(x_full.dtype), self.token_dim)
        query = x_full[:, :, -n_gen:, :].reshape(B * V, n_gen, D)
        key_value = self.pattern_proj(pattern).reshape(B * V, pattern.shape[-2], D)
        context, weights = self.attention(query, key_value, key_value, need_weights=True,
                                          average_attn_weights=True)
        if force_gate is None:
            gate = torch.sigmoid(self.gate(query))
        else:
            gate = torch.full(query.shape[:-1] + (1,), float(force_gate), device=x_full.device,
                              dtype=x_full.dtype)
        self.last_attention = weights.detach().reshape(B, V, n_gen, pattern.shape[-2])
        # Gate the complete correction, including its learned bias. This makes
        # ``force_gate=0`` an exact baseline recovery even after training.
        patches = (gate * self.residual_out(context)).reshape(B, V, n_gen * self.patch_len)
        return patches[..., :horizon], gate.reshape(B, V, n_gen, 1)


class HorizonPreHeadSymbolicFusion(nn.Module):
    """Inject symbolic retrieval into only the UniTS GEN tokens before its head.

    Unlike :class:`HorizonSymbolicResidualFusion`, this changes the final
    forecast-query representations *before* the stock ``ForecastHead`` maps
    them to values.  The adapter is still an exact identity at initialisation
    (and for ``force_gate=0``), because its internal output projection is zero.
    """

    def __init__(self, d: int, sym_dim: int, *, patch_len: int, num_heads: int = 4,
                 gate_init_bias: float = -2.0, attention_dropout: float = 0.0,
                 token_dim: int = 3):
        super().__init__()
        self.patch_len = patch_len
        self.inner = SymbolicCrossAttentionFusion(
            d, sym_dim, num_heads=num_heads, gate_init_bias=gate_init_bias,
            attention_dropout=attention_dropout, token_dim=token_dim,
        )

    @property
    def last_attention(self):
        return self.inner.last_attention

    def forward(self, x_full, sym, horizon: int, force_gate=None):
        if x_full.ndim != 4:
            raise ValueError(f"expected UniTS token states [B,V,T,d], got {tuple(x_full.shape)}")
        n_gen = math.ceil(horizon / self.patch_len)
        if n_gen > x_full.shape[2]:
            raise ValueError(f"forecast needs {n_gen} GEN tokens but only {x_full.shape[2]} are available")
        guided_gen, gate = self.inner(x_full[:, :, -n_gen:, :], sym, force_gate)
        # Avoid in-place modification of backbone output: autograd also needs
        # the unmodified token tensor for the native UniTS path.
        return torch.cat([x_full[:, :, :-n_gen, :], guided_gen], dim=2), gate


class HorizonLocationPreHeadSymbolicFusion(nn.Module):
    """Location-aware pre-head fusion for the final UniTS GEN query tokens."""

    def __init__(self, d: int, sym_dim: int, *, patch_len: int, num_heads: int = 4,
                 gate_init_bias: float = -2.0, attention_dropout: float = 0.0,
                 token_dim: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.inner = LocationAwareSymbolicCrossAttentionFusion(
            d, sym_dim, num_heads=num_heads, gate_init_bias=gate_init_bias,
            attention_dropout=attention_dropout, token_dim=token_dim,
        )

    @property
    def last_attention(self):
        return self.inner.last_attention

    def forward(self, x_full, sym, horizon: int, force_gate=None):
        n_gen = math.ceil(horizon / self.patch_len)
        if x_full.ndim != 4 or n_gen > x_full.shape[2]:
            raise ValueError("invalid UniTS forecast-token shape for location-aware fusion")
        guided_gen, gate = self.inner(x_full[:, :, -n_gen:, :], sym, force_gate)
        return torch.cat([x_full[:, :, :-n_gen, :], guided_gen], dim=2), gate


def build_fusion_adapter(d: int, sym_dim: int, fusion_cfg: dict | None = None,
                         *, patch_len: int | None = None) -> nn.Module:
    """Create the selected V1, V4, or forecast-only V5 symbolic adapter."""
    cfg = dict(fusion_cfg or {})
    kind = cfg.pop("type", "gated")
    if kind == "gated":
        return ResidualFusion(d, sym_dim, cfg)
    if kind == "cross_attention":
        allowed = {"num_heads", "gate_init_bias", "attention_dropout", "token_dim"}
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError(f"unknown cross_attention fusion settings: {unknown}")
        return SymbolicCrossAttentionFusion(d, sym_dim, **cfg)
    if kind == "horizon_cross_attention_residual":
        if patch_len is None:
            raise ValueError("horizon_cross_attention_residual is valid only for the UniTS forecast head")
        allowed = {"num_heads", "gate_init_bias", "attention_dropout", "token_dim"}
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError(f"unknown horizon_cross_attention_residual settings: {unknown}")
        return HorizonSymbolicResidualFusion(d, sym_dim, patch_len=patch_len, **cfg)
    if kind == "horizon_prehead_cross_attention":
        if patch_len is None:
            raise ValueError("horizon_prehead_cross_attention is valid only for the UniTS forecast head")
        allowed = {"num_heads", "gate_init_bias", "attention_dropout", "token_dim"}
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError(f"unknown horizon_prehead_cross_attention settings: {unknown}")
        return HorizonPreHeadSymbolicFusion(d, sym_dim, patch_len=patch_len, **cfg)
    if kind == "horizon_location_prehead_cross_attention":
        if patch_len is None:
            raise ValueError("horizon_location_prehead_cross_attention is valid only for the UniTS forecast head")
        allowed = {"num_heads", "gate_init_bias", "attention_dropout", "token_dim"}
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError(f"unknown horizon_location_prehead_cross_attention settings: {unknown}")
        return HorizonLocationPreHeadSymbolicFusion(d, sym_dim, patch_len=patch_len, **cfg)
    raise ValueError("unknown fusion type {!r}; expected 'gated', 'cross_attention', or "
                     "'horizon_cross_attention_residual', or 'horizon_prehead_cross_attention'".format(kind))


class _FusionState:
    """Per-call context set by the wrapper: which task is active and its Path B input."""

    def __init__(self):
        self.task_key, self.sym, self.force_gate, self.last_gate, self.last_pooled = None, None, None, None, None
        self.forecast_horizons = {}


class FusedCLSHead(nn.Module):
    def __init__(self, stock, state: _FusionState):
        super().__init__()
        self.stock, self.state, self.adapters = stock, state, nn.ModuleDict()

    def forward(self, x, category_token=None, return_feature=False):
        feat = self.stock(x, return_feature=True)  # stock CLS feature [B, V, 1, d]
        st = self.state
        st.last_pooled = feat
        if st.sym is not None and st.task_key in self.adapters:
            feat, st.last_gate = self.adapters[st.task_key](feat, st.sym[:, :, None, :], st.force_gate)
        if return_feature:
            return feat
        B, V, _, C = feat.shape  # identical to CLSHead.forward after the feature
        m = category_token.shape[2]
        feat = feat.expand(B, V, m, C)
        return torch.einsum("nvkc,nvmc->nvm", feat, category_token).mean(dim=1)


class FusedForecastHead(nn.Module):
    def __init__(self, stock, state: _FusionState):
        super().__init__()
        self.stock, self.state, self.adapters = stock, state, nn.ModuleDict()

    def forward(self, x_full, pred_len, token_len):
        st = self.state
        st.last_pooled = x_full
        if st.sym is not None and st.task_key in self.adapters:
            adapter = self.adapters[st.task_key]
            if isinstance(adapter, HorizonSymbolicResidualFusion):
                # V5 preserves the upstream UniTS decoder and corrects only its
                # final per-horizon output with symbolic GEN-token retrieval.
                base = self.stock(x_full, pred_len, token_len)
                horizon = st.forecast_horizons[st.task_key]
                delta, st.last_gate = adapter(x_full, st.sym, horizon, st.force_gate)
                # Stock ForecastHead returns [B, context+H, V]; UniTS slices the
                # final H rows after this wrapper returns. Leave its context rows
                # exactly unchanged and add symbolic corrections only to future H.
                full_delta = torch.zeros_like(base)
                full_delta[:, -horizon:, :] = delta.permute(0, 2, 1)
                return base + full_delta
            if isinstance(adapter, (HorizonPreHeadSymbolicFusion, HorizonLocationPreHeadSymbolicFusion)):
                horizon = st.forecast_horizons[st.task_key]
                x_full, st.last_gate = adapter(x_full, st.sym, horizon, st.force_gate)
                return self.stock(x_full, pred_len, token_len)
            s = st.sym[:, :, None, :].expand(-1, -1, x_full.shape[2], -1)
            x_full, st.last_gate = adapter(x_full, s, st.force_gate)
        return self.stock(x_full, pred_len, token_len)


# =========================================================================== wrapper
class UniTSDualPath(nn.Module):
    """Path A = stock UniTS (single task per cell); Path B arrives precomputed as ``sym`` [B, V, F]."""

    def __init__(self, *, task: str, dataset: str, n_channels: int, seq_len: int, out_dim: int,
                 sym_dim: int = 0, fusion_cfg: dict | None = None, repo_dir="third_party/UniTS",
                 checkpoint=None, seed: int = 0, extra_tasks: list | None = None, debug_random: bool = False,
                 partial_unfreeze_blocks: int = 1, partial_unfreeze_block_norms: bool = True):
        super().__init__()
        torch.manual_seed(seed)
        repo_dir, ckpt = ensure_units(repo_dir, None if debug_random else checkpoint)
        mod = load_units_module(repo_dir)
        if debug_random:
            log.warning("Using a randomly initialised test-only UniTS")
            args, sd = argparse.Namespace(**DEBUG_ARGS), None
        else:
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            a = raw["args"]
            args = argparse.Namespace(**{k: getattr(a, k) for k in DEBUG_ARGS})
            sd = {k.removeprefix("module."): v for k, v in raw["student"].items() if "cls_prompts" not in k}
        self.regime = None
        self.task, self.dataset, self.n_channels, self.seq_len, self.out_dim = task, dataset, n_channels, seq_len, out_dim
        self.d_model = args.d_model
        self.partial_unfreeze_blocks = max(0, int(partial_unfreeze_blocks))
        self.partial_unfreeze_block_norms = bool(partial_unfreeze_block_norms)
        primary = task_config(task, dataset, n_channels, seq_len, out_dim)
        configs = [primary] + [task_config(*t) for t in (extra_tasks or [])]
        self.task_ids = {c[0]: i for i, c in enumerate(configs)}
        self.task_key = primary[0]
        self.model = mod.Model(args, configs, pretrain=False)
        self.loaded_tokens = []
        if sd is not None:
            self._load_verified(sd, ckpt)
        self.state = _FusionState()
        self.state.forecast_horizons = {
            key: int(task_cfg["pred_len"])
            for key, task_cfg in configs
            if task_cfg["task_name"] in ("long_term_forecast", "short_term_forecast")
        }
        self.model.cls_head = FusedCLSHead(self.model.cls_head, self.state)
        self.model.forecast_head = FusedForecastHead(self.model.forecast_head, self.state)
        if sym_dim:
            self.add_fusion(self.task_key, sym_dim, fusion_cfg)
        self.set_regime("R1")

    def _load_verified(self, sd, ckpt):
        own = self.model.state_dict()
        missing_core = [k for k in own if k.startswith(FROZEN_PREFIXES + ("cls_head.", "forecast_head."))
                        and k not in sd]
        if missing_core:
            raise RuntimeError(f"UniTS checkpoint {ckpt} lacks backbone tensors: {missing_core[:5]}")
        loadable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        self.model.load_state_dict(loadable, strict=False)
        now = self.model.state_dict()
        bad = [k for k, v in loadable.items() if not torch.equal(now[k], v)]
        if bad:
            raise RuntimeError(f"UniTS weights not loaded correctly: {bad[:5]}")
        self.loaded_tokens = sorted(k for k in loadable if "tokens" in k)
        log.info("Verified %d pretrained UniTS tensors (%d task tokens reused: %s)", len(loadable),
                 len(self.loaded_tokens), self.loaded_tokens)

    def add_fusion(self, task_key: str, sym_dim: int, fusion_cfg: dict | None = None):
        """Install a late-fusion adapter for one task (the Phase 2 'fusion hook')."""
        head = self.model.cls_head if task_key.startswith("CLS_") else self.model.forecast_head
        kind = (fusion_cfg or {}).get("type", "gated")
        if kind in {"horizon_cross_attention_residual", "horizon_prehead_cross_attention",
                    "horizon_location_prehead_cross_attention"} and not task_key.startswith("LTF_"):
            raise ValueError(f"{kind} is forecasting-only")
        patch_len = head.stock.patch_len if isinstance(head, FusedForecastHead) else None
        head.adapters[task_key] = build_fusion_adapter(self.d_model, sym_dim, fusion_cfg, patch_len=patch_len)
        if self.regime:
            self.set_regime(self.regime)

    # ------------------------------------------------------------------ regimes
    def _is_new_param(self, name: str) -> bool:
        return ".adapters." in name

    def _is_task_param(self, name: str) -> bool:
        """This cell's own task parameters: its dataset/task tokens and its task head."""
        ds, key = self.dataset, self.task_key
        if name in (f"prompt_tokens.{ds}", f"mask_tokens.{ds}", f"cls_tokens.{key}", f"category_tokens.{key}"):
            return True
        head = "cls_head.stock." if self.task == "classification" else "forecast_head.stock."
        return name.startswith(head)

    def _is_partial_backbone_param(self, name: str) -> bool:
        """Whether a frozen-core parameter is trainable under R1.5.

        UniTS stores its encoder as ``blocks.<index>``.  We determine the last
        index from the loaded model rather than assuming a fixed layer count.
        """
        if self.partial_unfreeze_blocks < 1 or not name.startswith("blocks."):
            return False
        parts = name.split(".", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            return False
        n_blocks = len(self.model.blocks)
        index = int(parts[1])
        if index >= n_blocks - self.partial_unfreeze_blocks:
            return True
        return self.partial_unfreeze_block_norms and ".norm" in name.lower()

    def set_regime(self, regime: str):
        """R1 frozen, R1.5 final-block tuning, R2 full UniTS fine-tuning."""
        if regime not in ("R1", "R1.5", "R2"):
            raise ValueError(regime)
        self.regime = regime
        for n, p in self.model.named_parameters():
            p.requires_grad = (regime == "R2" or self._is_new_param(n) or self._is_task_param(n)
                               or (regime == "R1.5" and self._is_partial_backbone_param(n)))
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.regime in ("R1", "R1.5"):
            for n, mod in self.model.named_children():
                if n in ("patch_embeddings", "position_embedding", "prompt2forecat"):
                    mod.eval()
            if self.regime == "R1":
                self.model.blocks.eval()
            else:
                for i, block in enumerate(self.model.blocks):
                    if i < len(self.model.blocks) - self.partial_unfreeze_blocks:
                        block.eval()
        return self

    def param_groups(self, lr_new: float, lr_backbone: float):
        new, task, old = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (new if self._is_new_param(n) else task if not n.startswith(FROZEN_PREFIXES) else old).append(p)
        groups = [{"params": new + task, "lr": lr_new, "name": "new"}]
        if old:
            groups.append({"params": old, "lr": lr_backbone, "name": "backbone"})
        return groups

    def trainable_state_dict(self):
        if self.regime in ("R1.5", "R2"):
            return self.state_dict()
        return {k: v for k, v in self.state_dict().items() if not k.startswith(tuple("model." + p for p in FROZEN_PREFIXES))}

    # ------------------------------------------------------------------ io
    sym_per_channel = True  # UniTS fuses per variable, also for classification

    def batch_inputs(self, split, idx):
        """Stock UniTS input: the raw series (UniTS pads to the patch multiple itself)."""
        x = np.asarray(split.x_raw(idx), dtype=np.float32)
        return x, np.ones((len(x), x.shape[-1]), dtype=np.float32)

    def forward(self, x, input_mask=None, sym=None, force_gate=None, task_key=None):
        """x [B, V, L] (project convention) -> out: logits [B, m] | forecast [B, V, H] | recon [B, V, L]."""
        key = task_key or self.task_key
        st = self.state
        st.task_key, st.sym, st.force_gate, st.last_gate = key, sym, force_gate, None
        try:
            name = [c for c in self.model.configs_list if c[0] == key][0][1]["task_name"]
            out = self.model(x.permute(0, 2, 1), None, task_id=self.task_ids[key], task_name=name)
        finally:
            st.sym, st.force_gate = None, None
        if name != "classification":
            out = out.permute(0, 2, 1)
        return {"out": out, "gate": st.last_gate, "pooled": st.last_pooled}
