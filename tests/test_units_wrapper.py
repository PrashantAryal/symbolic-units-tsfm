"""Phase 2: the UniTS wrapper must be stock UniTS at init / with gate 0, and a fusion
hook for one task must not move another task's outputs (cross-task non-interference).

Fast tests use a randomly initialised tiny UniTS (explicit ``debug_random``); ``slow``
tests use the real x128 pretrained checkpoint (RUN_SLOW=1).
"""
import os
from pathlib import Path

import pytest
import torch

from symtsfm.models.units_wrapper import UniTSDualPath, load_units_module

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "UniTS"
CKPT = Path(os.environ.get("UNITS_CKPT", ROOT.parent / "runs" / "checkpoints" / "units_x128_pretrain_checkpoint.pth"))
slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 (needs UniTS x128 checkpoint)")
SPECS = {"classification": dict(n_channels=2, seq_len=100, out_dim=3),
         "forecasting": dict(n_channels=2, seq_len=96, out_dim=24),
         "anomaly": dict(n_channels=1, seq_len=100, out_dim=0)}


def _make(task, sym_dim=0, debug=True, dataset="TOY", **kw):
    fusion_cfg = kw.pop("fusion_cfg", {"proj_dim": 5})
    m = UniTSDualPath(task=task, dataset=dataset, sym_dim=sym_dim, fusion_cfg=fusion_cfg, repo_dir=REPO,
                      checkpoint=CKPT, debug_random=debug, seed=0, **{**SPECS[task], **kw})
    return m.eval()


def _x(task, B=3, seed=0):
    s = SPECS[task]
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, s["n_channels"], s["seq_len"], generator=g).cumsum(-1)


def _stock(wrapper):
    """A plain upstream UniTS Model with the wrapper's weights (heads un-wrapped)."""
    mod = load_units_module(REPO)
    stock = mod.Model(_args(wrapper), wrapper.model.configs_list)
    sd = {k.replace("cls_head.stock.", "cls_head.").replace("forecast_head.stock.", "forecast_head."): v
          for k, v in wrapper.model.state_dict().items() if ".adapters." not in k}
    stock.load_state_dict(sd, strict=True)
    return stock.eval()


def _args(w):
    import argparse

    m = w.model
    return argparse.Namespace(d_model=w.d_model, e_layers=len(m.blocks), n_heads=m.blocks[0].seq_att_block.attn_seq.num_heads,
                              patch_len=m.patch_len, stride=m.stride, prompt_num=m.prompt_num, dropout=0.1)


def _stock_out(stock, wrapper, x):
    name = wrapper.model.configs_list[0][1]["task_name"]
    out = stock(x.permute(0, 2, 1), None, task_id=0, task_name=name)
    return out if name == "classification" else out.permute(0, 2, 1)


@pytest.mark.parametrize("task", ["classification", "forecasting", "anomaly"])
def test_baseline_and_guided_init_equal_stock_units(task):
    base, guided = _make(task), _make(task, sym_dim=7)
    stock = _stock(base)
    x = _x(task)
    ref = _stock_out(stock, base, x)
    torch.testing.assert_close(base(x)["out"], ref)
    guided.model.load_state_dict(base.model.state_dict(), strict=False)  # same UniTS weights
    sym = torch.randn(3, SPECS[task]["n_channels"], 7)
    out = guided(x, sym=sym)
    torch.testing.assert_close(out["out"], ref)  # W_s zero-initialised -> stock output
    assert out["gate"] is not None


@pytest.mark.parametrize("task", ["classification", "forecasting", "anomaly"])
def test_gate_zero_recovers_baseline_with_trained_adapter(task):
    base, guided = _make(task), _make(task, sym_dim=7)
    guided.model.load_state_dict(base.model.state_dict(), strict=False)
    head = guided.model.cls_head if task == "classification" else guided.model.forecast_head
    with torch.no_grad():
        next(iter(head.adapters.values())).out.weight.normal_()
    x, sym = _x(task), torch.randn(3, SPECS[task]["n_channels"], 7)
    torch.testing.assert_close(guided(x, sym=sym, force_gate=0.0)["out"], base(x)["out"])
    assert not torch.allclose(guided(x, sym=sym)["out"], base(x)["out"])


@pytest.mark.parametrize("task", ["classification", "forecasting", "anomaly"])
def test_cross_attention_init_and_zero_gate_recover_baseline(task):
    cfg = {"type": "cross_attention", "num_heads": 4, "gate_init_bias": 0.0}
    base, guided = _make(task), _make(task, sym_dim=7, fusion_cfg=cfg)
    guided.model.load_state_dict(base.model.state_dict(), strict=False)
    x, sym = _x(task), torch.randn(3, SPECS[task]["n_channels"], 7)
    torch.testing.assert_close(guided(x, sym=sym)["out"], base(x)["out"])
    head = guided.model.cls_head if task == "classification" else guided.model.forecast_head
    adapter = next(iter(head.adapters.values()))
    with torch.no_grad():
        adapter.out.weight.normal_()
    torch.testing.assert_close(guided(x, sym=sym, force_gate=0.0)["out"], base(x)["out"])
    assert not torch.allclose(guided(x, sym=sym)["out"], base(x)["out"])


def test_horizon_cross_attention_residual_init_and_zero_gate_recover_baseline():
    """V5 touches forecast output patches only and is stock UniTS at initialization."""
    cfg = {"type": "horizon_cross_attention_residual", "num_heads": 4, "gate_init_bias": -2.0}
    base, guided = _make("forecasting"), _make("forecasting", sym_dim=7, fusion_cfg=cfg)
    guided.model.load_state_dict(base.model.state_dict(), strict=False)
    x, sym = _x("forecasting"), torch.randn(3, SPECS["forecasting"]["n_channels"], 7)
    torch.testing.assert_close(guided(x, sym=sym)["out"], base(x)["out"])
    adapter = next(iter(guided.model.forecast_head.adapters.values()))
    with torch.no_grad():
        adapter.residual_out.weight.normal_()
        adapter.residual_out.bias.normal_()
    torch.testing.assert_close(guided(x, sym=sym, force_gate=0.0)["out"], base(x)["out"])
    assert not torch.allclose(guided(x, sym=sym)["out"], base(x)["out"])


def test_horizon_prehead_cross_attention_init_and_zero_gate_recover_baseline():
    """V10 changes only GEN-token states before the stock forecast head."""
    cfg = {"type": "horizon_prehead_cross_attention", "num_heads": 4, "gate_init_bias": -2.0}
    base, guided = _make("forecasting"), _make("forecasting", sym_dim=7, fusion_cfg=cfg)
    guided.model.load_state_dict(base.model.state_dict(), strict=False)
    x, sym = _x("forecasting"), torch.randn(3, SPECS["forecasting"]["n_channels"], 7)
    torch.testing.assert_close(guided(x, sym=sym)["out"], base(x)["out"])
    adapter = next(iter(guided.model.forecast_head.adapters.values()))
    with torch.no_grad():
        adapter.inner.out.weight.normal_()
    torch.testing.assert_close(guided(x, sym=sym, force_gate=0.0)["out"], base(x)["out"])
    assert not torch.allclose(guided(x, sym=sym)["out"], base(x)["out"])


def test_horizon_location_prehead_cross_attention_init_and_zero_gate_recover_baseline():
    """V11 retains K match locations while preserving the zero-gate baseline."""
    cfg = {"type": "horizon_location_prehead_cross_attention", "num_heads": 4, "gate_init_bias": -3.0}
    # [presence, frequency, distance] * K + rarity + locations * K; K=2.
    base, guided = _make("forecasting"), _make("forecasting", sym_dim=9, fusion_cfg=cfg)
    guided.model.load_state_dict(base.model.state_dict(), strict=False)
    x, sym = _x("forecasting"), torch.randn(3, SPECS["forecasting"]["n_channels"], 9)
    torch.testing.assert_close(guided(x, sym=sym)["out"], base(x)["out"])
    adapter = next(iter(guided.model.forecast_head.adapters.values()))
    with torch.no_grad():
        adapter.inner.out.weight.normal_()
    torch.testing.assert_close(guided(x, sym=sym, force_gate=0.0)["out"], base(x)["out"])
    assert not torch.allclose(guided(x, sym=sym)["out"], base(x)["out"])


def test_r1_trains_only_task_tokens_head_and_fusion():
    m = _make("classification", sym_dim=4)
    names = {n for n, p in m.model.named_parameters() if p.requires_grad}
    assert names and all(n.startswith(("cls_head.", "prompt_tokens.TOY", "mask_tokens.TOY", "cls_tokens.CLS_TOY",
                                       "category_tokens.CLS_TOY")) for n in names)
    assert any(".adapters." in n for n in names)
    assert not any(n.startswith(("blocks.", "patch_embeddings.", "position_embedding.")) for n in names)
    m.set_regime("R2")
    assert all(p.requires_grad for p in m.parameters())


def test_r15_unfreezes_only_last_encoder_block_in_addition_to_r1_parameters():
    m = _make("forecasting", sym_dim=4, partial_unfreeze_blocks=1, partial_unfreeze_block_norms=False)
    m.set_regime("R1.5")
    names = {n for n, p in m.model.named_parameters() if p.requires_grad}
    last = len(m.model.blocks) - 1
    assert any(n.startswith(f"blocks.{last}.") for n in names)
    assert not any(n.startswith(f"blocks.{i}.") for i in range(last) for n in names)
    assert any(".adapters." in n for n in names)


def _non_interference(debug):
    """Classification eval before vs after adding (and training) a forecasting fusion hook."""
    extra = [("forecasting", "ETTh1", 7, 96, 96)]
    kw = dict(debug=debug, dataset="FordB", n_channels=1, seq_len=500, out_dim=2, extra_tasks=extra)
    m = _make("classification", **kw)
    xc = torch.randn(4, 1, 500).cumsum(-1)
    before = m(xc)["out"].detach().clone()
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    fkey = "LTF_ETTh1_p96"
    m.add_fusion(fkey, sym_dim=6, fusion_cfg={"proj_dim": 5})
    # train the forecasting fusion hook + forecast head (R1: backbone frozen) for a few steps
    for n, p in m.model.named_parameters():
        p.requires_grad = n.startswith("forecast_head.")
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m.train()
    xf, sym = torch.randn(4, 7, 96).cumsum(-1), torch.randn(4, 7, 6)
    for _ in range(3):
        loss = m(xf, sym=sym, task_key=fkey)["out"].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    # 1. only forecasting-side parameters moved; nothing classification reads changed
    changed = [k for k, v in m.state_dict().items() if k in sd0 and not torch.equal(v, sd0[k])]
    assert changed and all(k.startswith("model.forecast_head.") for k in changed), changed
    # 2. a fresh model with the post-training weights reproduces the classification logits bit-for-bit
    fresh = _make("classification", **kw)
    fresh.add_fusion(fkey, sym_dim=6, fusion_cfg={"proj_dim": 5})
    fresh.load_state_dict(m.state_dict())
    torch.testing.assert_close(fresh.eval()(xc)["out"], before, rtol=0, atol=0)
    # 3. the live model agrees too (up to CPU kernel-level float noise after a backward pass)
    torch.testing.assert_close(m(xc)["out"], before, rtol=0, atol=1e-6)
    return m


def test_cross_task_non_interference():
    _non_interference(debug=True)


@slow
def test_real_checkpoint_loads_verified_and_reuses_pretrained_tokens():
    m = _make("forecasting", debug=False, dataset="ETTh1", n_channels=7, seq_len=96, out_dim=96)
    assert "prompt_tokens.ETTh1" in m.loaded_tokens and "mask_tokens.ETTh1" in m.loaded_tokens
    x = torch.randn(2, 7, 96).cumsum(-1)
    torch.testing.assert_close(m(x)["out"], _stock_out(_stock(m), m, x))


@slow
def test_real_checkpoint_cross_task_non_interference():
    m = _non_interference(debug=False)
    assert "prompt_tokens.FordB" in m.loaded_tokens
