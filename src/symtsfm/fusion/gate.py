"""Late-fusion gate used by the UniTS wrapper.

    p = Linear(z_b)                          Path B lightweight projection
    g = sigmoid(Linear([z_a ; p]))           scalar or per-feature sigmoid gate
    fused = [z_a ; g * p]                    fed to the task head

With g -> 0 the head sees ``[z_a ; 0]``, i.e. exactly the baseline head's input
(the extra head columns multiply zeros), so the symbolic path is cleanly
ablatable. ``g`` is returned and logged as an interpretability artefact.
"""
from __future__ import annotations

import torch
from torch import nn


class LateFusionGate(nn.Module):
    def __init__(self, d_a: int, d_b: int, proj_dim: int = 32, gate_init_bias: float = 0.0,
                 vector_gate: bool = False):
        super().__init__()
        self.d_a, self.d_b, self.proj_dim = d_a, d_b, proj_dim
        self.vector_gate = vector_gate
        self.proj = nn.Linear(d_b, proj_dim)
        self.gate = nn.Linear(d_a + proj_dim, proj_dim if vector_gate else 1)
        nn.init.zeros_(self.gate.weight)  # start as a constant gate sigmoid(bias)
        nn.init.constant_(self.gate.bias, gate_init_bias)

    @property
    def out_dim(self) -> int:
        return self.d_a + self.proj_dim

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, force_gate: float | None = None):
        """Return a fused vector and gate.

        With ``vector_gate=False`` (V1/V2), the gate has shape ``[..., 1]``:
        every projected symbolic feature is scaled together.  With
        ``vector_gate=True`` (V3), it has shape ``[..., proj_dim]`` and can
        retain or suppress each projected symbolic component independently.
        """
        p = self.proj(z_b)
        if force_gate is None:
            g = torch.sigmoid(self.gate(torch.cat([z_a, p], dim=-1)))
        else:  # ablation hook: pin the gate (e.g. 0.0 recovers the baseline head input)
            shape = p.shape if self.vector_gate else p.shape[:-1] + (1,)
            g = torch.full(shape, float(force_gate), device=p.device, dtype=p.dtype)
        return torch.cat([z_a, g * p], dim=-1), g


def gate_summary(g: torch.Tensor | None) -> dict:
    if g is None:
        return {"gate_mean": None, "gate_std": None, "gate_min": None, "gate_max": None}
    g = g.detach().float().flatten()
    return {
        "gate_mean": g.mean().item(),
        "gate_std": g.std().item() if g.numel() > 1 else 0.0,
        "gate_min": g.min().item(),
        "gate_max": g.max().item(),
    }
