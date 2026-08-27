"""SwiGLU MLP for GLM-5.3-Flash's leading dense layers and per-layer shared experts.

The FP8 checkpoint stores these as block-fp8 (128x128 ``weight_scale_inv``), served
natively through ``Fp8BlockLinear`` via the shared quant-linear factory; a bf16
checkpoint falls back to plain replicated linears through the same factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from freetoken.layers import BaseOP
from freetoken.models.quant_linear import make_replicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig


class Glm53GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig, hidden_size: int, intermediate_size: int):
        self.gate_proj = make_replicated(config, hidden_size, intermediate_size)
        self.up_proj = make_replicated(config, hidden_size, intermediate_size)
        self.down_proj = make_replicated(config, intermediate_size, hidden_size)

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        return self.down_proj.forward(F.silu(gate) * up)


__all__ = ["Glm53GatedMLP"]
