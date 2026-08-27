"""GLM-5.3-Flash sparse-attention layer: NoPE MLA + k-pool DSA indexer.

Same MLA weight-absorption serving shape as GLM-5.2 (``glm_moe_dsa/attention.py``)
with two deltas, both faithful to the HF reference ``modeling_glm5_next``:

* **NoPE** -- ``qk_rope_head_dim == 0``. No rotary anywhere in the sparse path
  (main attention or indexer); position information comes from the KDA layers and
  causal masking. The latent row is just ``c_kv`` (kv_lora_rank wide).
* **k-pool indexer** -- the lightning indexer scores POOLS of ``index_kpool``
  tokens: each pool's key is a channel-wise softmax(gate + ape)-weighted mean of
  its member keys. This module owns the projections (``wq_b``/``wk``/``k_norm``/
  ``weights_proj``) plus the compression params (``index_kpool_compress_gate`` /
  ``index_kpool_compress_ape``); scoring, top-k pool selection, expansion back to
  token rows and the tail append live in the ``dsa`` backend.

All 45 checkpoint layers declare an indexer type but only the 11 sparse layers own
one -- there is no IndexShare sharing in GLM-5.3 (every sparse layer is "full").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.models.glm_moe_dsa.attention import _IdxLayerNorm
from freetoken.models.quant_linear import make_replicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Glm53Indexer(BaseOP):
    """k-pool DSA lightning indexer (bf16 in every quant mode, like GLM-5.2's --
    the projections are small and the top-k boundary is precision-sensitive)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm_dsa_args
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.kpool = args.index_kpool
        self.wq_b = LinearReplicated(args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False)
        self.wk = LinearReplicated(args.hidden_size, self.head_dim, has_bias=False)
        self.k_norm = _IdxLayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = LinearReplicated(args.hidden_size, self.n_heads, has_bias=False)
        # k-pool compression: per-token gate scores + a per-pool-position additive
        # prior (ape), combined channel-wise in the backend's pooled-key softmax.
        self.index_kpool_compress_gate = torch.empty(self.head_dim, args.hidden_size)
        self.index_kpool_compress_ape = torch.empty(self.kpool, self.head_dim)

    def compute(
        self, x: torch.Tensor, q_resid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token indexer projections (NoPE: no rotary): ``(q [T, H, D], k [T, D],
        w [T, H] fp32, gate [T, D], ape [kpool, D])``."""
        t = x.shape[0]
        q = self.wq_b.forward(q_resid).view(t, self.n_heads, self.head_dim)
        k = self.k_norm.forward(self.wk.forward(x)).view(t, self.head_dim)
        w = self.weights_proj.forward(x).float() * (self.n_heads**-0.5)
        gate = torch.nn.functional.linear(x, self.index_kpool_compress_gate.to(x.dtype))
        return q, k, w, gate, self.index_kpool_compress_ape


class Glm53Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm_dsa_args
        self.layer_id = layer_id
        self.indexer = (
            Glm53Indexer(config, layer_id)
            if args.indexer_types and args.indexer_types[layer_id] == "full"
            else None
        )
        self.num_heads = args.num_heads
        self.qk_head_dim = args.qk_head_dim  # == qk_nope_head_dim (NoPE)
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank

        # block-fp8 checkpoint: the big projections are native fp8 (Fp8BlockLinear);
        # kv_b stays bf16 (consumed as bmm operands by the MLA absorption).
        self.q_a_proj = make_replicated(config, args.hidden_size, args.q_lora_rank)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        self.q_b_proj = make_replicated(
            config, args.q_lora_rank, self.num_heads * self.qk_head_dim
        )
        # NoPE: kv_a projects to the bare latent (no rope tail).
        self.kv_a_proj_with_mqa = make_replicated(config, args.hidden_size, self.kv_lora_rank)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        # kv_b stays bf16 in every mode: consumed as bmm operands by the MLA
        # absorption below, not through a Linear forward (glm_moe_dsa precedent).
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank,
            self.num_heads * (args.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
        )
        self.o_proj = make_replicated(config, self.num_heads * self.v_head_dim, args.hidden_size)
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    # kv_b split, cached in bmm-ready bf16 layout (glm_moe_dsa's _kv_b, same math)
    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads, self.qk_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            self._w_uk = w[:, : self.qk_head_dim, :].contiguous()
            self._w_uv = w[:, self.qk_head_dim :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None  # checkpoint layout freed; repacked forms serve

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        t = x.shape[0]
        w_uk, w_uv = self._kv_b()

        q_a_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_a_resid)
        q_nope = q.view(t, self.num_heads, self.qk_head_dim)

        c_kv = self.kv_a_layernorm.forward(self.kv_a_proj_with_mqa.forward(x))

        # Absorb kv_b's k-part into the query: q_nope[H,T,nope] @ W_uk[H,nope,lora].
        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(0, 1)

        indexer_qkw = (
            self.indexer.compute(x, q_a_resid)
            if self.indexer is not None and getattr(ctx.attn_backend, "dsa_enabled", False)
            else None
        )

        # NoPE: the rope half is empty; the backend cats it into the (bare) latent.
        empty_pe = q_absorbed.new_empty(t, self.num_heads, 0)
        empty_rope = c_kv.new_empty(t, 0)
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(), empty_pe, c_kv.contiguous(), empty_rope,
            self.layer_id, ctx.batch, indexer_qkw=indexer_qkw,
        )  # [T, H, kv_lora_rank]

        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        return self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim))


__all__ = ["Glm53Attention", "Glm53Indexer"]
