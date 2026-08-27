"""GLM-5.3-Flash (``glm5_next``) hyperparameters.

GLM-5.3-Flash is a hybrid of three lineages FreeToken already serves:

* the sparse-attention layers are GLM-5.2's MLA + DSA (``glm_moe_dsa``) with two
  twists -- NoPE (``qk_rope_head_dim == 0``, no rope anywhere in the sparse path,
  indexer included) and a k-pool-compressed indexer (pools of ``index_kpool``
  tokens are scored instead of individual tokens, winners expand back to raw
  token indices, the incomplete tail pool is always selected);
* the linear-attention layers are Kimi KDA (per-channel decay gated delta rule
  with a safe sigmoid lower-bound gate), served by the vendored fla KDA kernels;
* the residual scheme is DeepSeek-V4's manifold-constrained Hyper-Connections
  (mHC), whose Triton kernels live under ``kernel/triton/dsv4``.

This payload extends the GLM-5.2 args with the KDA / k-pool / mHC dims and rides
``ModelConfig.glm_dsa_args`` (the GLM MLA/DSA payload slot -- the ``dsa`` backend
and the pool factory read the shared fields through the same names).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from freetoken.models.glm_moe_dsa.args import GlmMoeDsaArgs


@dataclass(frozen=True)
class Glm53Args(GlmMoeDsaArgs):
    # DSA k-pool compression (indexer scores pooled keys, not tokens)
    index_kpool: int = 1
    index_kpool_always_select_tail: bool = True
    # KDA linear attention
    linear_num_heads: int = 0
    linear_head_dim: int = 0
    linear_conv_kernel: int = 0
    linear_lower_bound: float | None = None
    # per-layer taxonomy from the checkpoint ("linear_attention" / "deepseek_sparse_attention")
    layer_types: Tuple[str, ...] = ()
    # manifold-constrained Hyper-Connections
    hc_mult: int = 0
    hc_sinkhorn_iters: int = 0
    hc_eps: float = 1e-6

    @property
    def kda_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def sparse_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t != "linear_attention")


def load_args(hf_config: Any) -> Glm53Args:
    text = getattr(hf_config, "text_config", hf_config)
    lin = getattr(text, "linear_attn_config", None) or {}
    get = lin.get if isinstance(lin, dict) else (lambda k, d=None: getattr(lin, k, d))
    layer_types = tuple(getattr(text, "layer_types", ()) or ())
    assert layer_types, "GLM-5.3 config must carry layer_types"
    # NoPE: no rope in the sparse path (main attention or indexer); rope fields are
    # kept only because the parent dataclass carries them (rotary tables are never built).
    assert int(getattr(text, "qk_rope_head_dim", 0)) == 0, "GLM-5.3-Flash is NoPE"
    assert bool(getattr(text, "index_kpool_always_select_tail", True)), (
        "only the always-select-tail k-pool variant is supported (the checkpoint sets it)"
    )
    return Glm53Args(
        hidden_size=text.hidden_size,
        num_heads=text.num_attention_heads,
        q_lora_rank=text.q_lora_rank,
        kv_lora_rank=text.kv_lora_rank,
        qk_nope_head_dim=text.qk_nope_head_dim,
        qk_rope_head_dim=int(getattr(text, "qk_rope_head_dim", 0)),
        v_head_dim=text.v_head_dim,
        norm_eps=text.rms_norm_eps,
        rope_theta=float(getattr(text, "rope_theta", 10000.0) or 10000.0),
        rope_interleave=True,
        indexer_rope_interleave=bool(getattr(text, "indexer_rope_interleave", True)),
        max_position=text.max_position_embeddings,
        index_n_heads=int(getattr(text, "index_n_heads", 0)),
        index_head_dim=int(getattr(text, "index_head_dim", 0)),
        index_topk=int(getattr(text, "index_topk", 0)),
        # The checkpoint declares indexer_types "full" for EVERY layer, but only the
        # sparse layers own an indexer -- KDA layers never reach the dsa backend.
        # Recode KDA layers as "linear" so the backend's slot walk skips them (its
        # slot count must match the pool's num_index_layers).
        indexer_types=tuple(
            "linear" if t == "linear_attention" else "full" for t in layer_types
        ),
        index_kpool=int(getattr(text, "index_kpool", 1) or 1),
        index_kpool_always_select_tail=bool(
            getattr(text, "index_kpool_always_select_tail", True)
        ),
        linear_num_heads=int(get("num_heads", 0) or 0),
        linear_head_dim=int(get("head_dim", 0) or 0),
        linear_conv_kernel=int(get("short_conv_kernel_size", 4) or 4),
        linear_lower_bound=(
            float(get("gate_lower_bound")) if get("gate_lower_bound") is not None else None
        ),
        layer_types=layer_types,
        hc_mult=int(getattr(text, "hc_mult", 0) or 0),
        hc_sinkhorn_iters=int(getattr(text, "hc_sinkhorn_iters", 0) or 0),
        hc_eps=float(getattr(text, "hc_eps", 1e-6) or 1e-6),
    )


__all__ = ["Glm53Args", "load_args"]
