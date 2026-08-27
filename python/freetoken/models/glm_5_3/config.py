"""Engine-facing config for GLM-5.3-Flash (``glm5_next``).

Two attention groups drive the engine (the pool factory, the KV cost model and the
backend capability gate all read the same specs):

* ``full`` -- the 11 sparse-MLA layers. NoPE latent-KV (``head_dim`` == the bare
  ``kv_lora_rank``; no rope tail) with DSA index dims plus the k-pool gate slab
  (``index_gate_dim``), so the pool grows a gate-score slab beside the index keys.
  ``mla=True`` + index dims -> AttnType.DSA -> the ``dsa`` backend, which handles
  pool-compressed selection when ``glm_dsa_args.index_kpool > 1``.
* ``linear`` -- the 34 KDA layers, on the standard ``LinearGatedDeltaGroupConfig``
  (KDA's conv + [K, V] recurrent state has exactly the GDN pool geometry).

The FP8 checkpoint is DeepSeek-style 128x128 block-fp8 over the sparse-attention
projections, dense/shared MLPs and the routed experts (KDA projections, ``kv_b``,
the indexer and every mHC/norm/embedding tensor stay bf16); ``expert_quant`` is
``fp8_block`` and the dense projections ride the shared ``Fp8BlockLinear`` path.
Serving is text-only (the vision tower is dropped), matching the repo contract.
"""

from __future__ import annotations

import os
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import load_args


def _fp8_block(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return "none", None
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or get("quant_algo") or "").lower()
    block = get("weight_block_size")
    if method == "fp8" and block:
        bs = tuple(int(x) for x in block)
        assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
        return "fp8_block", bs
    return "none", None


def _dsa_on(args, num_layers: int) -> bool:
    """DSA serving switch, resolved ONCE here into the attention-group spec (same
    contract as glm_moe_dsa: pool factory / cost model / backend read the spec)."""
    return (
        args.index_topk > 0
        and args.index_head_dim > 0
        and os.getenv("FREETOKEN_GLM_DSA", "1") != "0"
    )


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)
    args = load_args(hf_config)
    # NoPE latent: the paged pool row is the bare c_kv (no rope tail).
    latent_dim = args.kv_lora_rank + args.qk_rope_head_dim  # 512 + 0

    num_layers = text.num_hidden_layers
    # Dev/testing only: cap the layer count so the forward path / KV / offload cache
    # can be exercised without the full expert set (glm_moe_dsa precedent).
    _cap = os.environ.get("FREETOKEN_GLM_DSA_MAX_LAYERS")
    if _cap:
        num_layers = min(num_layers, int(_cap))

    sparse_ids = tuple(i for i in args.sparse_layer_ids if i < num_layers)
    kda_ids = tuple(i for i in args.kda_layer_ids if i < num_layers)
    dsa = _dsa_on(args, num_layers)
    kpool = args.index_kpool if dsa else 0

    # NoPE: rotary_dim 0 -- no rope table is ever built; the field only sizes specs.
    rotary_config = RotaryConfig(
        head_dim=args.qk_head_dim,
        rotary_dim=0,
        max_position=args.max_position,
        base=args.rope_theta,
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=sparse_ids,
        num_kv_heads=1,  # single shared MLA latent
        head_dim=latent_dim,
        rotary_config=rotary_config,
        mla=True,
        index_head_dim=args.index_head_dim if dsa else 0,
        num_index_layers=len(sparse_ids) if dsa else 0,
        # k-pool gate-score slab beside the index keys (recomputed pooled keys).
        index_gate_dim=args.index_head_dim if (dsa and kpool > 1) else 0,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=kda_ids,
        num_key_heads=args.linear_num_heads,
        num_value_heads=args.linear_num_heads,
        key_head_dim=args.linear_head_dim,
        value_head_dim=args.linear_head_dim,
        conv_kernel_dim=args.linear_conv_kernel,
        output_gate=True,
    )
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    expert_quant, weight_block_size = _fp8_block(hf_config)

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,
        head_dim=latent_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=text.intermediate_size,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        num_experts=(
            getattr(text, "n_routed_experts", None) or getattr(text, "num_experts", 0)
        ),
        num_experts_per_tok=text.num_experts_per_tok,
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0)
        or text.intermediate_size,
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        model_type=getattr(text, "model_type", "glm5_next_text"),
        architectures=getattr(hf_config, "architectures", ["Glm5NextForConditionalGeneration"]),
        moe_enabled=True,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        first_k_dense_replace=int(getattr(text, "first_k_dense_replace", 0)),
        n_shared_experts=int(getattr(text, "n_shared_experts", 0)),
        routed_scaling_factor=float(getattr(text, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(text, "n_group", 1)),
        topk_group=int(getattr(text, "topk_group", 1)),
        attn_sm_scale=args.qk_head_dim**-0.5,
        has_attn_bias=bool(getattr(text, "attention_bias", False)),
        attention_groups=groups,
        glm_dsa_args=args,
    )


__all__ = ["parse_config"]
