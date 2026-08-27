"""Weight loading for GLM-5.3-Flash (``glm5_next``), block-fp8 checkpoint.

Checkpoint namespace is ``model.language_model.layers.N...`` (multimodal wrapper);
the vision tower and the trailing MTP layer (``layers.45`` + its ``eh_proj``/
``enorm``/``hnorm``/``shared_head``) are dropped -- FreeToken serves text-only,
no speculative head in v1.

Per layer kind:

* **KDA layers** (all-bf16 in the FP8 checkpoint): ``q/k/v_proj`` concatenate into
  the fused ``in_proj_qkv`` GEMM, ``q/k/v_conv1d`` into the depthwise ``conv1d``
  (order q|k|v, matching the fused split); the low-rank gates, ``A_log`` /
  ``dt_bias`` (fp32-exempt from the model-dtype downcast), ``o_norm`` and
  ``o_proj`` stream through verbatim.
* **Sparse layers**: the big projections are native block-fp8 (``.weight`` e4m3 +
  ``.weight_scale_inv`` bf16 pass through verbatim into the ``Fp8BlockLinear``
  buffers); ``kv_b_proj`` and the whole indexer (incl. the k-pool ``compress_gate``
  / ``compress_ape``) stay bf16.
* **mHC**: the flat ``hc_attn_*`` / ``hc_ffn_*`` layer keys map onto the
  ``attn_hc.*`` / ``ffn_hc.*`` module buffers (fp32).

Routed experts are block-fp8 and served from the offload cache;
``setup_offload_expert_banks`` reuses the qwen3_5_moe block-fp8 bank builder
outright -- the expert checkpoint keys and MoE dims match exactly.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import iter_weight_files

# The qwen3_5_moe block-fp8 expert-bank builder matches this checkpoint exactly:
# same ``model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj`` keys,
# same 128x128 ``weight_scale_inv`` layout, dims read off the shared ModelConfig.
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.models.loader import drop_page_cache
from freetoken.models.qwen3_5_moe.weight import (  # noqa: F401
    setup_offload_expert_banks,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")

# mHC checkpoint keys sit flat on the layer; the module owns them under two sites.
_HC_RENAMES = {
    ".hc_attn_fn": ".attn_hc.fn",
    ".hc_attn_base": ".attn_hc.base",
    ".hc_attn_scale": ".attn_hc.scale",
    ".hc_ffn_fn": ".ffn_hc.fn",
    ".hc_ffn_base": ".ffn_hc.base",
    ".hc_ffn_scale": ".ffn_hc.scale",
}

# KDA fusions: concat checkpoint parts (dim 0) in this exact order to match the
# fused module buffers. fused_suffix -> ordered part suffixes.
_KDA_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.in_proj_qkv.weight": (
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ),
    ".self_attn.conv1d.weight": (
        ".self_attn.q_conv1d.weight", ".self_attn.k_conv1d.weight", ".self_attn.v_conv1d.weight",
    ),
}

# MTP-only tensors that appear on the trailing layer.
_MTP_SUFFIXES = (".eh_proj.weight", ".enorm.weight", ".hnorm.weight", ".shared_head.norm.weight")


def _rename(raw_name: str, num_layers: int) -> str | None:
    """HF key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith(("model.visual.", "visual.", "mtp.")):
        return None
    name = raw_name
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model.") :]
    m = _LAYER_RE.match(name)
    if m and int(m.group(1)) >= num_layers:
        return None  # trailing MTP layer (and any dev-capped tail)
    if name.endswith(_MTP_SUFFIXES):
        return None
    for suffix, repl in _HC_RENAMES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)] + repl
    if name.endswith(".mlp.gate.e_score_correction_bias"):
        return name.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
    return name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    for fused_suffix, parts in _KDA_FUSIONS.items():
        for idx, part in enumerate(parts):
            if name.endswith(part):
                key = name[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return key, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3-Flash routed experts are quantized (block-fp8 or NVFP4) and only "
        "support the offload backend; they are loaded into the offload cache via "
        "setup_offload_expert_banks."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    num_layers = config.num_layers
    tp_info = get_tp_info()
    if tp_info.size > 1:
        raise NotImplementedError("glm_5_3 weight loading currently supports TP=1 only")

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading GLM-5.3 dense weights",
        disable=not tp_info.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if _EXPERT_RE.search(raw_name):
                    continue  # routed experts -> offload cache
                name = _rename(raw_name, num_layers)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                # e_score bias: HF keeps fp32; store model-dtype like glm_moe_dsa.
                if name.endswith(".mlp.e_score_correction_bias"):
                    tensor = tensor.to(torch.bfloat16)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue
                yield name, tensor
    assert not fuse_buf, f"Incomplete KDA fusions: {list(fuse_buf.keys())}"


# ======================================================================================
# NVFP4 routed experts (modelopt community quants, e.g. LibertAIDAI/GLM-5.3-Flash-NVFP4)
# ======================================================================================
# Experts-only quant: per-expert un-fused ``weight`` (packed uint8) + ``weight_scale``
# (fp8 block) + ``weight_scale_2`` (global), under the SAME key layout as the FP8
# checkpoint. Dense tensors are plain bf16 and ride iter_weights unchanged. The bank
# index is the MoE layer (global layer minus the dense prefix), matching how
# make_moe_layer addresses the offload cache.
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
def _nvfp4_layer_to_bank(layer: int, config) -> int | None:
    """Bank index = MoE layer (global minus the dense prefix). The trailing MTP
    layer (checkpoint layer 45) carries its own NVFP4 experts -- outside the
    served text stack, so it maps to None (skipped), not an error."""
    bank = layer - config.first_k_dense_replace
    if bank < 0 or bank >= config.num_moe_layers:
        return None
    return bank


_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_nvfp4_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts",
)


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    return load_nvfp4_expert_source_banks_parallel(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers, chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "iter_weights",
    "setup_offload_expert_banks",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
