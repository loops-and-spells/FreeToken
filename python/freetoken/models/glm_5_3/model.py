"""GLM-5.3-Flash model graph: mHC residual streams over a KDA / sparse-MLA hybrid.

Every decoder layer runs both sublayers inside DeepSeek-V4-style manifold-constrained
Hyper-Connections (``hc_mult`` residual streams; the shared Triton kernels live in
``kernel/triton/dsv4``): collapse the streams to one lane (``hc_pre``), run
norm -> mixer/MLP, then place the output back and mix the streams (``hc_post``).
The final collapse is an UNWEIGHTED mean (``Glm5NextTextHyperHead``) -- unlike
DeepSeek-V4 there is no learned head mix -- followed by the final norm.

The mixer is a KDA linear-attention op or a NoPE sparse-MLA op per
``layer_types``; the MLP is dense for the first ``first_k_dense_replace`` layers
and the sigmoid-routed MoE block after.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.hc import hc_pre_combine, hc_post_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Glm53Attention
from .kda import Glm53KDA
from .mlp import Glm53GatedMLP
from .moe import Glm53SparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _HyperConnection(BaseOP):
    """One mHC site (attention or ffn): owns ``fn``/``base``/``scale`` (checkpoint
    keys ``hc_attn_*`` / ``hc_ffn_*`` sit flat on the layer; weight.py maps them
    onto these buffers) and computes the collapse/placement weights."""

    def __init__(self, hc_mult: int, hidden_size: int, sinkhorn_iters: int,
                 eps: float, norm_eps: float):
        mix = (2 + hc_mult) * hc_mult
        self.fn = torch.empty(mix, hc_mult * hidden_size, dtype=torch.float32)
        self.base = torch.empty(mix, dtype=torch.float32)
        self.scale = torch.empty(3, dtype=torch.float32)
        self._hc = hc_mult
        self._dim = hidden_size
        self._iters = sinkhorn_iters
        self._eps = eps
        self._norm_eps = norm_eps

    def pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` [T, hc, D] -> (collapsed [T, D], post [T, hc], comb [T, hc, hc])."""
        t = x.shape[0]
        dtype = x.dtype
        xf = x.reshape(t, self._hc * self._dim).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self._norm_eps)
        mixes = F.linear(xf, self.fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale, self.base, self._hc, self._iters, self._eps
        )
        y = hc_pre_combine(xf.view(t, self._hc, self._dim), pre, dtype)
        return y, post, comb.view(t, self._hc, self._hc)

    def post(self, y: torch.Tensor, residual: torch.Tensor,
             post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """Sublayer output [T, D] + residual streams [T, hc, D] -> new streams."""
        return hc_post_combine(y, residual, post, comb)


class Glm53DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm_dsa_args
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.self_attn: BaseOP = Glm53KDA(
                hidden_size=config.hidden_size,
                num_heads=args.linear_num_heads,
                head_dim=args.linear_head_dim,
                conv_kernel_size=args.linear_conv_kernel,
                rms_norm_eps=config.rms_norm_eps,
                lower_bound=args.linear_lower_bound,
                layer_id=layer_id,
            )
        else:
            self.self_attn = Glm53Attention(config, layer_id)
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = Glm53SparseBlock(config, layer_id)
        else:
            self.mlp = Glm53GatedMLP(config, config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = _HyperConnection(
            args.hc_mult, config.hidden_size, args.hc_sinkhorn_iters,
            args.hc_eps, config.rms_norm_eps,
        )
        self.ffn_hc = _HyperConnection(
            args.hc_mult, config.hidden_size, args.hc_sinkhorn_iters,
            args.hc_eps, config.rms_norm_eps,
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x carries the hc_mult residual streams: [T, hc, D].
        residual = x
        y, post, comb = self.attn_hc.pre(x)
        y = self.input_layernorm.forward(y)
        y = self.self_attn.forward(y)
        x = self.attn_hc.post(y, residual, post, comb)

        residual = x
        y, post, comb = self.ffn_hc.pre(x)
        y = self.post_attention_layernorm.forward(y)
        y = self.mlp.forward(y)
        return self.ffn_hc.post(y, residual, post, comb)


class Glm53MTPLayer(BaseOP):
    """The trailing MTP draft layer (checkpoint layer ``num_layers``): DeepSeek-MTP
    wrappers around a plain (no-mHC) DSA decoder block with its own routed expert
    set (the trailing offload bank). Draft input pairs the main model's post-norm
    hidden at position ``i`` with the embedding of token ``i+1``; the output goes
    through ``shared_head_norm`` into the shared lm_head for the draft logits."""

    def __init__(self, config: ModelConfig, layer_id: int):
        from freetoken.models.quant_linear import make_replicated

        h = config.hidden_size
        self.enorm = RMSNorm(h, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(h, eps=config.rms_norm_eps)
        self.eh_proj = make_replicated(config, 2 * h, h)
        self.input_layernorm = RMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(h, eps=config.rms_norm_eps)
        self.self_attn = Glm53Attention(config, layer_id)
        self.mlp = Glm53SparseBlock(config, layer_id)
        self.shared_head_norm = RMSNorm(h, eps=config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.eh_proj.forward(
            torch.cat([self.enorm.forward(emb), self.hnorm.forward(hidden)], dim=-1)
        )
        y = self.input_layernorm.forward(x)
        x = x + self.self_attn.forward(y)
        y = self.post_attention_layernorm.forward(x)
        x = x + self.mlp.forward(y)
        return x  # raw block output; caller applies shared_head_norm for logits


class Glm53Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self._hc_mult = config.glm_dsa_args.hc_mult
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Glm53DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.mtp_layer_id is not None:
            self.mtp = Glm53MTPLayer(config, config.mtp_layer_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        # Expand into the hc residual streams (all streams start equal).
        x = x.unsqueeze(1).expand(-1, self._hc_mult, -1).contiguous()
        for layer in self.layers.op_list:
            x = layer.forward(x)
        # Final collapse is an unweighted mean (Glm5NextTextHyperHead), then norm.
        return self.norm.forward(x.mean(dim=1))


class Glm5NextForConditionalGeneration(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.mtp_enabled = config.mtp_layer_id is not None
        self.model = Glm53Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook: materialize the sparse layers' bmm-ready
        kv_b split and free the checkpoint-layout originals (glm_moe_dsa precedent)."""
        for layer in self.model.layers.op_list:
            if not layer._is_linear:
                layer.self_attn.prepare_for_runtime()
        if getattr(self.model, "mtp", None) is not None:
            self.model.mtp.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        # Stashed for the MTP draft pass (post-norm hidden, pre-lm_head). During
        # CUDA graph capture this binds the graph-pool tensor, whose contents
        # refresh on every replay -- the reference stays valid.
        self.last_hidden = output
        return self.lm_head.forward(output)

    def mtp_draft(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Draft logits: MTP block over (main hidden at the batch's positions,
        embedding of the tokens sampled for the NEXT positions). Runs inside the
        same forward-batch ctx as the main pass -- the MTP layer's KV row is
        written at the same out_loc, and its attention reads the same metadata.
        ``last_mtp_hidden`` stashes the block output BEFORE shared_head_norm:
        chained drafting feeds it back as the next iteration's previous hidden
        (the deepseek-MTP recurrence)."""
        emb = self.model.embed_tokens.forward(token_ids)
        out = self.model.mtp.forward(hidden, emb)
        self.last_mtp_hidden = out
        return self.lm_head.forward(self.model.mtp.shared_head_norm.forward(out))


__all__ = ["Glm5NextForConditionalGeneration"]
