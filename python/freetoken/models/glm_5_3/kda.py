"""GLM-5.3-Flash KDA (Kimi Delta Attention) linear-attention op.

Same serving shape as qwen3_5_moe's GatedDeltaNet -- per-request conv + recurrent
state in ``ctx.linear_state_pool``, fla chunk kernel for prefill, fused packed
kernel for decode -- with the KDA differences:

* the decay gate is PER-CHANNEL (``[HV, K]``), computed in-kernel from the raw
  low-rank forget projection with the safe lower-bound variant
  ``lower_bound * sigmoid(exp(A_log) * (f(x) + dt_bias))`` (GLM-5.3 sets
  ``gate_lower_bound = -5.0``);
* q/k/v are separate checkpoint projections fused at load into one GEMM
  (``in_proj_qkv``), all bf16 (the checkpoint's FP8 pass skips KDA layers);
* the output gate is a low-rank projection through a SIGMOID-gated RMSNorm
  (``o_norm``), fp32-strict like the HF reference.

Parameter names follow the checkpoint (``q_proj``/``k_proj``/``v_proj`` fused ->
``in_proj_qkv``; ``q/k/v_convNd`` fused -> ``conv1d``; ``f_a_proj``/``f_b_proj``/
``b_proj``/``g_a_proj``/``g_b_proj``/``A_log``/``dt_bias``/``o_norm``/``o_proj``).
"""

from __future__ import annotations

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _SigmoidGatedRMSNorm(BaseOP):
    """RMSNorm of x followed by a sigmoid(z) gate (HF Glm5NextTextRMSNormGated:
    fp32-strict norm, ``norm(x) * sigmoid(z)``), via the fused fla kernel."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation="sigmoid",
        )


class Glm53KDA(BaseOP):
    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int,
        conv_kernel_size: int, rms_norm_eps: float, lower_bound: float | None,
        layer_id: int,
    ):
        self.layer_id = layer_id
        # The fla kernels read/write the recurrent state as [V, K] while the
        # LinearStatePool declares [K, V]; they coincide only when the head dims are
        # equal (same invariant the GDN op documents). GLM-5.3 is 128/128.
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_dim = num_heads * head_dim
        self.conv_dim = 3 * self.qkv_dim
        self.conv_kernel_size = conv_kernel_size
        self.lower_bound = lower_bound

        # q|k|v fused into one GEMM (checkpoint stores them separately; the loader
        # concatenates). All KDA projections are bf16 in the FP8 checkpoint.
        self.in_proj_qkv = LinearColParallelMerged(
            hidden_size, [self.qkv_dim] * 3, has_bias=False
        )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Low-rank forget gate (f_b(f_a(x)) -> raw per-channel gate input) + beta.
        self.f_a_proj = LinearReplicated(hidden_size, head_dim, has_bias=False)
        self.f_b_proj = LinearReplicated(head_dim, self.qkv_dim, has_bias=False)
        self.b_proj = LinearReplicated(hidden_size, num_heads, has_bias=False)
        # Low-rank output gate feeding the sigmoid-gated norm.
        self.g_a_proj = LinearReplicated(hidden_size, head_dim, has_bias=False)
        self.g_b_proj = LinearReplicated(head_dim, self.qkv_dim, has_bias=False)
        # Gating params fp32 (the kernels read them as fp32; the load downcast
        # exempts *.A_log / *.dt_bias). dt_bias is PER-CHANNEL for KDA.
        self.A_log = torch.empty(num_heads, dtype=torch.float32)
        self.dt_bias = torch.empty(self.qkv_dim, dtype=torch.float32)
        self.o_norm = _SigmoidGatedRMSNorm(head_dim, eps=rms_norm_eps)
        self.o_proj = LinearReplicated(self.qkv_dim, hidden_size, has_bias=False)

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        conv_in = self.in_proj_qkv.forward(hidden_states)  # [T, 3*qkv_dim]
        a = self.f_b_proj.forward(self.f_a_proj.forward(hidden_states))  # raw gate [T, HV*K]
        b = self.b_proj.forward(hidden_states)  # raw beta [T, HV]
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            from freetoken.kernel.fla.fused_recurrent import (
                fused_recurrent_kda_packed_decode,
            )

            mixed = causal_conv1d_decode(
                conv_in, pool.conv_states[li], self._conv_weight(), fla.cache_indices
            )  # silu(conv) [B, conv_dim]
            B = mixed.shape[0]
            out = mixed.new_empty(B, 1, self.num_heads, self.head_dim)
            fused_recurrent_kda_packed_decode(
                mixed.contiguous(),
                a.float().contiguous(),
                b.float().contiguous(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=self.head_dim**-0.5,
                initial_state=pool.recurrent_states[li],
                out=out,
                ssm_state_indices=fla.cache_indices,
                use_qk_l2norm_in_kernel=True,
                lower_bound=self.lower_bound,
            )
            core_out = out.view(B, self.num_heads, self.head_dim)
        else:
            from freetoken.kernel.fla.kda import chunk_kda

            x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
            mixed = causal_conv1d_varlen(
                x, self._conv_weight(), pool.conv_states[li],
                fla.cu_seqlens, fla.cache_indices, fla.has_initial_state,
            ).transpose(0, 1)  # [total, conv_dim], silu applied
            q, k, v = torch.split(mixed, [self.qkv_dim] * 3, dim=-1)
            q = q.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            k = k.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            v = v.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            g_raw = a.float().reshape(1, total, self.num_heads, self.head_dim)
            beta = torch.sigmoid(b.float()).reshape(1, total, self.num_heads)
            # The chunk kernel reads + writes initial_state[cache_indices] in place;
            # fresh sequences must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            result = chunk_kda(
                q, k, v, g_raw, beta,
                initial_state=pool.recurrent_states[li],
                initial_state_indices=fla.cache_indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=fla.cu_seqlens,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.lower_bound,
                output_intermediate_states=track,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result
            core_out = core_out.reshape(total, self.num_heads, self.head_dim)

        gate = self.g_b_proj.forward(self.g_a_proj.forward(hidden_states))
        out = self.o_norm.forward(
            core_out.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim)
        ).reshape(total, self.qkv_dim)
        return self.o_proj.forward(out)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Hybrid-radix chunk-boundary snapshot (same contract as the GDN op: h rows
        are [V, K], the pool is [K, V]; identical because head dims are equal)."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))


__all__ = ["Glm53KDA"]
