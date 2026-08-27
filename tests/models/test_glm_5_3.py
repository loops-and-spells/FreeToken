"""GLM-5.3-Flash numerics and config-surface tests.

Kernel/module math is checked against the HF ``glm5_next`` reference (a HARD
import, matching test_glm_dsa.py: if transformers is missing or too old the
suite must fail, not silently skip). GPU tests exercise the vendored fla KDA
kernels, the mHC Triton kernels and the k-pool selection; the config tests are
CPU-only.
"""

from __future__ import annotations

import pytest
import torch

# HARD import (no silent skip): the reference implementation is the oracle.
from transformers.models.glm5_next import modeling_glm5_next as ref

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# ---------------------------------------------------------------------------------
# Config surface (CPU)
# ---------------------------------------------------------------------------------

_LAYER_TYPES = ["linear_attention"] * 3 + ["deepseek_sparse_attention"] + ["linear_attention"] * 3 + [
    "deepseek_sparse_attention"
]


class _TextNS:
    """Duck-typed text_config with GLM-5.3-Flash geometry (8 layers: LLLA LLLA)."""

    model_type = "glm5_next_text"
    num_hidden_layers = 8
    hidden_size = 64
    intermediate_size = 128
    vocab_size = 512
    num_attention_heads = 4
    rms_norm_eps = 1e-5
    hidden_act = "silu"
    tie_word_embeddings = False
    max_position_embeddings = 4096
    q_lora_rank = 32
    kv_lora_rank = 16
    qk_nope_head_dim = 32
    qk_rope_head_dim = 0
    v_head_dim = 32
    index_n_heads = 2
    index_head_dim = 16
    index_topk = 64
    index_kpool = 4
    index_kpool_always_select_tail = True
    indexer_rope_interleave = True
    indexer_types = ["full"] * 8
    layer_types = _LAYER_TYPES
    linear_attn_config = {
        "num_heads": 4,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1, 2, 4, 5, 6],
        "full_attn_layers": [3, 7],
    }
    hc_mult = 4
    hc_sinkhorn_iters = 20
    hc_eps = 1e-6
    n_routed_experts = 8
    num_experts_per_tok = 2
    n_shared_experts = 1
    moe_intermediate_size = 32
    norm_topk_prob = True
    first_k_dense_replace = 3
    routed_scaling_factor = 2.5
    n_group = 1
    topk_group = 1
    attention_bias = False


class _HFNS:
    architectures = ["Glm5NextForConditionalGeneration"]
    model_type = "glm5_next"
    text_config = _TextNS()
    quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "fmt": "e4m3",
    }


def test_parse_config_groups_and_quant():
    from freetoken.attention.base import AttnType
    from freetoken.models.glm_5_3.config import parse_config

    cfg = parse_config(_HFNS())
    assert cfg.expert_quant == "fp8_block" and cfg.weight_block_size == (128, 128)
    assert cfg.has_linear_attention and cfg.has_hybrid_attention
    lin = cfg.linear_attention_group()
    assert lin.layer_ids == (0, 1, 2, 4, 5, 6)
    assert lin.key_head_dim == lin.value_head_dim == 16
    specs = [s for s in cfg.kv_cache_group_specs() if s.num_layers > 0]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.layer_ids == (3, 7)
    assert spec.mla and spec.attn_type == AttnType.DSA
    # NoPE latent: bare kv_lora_rank, no rope tail.
    assert spec.head_dim == 16
    assert spec.index_head_dim == 16 and spec.num_index_layers == 2
    # k-pool gate slab is declared on the spec so the pool and cost model agree.
    assert spec.index_gate_dim == 16
    assert cfg.attn_sm_scale == pytest.approx(32 ** -0.5)
    # KDA layers are recoded "linear" so the dsa backend's slot walk skips them.
    assert cfg.glm_dsa_args.indexer_types[0] == "linear"
    assert cfg.glm_dsa_args.indexer_types[3] == "full"


def test_dsa_pool_layer_ids_and_gate_slab():
    from freetoken.kvcache.dsa_pool import DSAKVCache

    pool = DSAKVCache(
        latent_dim=16, num_layers=8, num_pages=32, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cpu"),
        index_head_dim=16, num_index_layers=2, layer_ids=(3, 7), index_gate_dim=16,
    )
    # Only the MLA layers are backed; global ids remap to dense slabs.
    assert pool._kv_buffer.shape[1] == 2
    assert pool.k_cache(3).data_ptr() == pool._kv_buffer[0, 0].data_ptr()
    assert pool.k_cache(7).data_ptr() == pool._kv_buffer[0, 1].data_ptr()
    with pytest.raises(KeyError):
        pool.k_cache(0)  # a KDA layer holds no latent KV
    g = pool.index_gate_cache(0)
    assert g.shape == (32, 16)
    kv_bytes, _ = pool.unit_bytes()
    # latent (2B x 16 x 2 layers) + index keys (2B x 16 x 2) + gates (2B x 16 x 2)
    assert kv_bytes == 64 + 64 + 64


# ---------------------------------------------------------------------------------
# KDA kernels vs the HF reference recurrence (GPU)
# ---------------------------------------------------------------------------------


def _kda_inputs(B, T, H, K, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=torch.float32)
    q, k, v = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    raw_a = mk(B, T, H, K)  # raw forget-gate input (f_b(f_a(x)) + dt_bias happens in-kernel)
    beta_raw = mk(B, T, H)
    A_log = torch.zeros(H, device=device)  # HF zero-init for the lower-bound branch
    dt_bias = mk(H * K)[0] if False else torch.randn(H * K, generator=g).to(device) * 0.1
    return q, k, v, raw_a, beta_raw, A_log, dt_bias


def _ref_gate(raw_a, A_log, dt_bias, lower_bound):
    # HF Glm5NextTextForgetGate, lower-bound branch:
    # lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
    B, T, H, K = raw_a.shape
    x = raw_a + dt_bias.view(1, 1, H, K)
    return lower_bound * torch.sigmoid(torch.exp(A_log).view(1, 1, H, 1) * x)


@cuda
def test_chunk_kda_matches_reference_recurrence():
    from freetoken.kernel.fla.kda import chunk_kda

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T, H, K = 1, 200, 4, 64  # T deliberately not a multiple of the 64-chunk
    lower_bound = -5.0
    q, k, v, raw_a, beta_raw, A_log, dt_bias = _kda_inputs(B, T, H, K, device)
    beta = torch.sigmoid(beta_raw)

    state = torch.zeros(2, H, K, K, device=device, dtype=torch.float32)
    indices = torch.tensor([1], device=device, dtype=torch.int32)
    cu = torch.tensor([0, T], device=device, dtype=torch.int32)
    out = chunk_kda(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
        raw_a, beta,
        initial_state=state, initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
        A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
    )
    if isinstance(out, tuple):
        out = out[0]

    g_ref = _ref_gate(
        raw_a.float(), A_log.float(), dt_bias.float(), lower_bound
    )
    ref_out, ref_state = ref.chunk_kimi_delta_attention(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
        g=g_ref, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out = out.view(B, T, H, K).float()
    torch.testing.assert_close(out, ref_out.float(), atol=2e-2, rtol=2e-2)
    # the pool-slot state layout is [V, K]; the reference returns [K, V]
    torch.testing.assert_close(
        state[1].float(), ref_state[0].transpose(-1, -2).float(), atol=2e-2, rtol=2e-2
    )


@cuda
def test_kda_packed_decode_matches_reference_step():
    from freetoken.kernel.fla.fused_recurrent import fused_recurrent_kda_packed_decode

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, K = 3, 4, 64
    lower_bound = -5.0
    q, k, v, raw_a, beta_raw, A_log, dt_bias = _kda_inputs(B, 1, H, K, device, seed=1)
    state0 = torch.randn(B + 1, H, K, K, device=device) * 0.1

    mixed = torch.cat(
        [q.view(B, H * K), k.view(B, H * K), v.view(B, H * K)], dim=-1
    ).to(torch.bfloat16)
    state = state0.clone()
    out = torch.empty(B, 1, H, K, device=device, dtype=torch.bfloat16)
    indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    fused_recurrent_kda_packed_decode(
        mixed.contiguous(), raw_a.view(B, H * K).contiguous(),
        beta_raw.view(B, H).contiguous(),
        A_log=A_log, dt_bias=dt_bias, scale=K ** -0.5,
        initial_state=state, out=out, ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True, lower_bound=lower_bound,
    )

    g_ref = _ref_gate(raw_a.float(), A_log.float(), dt_bias.float(), lower_bound)
    # The kernel reads bf16 conv output; feed the reference the same rounding.
    qb = mixed[:, : H * K].view(B, 1, H, K).float()
    kb = mixed[:, H * K : 2 * H * K].view(B, 1, H, K).float()
    vb = mixed[:, 2 * H * K :].view(B, 1, H, K).float()
    ref_out, ref_state = ref.recurrent_kimi_delta_attention(
        qb, kb, vb, g=g_ref, beta=torch.sigmoid(beta_raw.float()),
        # the packed kernel's state layout is [V, K]; K == V so a transpose maps them
        initial_state=state0[1:].transpose(-1, -2),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(
        out.view(B, H, K).float(), ref_out.view(B, H, K).float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        state[1:].float(), ref_state.transpose(-1, -2).float(), atol=2e-2, rtol=2e-2
    )


# ---------------------------------------------------------------------------------
# k-pool pooled keys + causal pool selection vs the HF indexer math (GPU)
# ---------------------------------------------------------------------------------


class _PoolBackendStub:
    """The three DSAAttnBackend attrs the k-pool helpers touch."""

    from freetoken.attention.dsa import DSAAttnBackend as _B

    index_kpool = 4

    def __init__(self, keys, gates):
        self._keys, self._gates = keys, gates
        self.kvcache = self
        self.device = keys.device
        self.index_scale = keys.shape[-1] ** -0.5
        self.index_topk = 8
        self._idx_slot = {0: 0}

    def index_k_cache(self, slot):
        return self._keys

    def index_gate_cache(self, slot):
        return self._gates

    _kpool_pooled_keys = _B._kpool_pooled_keys
    _kpool_expand = _B._kpool_expand
    _decode_select_kpool = _B._decode_select_kpool
    indexer_select_decode = _B.indexer_select_decode
    dsa_map_rows = staticmethod(_B.dsa_map_rows)


@cuda
def test_kpool_pooled_keys_match_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    kv_len, D, kp = 21, 16, 4
    keys = torch.randn(kv_len, D, device=device, dtype=torch.bfloat16)
    gates = torch.randn(kv_len, D, device=device, dtype=torch.bfloat16)
    ape = torch.randn(kp, D, device=device)

    stub = _PoolBackendStub(keys, gates)
    rows = torch.arange(kv_len, device=device, dtype=torch.int32).view(1, -1)
    num_pools = kv_len // kp
    pooled = stub._kpool_pooled_keys(0, rows, num_pools, ape)[0]  # [P, D]

    # Reference: per complete pool, channel-wise softmax(gate + ape) weighted mean
    # (HF get_pooled_states with index_kpool_compress=True).
    ref_keys = keys[: num_pools * kp].view(num_pools, kp, D).float()
    logits = gates[: num_pools * kp].view(num_pools, kp, D).float() + ape.float()[None]
    ref_pooled = (torch.softmax(logits, dim=1) * ref_keys).sum(dim=1)
    torch.testing.assert_close(pooled, ref_pooled, atol=1e-3, rtol=1e-3)


@cuda
def test_kpool_decode_selection_covers_tail_and_causality():
    torch.manual_seed(0)
    device = torch.device("cuda")
    kv_len, D, kp, Hi = 23, 16, 4, 2
    keys = torch.randn(64, D, device=device, dtype=torch.bfloat16)
    gates = torch.randn(64, D, device=device, dtype=torch.bfloat16)
    ape = torch.randn(kp, D, device=device)
    stub = _PoolBackendStub(keys, gates)

    class _MD:
        rows = torch.arange(64, device=device, dtype=torch.int32).view(1, -1)
        kvlen = torch.tensor([kv_len], device=device, dtype=torch.int32)

    q_idx = torch.randn(1, Hi, D, device=device, dtype=torch.bfloat16)
    w = torch.randn(1, Hi, device=device)
    sel, cnt = stub._decode_select_kpool(_MD(), 0, q_idx, w, ape)
    sel = sel.view(-1)
    picked = sel[sel >= 0].tolist()
    # every selected row must be visible (physical row == position here)
    assert all(p < kv_len for p in picked)
    # the incomplete tail pool (positions 20..22) is always selected
    for tail_pos in range((kv_len // kp) * kp, kv_len):
        assert tail_pos in picked
    # selected complete pools expand to whole pools: positions come in runs of kp
    non_tail = [p for p in picked if p < (kv_len // kp) * kp]
    assert len(non_tail) % kp == 0
    for base in {p - p % kp for p in non_tail}:
        assert all(base + i in non_tail for i in range(kp))
    # top-(index_topk // kp) pools == 2 pools + tail
    assert len(non_tail) == (stub.index_topk // kp) * kp


# ---------------------------------------------------------------------------------
# mHC vs the HF hyper-connection reference (GPU)
# ---------------------------------------------------------------------------------


@cuda
def test_hyper_connection_matches_reference_decoder_math():
    torch.manual_seed(0)
    device = torch.device("cuda")
    T, hc, D = 5, 4, 32
    from freetoken.models.glm_5_3.model import _HyperConnection

    site = _HyperConnection(hc, D, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-5)
    site.fn = torch.randn((2 + hc) * hc, hc * D, device=device) * 0.02
    site.base = torch.zeros((2 + hc) * hc, device=device)
    site.scale = torch.ones(3, device=device)

    x = torch.randn(T, hc, D, device=device, dtype=torch.bfloat16)
    y_col, post, comb = site.pre(x)
    sub_out = torch.randn(T, D, device=device, dtype=torch.bfloat16)
    streams = site.post(sub_out, x, post, comb)

    # Reference: DeepseekV4HyperConnection math (per the HF glm5_next decoder):
    # mixes = fn(norm-rescaled flat streams); pre/post/comb from hc_split_sinkhorn;
    # out = post * sub_out (expanded) + comb^T @ residual.
    xf = x.reshape(T, hc * D).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
    mixes = (xf @ site.fn.t()) * rsqrt
    pre_r = torch.sigmoid(mixes[:, :hc]) + 1e-6
    post_r = 2 * torch.sigmoid(mixes[:, hc : 2 * hc])
    comb_r = mixes[:, 2 * hc :].view(T, hc, hc).softmax(dim=-1) + 1e-6
    comb_r = comb_r / (comb_r.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb_r = comb_r / (comb_r.sum(dim=-1, keepdim=True) + 1e-6)
        comb_r = comb_r / (comb_r.sum(dim=-2, keepdim=True) + 1e-6)
    y_ref = (pre_r.unsqueeze(-1) * x.float().reshape(T, hc, D)).sum(dim=1)
    streams_ref = post_r.unsqueeze(-1) * sub_out.float().unsqueeze(1) + torch.matmul(
        comb_r.transpose(-1, -2), x.float().reshape(T, hc, D)
    )
    torch.testing.assert_close(y_col.float(), y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(streams.float(), streams_ref, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------------
# Backend-level: NoPE + k-pool ragged prefill vs a from-scratch reference (GPU)
# ---------------------------------------------------------------------------------


def _make_kpool_backend(pages=256, topk=16, kp=4, latent=64, idx_d=16):
    from types import SimpleNamespace

    from freetoken import core
    from freetoken.attention.dsa import DSAAttnBackend
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache.dsa_pool import DSAKVCache

    core._GLOBAL_CTX = None
    ctx = Context(page_size=1)
    ctx.page_table = torch.zeros(4, pages, dtype=torch.int32, device="cuda")
    ctx.kv_cache = DSAKVCache(
        latent, 2, pages, 1, torch.bfloat16, torch.device("cuda"),
        index_head_dim=idx_d, num_index_layers=1, index_gate_dim=idx_d,
    )
    set_global_ctx(ctx)
    args = SimpleNamespace(
        kv_lora_rank=latent, qk_rope_head_dim=0, qk_head_dim=latent,
        index_topk=topk, index_head_dim=idx_d,
        indexer_types=("full", "shared"),
        index_kpool=kp,
    )
    cfg = SimpleNamespace(glm_dsa_args=args, num_qo_heads=4, attn_sm_scale=None, num_layers=2)
    return DSAAttnBackend(cfg), ctx


@cuda
def test_backend_kpool_nope_prefill_matches_reference():
    """One long request through the BACKEND with k-pool selection and a NoPE latent
    (rope width 0): pooled scoring, causal pool top-k, expansion + tail, gathered
    sparse attention -- all checked against a from-scratch torch reference that
    follows the HF indexer semantics."""
    from types import SimpleNamespace

    torch.manual_seed(7)
    dv, h, idx_h, idx_d, topk, kp = 64, 4, 2, 16, 16, 4
    backend, ctx = _make_kpool_backend(topk=topk, kp=kp, latent=dv, idx_d=idx_d)
    pool = ctx.kv_cache
    scale = backend.sm_scale

    # one request: kv 100 > topk 16 -> real k-pool selection; extend of 10 queries
    kv, ext = 100, 10
    ctx.page_table[0, :kv] = torch.arange(kv, device="cuda")
    reqs = [SimpleNamespace(extend_len=ext, device_len=kv, table_idx=0)]
    positions = torch.arange(kv - ext, kv).cuda()
    out_loc = torch.arange(kv - ext, kv).cuda()

    n_hist = kv - ext
    hist_loc = torch.arange(n_hist).cuda()
    hist_ckv = torch.randn(n_hist, dv, device="cuda", dtype=torch.bfloat16)
    for lid in (0, 1):
        pool.store_kv(hist_ckv, hist_ckv.new_empty(n_hist, 0), hist_loc, lid)
    hist_k = torch.randn(n_hist, idx_d, device="cuda", dtype=torch.bfloat16)
    hist_g = torch.randn(n_hist, idx_d, device="cuda", dtype=torch.bfloat16)
    pool.store_index_k(hist_k, hist_loc, 0)
    pool.store_index_gate(hist_g, hist_loc, 0)

    batch = SimpleNamespace(reqs=reqs, positions=positions, out_loc=out_loc,
                            active_table_idx=None, attn_metadata=None)
    backend.prepare_metadata(batch)

    q_nope = torch.randn(ext, h, dv, device="cuda", dtype=torch.bfloat16)
    q_pe = q_nope.new_empty(ext, h, 0)
    c_kv = torch.randn(ext, dv, device="cuda", dtype=torch.bfloat16)
    k_rope = c_kv.new_empty(ext, 0)
    ape = torch.randn(kp, idx_d, device="cuda")
    qkw = (torch.randn(ext, idx_h, idx_d, device="cuda", dtype=torch.bfloat16),
           torch.randn(ext, idx_d, device="cuda", dtype=torch.bfloat16),
           torch.randn(ext, idx_h, device="cuda").abs(),
           torch.randn(ext, idx_d, device="cuda", dtype=torch.bfloat16),
           ape)

    o = backend.mla_forward(q_nope, q_pe, c_kv, k_rope, 0, batch, indexer_qkw=qkw)

    # ---- reference: HF k-pool indexer semantics + masked dense attention ----
    all_k = pool.index_k_cache(0)[:kv].float()
    all_g = pool.index_gate_cache(0)[:kv].float()
    slab = pool.latent_rows(0)
    q_idx, _, w = qkw[0], qkw[1], qkw[2]
    q_cat = torch.cat([q_nope, q_pe], -1)
    for j in range(ext):
        p = kv - ext + j  # absolute position
        n_vis_pools = (p + 1) // kp
        keys = all_k[: n_vis_pools * kp].view(n_vis_pools, kp, idx_d)
        logits = all_g[: n_vis_pools * kp].view(n_vis_pools, kp, idx_d) + ape.float()[None]
        pooled = (torch.softmax(logits, dim=1) * keys).sum(1)  # [P, D]
        s = (torch.einsum("hd,pd->hp", q_idx[j].float(), pooled).relu()
             * (idx_d**-0.5) * w[j][:, None].float()).sum(0)
        k_pools = min(topk // kp, n_vis_pools)
        sel_pools = s.topk(k_pools).indices
        sel_pos = (sel_pools[:, None] * kp + torch.arange(kp, device="cuda")).flatten()
        tail = torch.arange(n_vis_pools * kp, p + 1, device="cuda")
        sel_pos = torch.cat([sel_pos, tail])
        ref = _kpool_ref_attend(q_cat[j], slab, sel_pos, scale, dv)
        assert (o[j].float() - ref).abs().max().item() < 3e-2, f"q{j}"


def _kpool_ref_attend(q_cat, pool_rows, live_rows, scale, dv):
    k = pool_rows[live_rows.long()].float()
    s = (q_cat.float() @ k.T) * scale
    return s.softmax(-1) @ k[:, :dv]


# ---------------------------------------------------------------------------------
# NVFP4 community-quant surface (experts-only quant; dense stays bf16)
# ---------------------------------------------------------------------------------


def test_parse_config_nvfp4_expert_quant():
    from freetoken.models.glm_5_3.config import parse_config

    class _NVFP4HF(_HFNS):
        quantization_config = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {"group_0": {"targets": ["Linear"]}},
        }

    cfg = parse_config(_NVFP4HF())
    assert cfg.expert_quant == "nvfp4"
    assert cfg.weight_block_size is None


def test_nvfp4_source_spec_matches_checkpoint_keys():
    from freetoken.models.glm_5_3.weight import _NVFP4_EXPERT_KEY_RE, _NVFP4_SOURCE_SPEC

    m = _NVFP4_EXPERT_KEY_RE.match(
        "model.language_model.layers.3.mlp.experts.287.down_proj.weight_scale_2"
    )
    assert m and m["layer"] == "3" and m["expert"] == "287" and m["kind"] == "weight_scale_2"
    # MTP and vision keys never match
    assert _NVFP4_EXPERT_KEY_RE.match("mtp.layers.45.mlp.experts.0.gate_proj.weight") is None
    assert _NVFP4_EXPERT_KEY_RE.match(
        "model.visual.blocks.0.mlp.experts.0.gate_proj.weight"
    ) is None
    # bank index is the MoE layer (global minus the dense prefix)
    class _C:
        first_k_dense_replace = 3
        num_moe_layers = 42

    assert _NVFP4_SOURCE_SPEC.layer_to_bank(3, _C) == 0
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(44, _C) == 41
    # the MTP layer (45) carries its own experts in the checkpoint: skipped
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(45, _C) is None
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(0, _C) is None
