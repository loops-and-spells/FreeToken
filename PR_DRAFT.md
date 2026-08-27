# feat(glm_5_3): serve GLM-5.3-Flash (hybrid KDA + NoPE sparse MLA, mHC, k-pool DSA)

Closes #215.

## What

Adds the GLM-5.3-Flash model family (`zai-org/GLM-5.3-Flash`, HF arch
`Glm5NextForConditionalGeneration`, block-fp8 checkpoint, 320B-A18B). The
architecture is a three-way graft of lineages FreeToken already serves, and the
implementation leans on that:

* **34 KDA linear-attention layers** — served by the KDA kernels vendored from
  sglang's fla tree (chunk prefill + packed decode, per-channel decay with the
  safe `lower_bound * sigmoid(exp(A_log) * (g + bias))` gate computed in-kernel).
  The op mirrors `qwen3_5_moe`'s GatedDeltaNet shape: same LinearStatePool
  geometry, same hybrid-radix chunk-boundary snapshots.
* **11 NoPE sparse-MLA layers** — GLM-5.2's MLA + DSA machinery with two deltas:
  no rope anywhere in the sparse path (`qk_rope_head_dim == 0`; the gathered-KV
  kernel's rope half is now compile-time elided at `D_R == 0`), and a
  k-pool-compressed indexer: pools of `index_kpool` tokens are scored (channel-wise
  `softmax(gate + ape)`-weighted key means), winners are selected causally at pool
  granularity through the existing ratio-based selection, expanded back to token
  rows, with the incomplete tail pool always selected. Pooled keys are recomputed
  at selection time from the index-key slab plus a new gate-score slab on
  `DSAKVCache` (declared on the group spec so the pool and KV cost model agree).
* **mHC residual streams** — DeepSeek-V4's manifold-constrained Hyper-Connections
  through the existing `kernel/triton/dsv4` kernels; GLM's final collapse is an
  unweighted mean (no learned head mix).
* **MoE** — GLM-5.2-style sigmoid/noaux_tc routing; the routed experts are
  block-fp8 and reuse `qwen3_5_moe`'s fp8 expert-bank builder verbatim (identical
  checkpoint key layout and MoE dims).

Engine change of note: `MLAKVCache`/`DSAKVCache` learn the same optional
`layer_ids` remap `MHAKVCache` has, so a hybrid model backs latent slabs only for
its MLA layers (4x over-allocation otherwise).

Text-only serving (vision tower dropped, matching the repo contract); the MTP
layer is not loaded (no speculative decoding in v1).

## Validation

* `tests/models/test_glm_5_3.py` (16 tests) oracles the new math against the HF
  `glm5_next` reference: KDA chunk + decode kernels vs the reference recurrence
  (fp32 chained), pooled keys + pool selection vs the indexer semantics, mHC
  pre/post vs the hyper-connection math, and a backend-level NoPE + k-pool
  ragged prefill vs a from-scratch oracle.
* Real-weights e2e (dev layer cap `FREETOKEN_GLM_DSA_MAX_LAYERS=5`, partial
  checkpoint): `ft serve` boots, generates on short prompts and on a 3k-token
  prompt (past `index_topk` → real k-pool selection). Next-token argmax agrees
  with HF transformers (same capped weights, GPU) on 7/8 diverse prompts, with
  the 8th picking HF's #2 — consistent with W8A8 (HF fp8 kernels) vs W8A16
  (FreeToken) precision, not a mapping error.
* **Full 45-layer serving** (LibertAIDAI/GLM-5.3-Flash-NVFP4, 181 GiB): on a
  single 96 GB serving GPU + a second card holding 22 MoE layers' expert banks
  device-resident (the new multi-device bank placement) + ~76 GiB pinned host
  RAM. Coherent thinking-mode answers, **~29 tok/s decode** (CUDA graphs,
  ~24% decode expert-cache miss rate), steady across requests. This checkpoint
  stores per-expert NVFP4 with a separate expert set on the trailing MTP layer,
  which the source spec maps to None (skipped).
* Full suite: no regressions (the one failing test on my machine,
  `test_muse_glimmer_parsers.py::test_mixed_quant_groups_rejected_in_any_order`,
  fails identically on unmodified main — env issue).

Tested on: 2x RTX PRO 6000 Blackwell (SM120), driver <fill>, CUDA 13 / torch
2.11.0+cu130, Linux, 128 GB RAM. Checkpoints: `zai-org/GLM-5.3-Flash`
(block-fp8, layer-capped numerics validation) and
`LibertAIDAI/GLM-5.3-Flash-NVFP4` (full 45-layer serving). The native block-fp8
release needs ~300 GiB of expert residency — beyond this machine; numbers
welcome from anyone with the RAM.

## Companion changes in this branch

* **Multi-device / device-resident expert banks** (`FREETOKEN_DEVICE_BANK_LAYERS`):
  trailing MoE layers' banks settle on other CUDA devices instead of host RAM —
  capacity for expert sets larger than RAM on hosts with an idle card. Two
  hard-won correctness notes are encoded in the code: (1) device banks must be
  allocated in LEGACY segments — torch's expandable (VMM) segments are not
  peer-mappable and the copy kernels Warp-MMU-fault dereferencing them (found
  via GPU coredump; memcpys still work, so warmup passes and the first real
  request dies); (2) CPython's anonymous mmap is MAP_SHARED, so freeing a
  migrated layer's host pages needs `MADV_REMOVE`, not `MADV_DONTNEED`
  (measured: DONTNEED returns 0 of 4 GiB, REMOVE returns all of it); (3) P2P
  reads do NOT count as activity on the target card's clock governor -- an
  otherwise idle bank device sags to P8 with its memory clock at ~3% speed and
  decode collapses ~3.7x (measured 8 vs 29 tok/s). The engine holds bank
  devices awake with a tiny periodic kernel (`FREETOKEN_BANK_KEEPALIVE=0` opts
  out when clocks are locked via `nvidia-smi -lmc`).
* **Per-layer hybrid decode with device banks**: `--moe-backend hybrid` now
  composes with device banks -- host-resident layers use the hybrid PCIe+CPU
  co-compute, device-bank layers (no host copy) keep the plain GPU offload
  path. Opt-in; on my machine pure offload measured faster (the benched
  CPU-bandwidth ratio was optimistic under real serving), so nothing changes
  by default.
* `/health`-poll-friendly ops flags worth knowing: `--expert-load serial` is
  required when device banks are active (the parallel reader fills layers out
  of order and defeats the migrate-and-drop RAM pacing), and
  `FREETOKEN_KERNEL_CACHE_JOBS` caps the first-boot kernel-compile fan-out
  (defaults to all cores; each job is a compiler using real RAM).

## Known limits / follow-ups

* CUDA graph capture not yet validated with device banks (entry serves eager).
* k-pool decode recomputes pooled keys from the key+gate slabs every step
  (gather-bound); a cached pooled-key slab or a fused gather-pool-score kernel
  halves the traffic. Marked in-code.
* marlin/b12x NVFP4 backends repack source banks in place on the host, so
  device banks force the native triton layout for now.
* MTP speculative decoding not wired.
