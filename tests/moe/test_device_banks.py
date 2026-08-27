"""Device-resident expert banks: a layer's banks live on a SECOND CUDA device
instead of host RAM (FREETOKEN_DEVICE_BANK_LAYERS), read by the same copy paths
over PCIe P2P. Capacity feature for expert sets larger than RAM on hosts with
an idle card."""

from __future__ import annotations

import mmap

import pytest
import torch

from freetoken.moe.host_banks import DEVICE_LABEL_PREFIX, HostBank, _settle

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")


@cuda
def test_hostbank_migrate_moves_tensor_and_drops_pages():
    bank = HostBank((8, 1024), torch.bfloat16)
    ref = torch.randn(8, 1024, dtype=torch.bfloat16)
    bank.tensor.copy_(ref)
    _settle(bank, f"{DEVICE_LABEL_PREFIX}cuda:0")
    assert bank.tensor.device.type == "cuda"
    assert torch.equal(bank.tensor.cpu(), ref)
    # the host mapping survives (fill-time views may exist) but reads as zeros
    # after MADV_DONTNEED -- physical pages were returned to the OS
    assert isinstance(bank._buf, mmap.mmap)


@cuda
def test_hostbank_migrate_refuses_after_pin():
    bank = HostBank((4, 64), torch.bfloat16)
    bank.pin()
    with pytest.raises(AssertionError):
        bank.migrate("cuda:0")


@two_gpus
def test_offload_cache_serves_device_bank_layer_on_gpu_path():
    from freetoken.moe.offload_cache import OffloadMoeCache

    torch.manual_seed(0)
    E = 4
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=E,
        cache_size=6,
        device=torch.device("cuda:1"),
    )
    # map peer memory in torch's context (what the engine overlay does)
    from freetoken.engine.engine import _enable_peer_access

    _enable_peer_access(torch.device("cuda", 1), 0)
    # layer 0 banks in host RAM (pinned semantics), layer 1 banks on cuda:0
    gate_up = [torch.randn(E, 32, 8), torch.randn(E, 32, 8).to("cuda:0")]
    down = [torch.randn(E, 8, 16), torch.randn(E, 8, 16).to("cuda:0")]
    # device labels are NOT "unpinned": no cpu_layer_ids routing required
    cache.set_bank_sources(
        {"gate_up": gate_up, "down": down},
        layer_residency=["pinned", f"{DEVICE_LABEL_PREFIX}cuda:0"],
    )
    assert cache._unpinned_layers == frozenset()

    # whole-layer materialize from the DEVICE bank must land the right rows
    # (the engine always runs with the serving device current)
    with torch.cuda.device(1):
        cache.materialize_layer(1)
        cache.copy_missing()
        torch.cuda.synchronize(1)
    got = cache.bank_caches["gate_up"][:E].cpu()
    assert torch.allclose(got, gate_up[1].cpu(), atol=0, rtol=0)


def test_engine_overlay_parses_and_labels(monkeypatch):
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine

    monkeypatch.setenv("FREETOKEN_DEVICE_BANK_LAYERS", "cuda:0=2")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs for the peer-access probe")
    self = SimpleNamespace(device=torch.device("cuda", 1))
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_moe_layers=5, nvfp4_backend="triton"),
        moe_prefill_overlap=False,
    )
    labels = Engine._overlay_device_bank_layers(self, config, None, frozenset({4}))
    # trailing non-CPU layers get the device label; the CPU layer is skipped
    assert labels == [
        "pinned", "pinned", f"{DEVICE_LABEL_PREFIX}cuda:0",
        f"{DEVICE_LABEL_PREFIX}cuda:0", "pinned",
    ]
