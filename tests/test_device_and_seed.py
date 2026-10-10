from __future__ import annotations

import numpy as np
import pytest
import torch

from mfsi_cxr.utils import device
from mfsi_cxr.utils.seed import set_seed


def test_forced_cpu() -> None:
    assert device.resolve_device("cpu") == "cpu"


@pytest.mark.parametrize("pref", ["auto", "cuda", "mps"])
def test_fallback_to_cpu_without_accelerators(monkeypatch: pytest.MonkeyPatch, pref: str) -> None:
    monkeypatch.setattr(device, "_cuda_ok", lambda: False)
    monkeypatch.setattr(device, "_mps_ok", lambda: False)
    assert device.resolve_device(pref) == "cpu"


def test_fallback_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_cuda_ok", lambda: True)
    monkeypatch.setattr(device, "_mps_ok", lambda: True)
    assert device.resolve_device("auto") == "cuda"
    assert device.resolve_device("mps") == "mps"
    monkeypatch.setattr(device, "_cuda_ok", lambda: False)
    assert device.resolve_device("cuda") == "mps"


def test_unknown_device() -> None:
    with pytest.raises(ValueError):
        device.resolve_device("tpu")


def test_set_seed_is_reproducible() -> None:
    set_seed(7)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
