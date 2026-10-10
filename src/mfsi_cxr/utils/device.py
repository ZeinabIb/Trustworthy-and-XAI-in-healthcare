"""Device choice with a safe fallback: cuda -> mps -> cpu. No code path may require a GPU."""

from __future__ import annotations

from typing import Any

DEVICES = ("auto", "cuda", "mps", "cpu")


def _cuda_ok() -> bool:
    import torch

    return torch.cuda.is_available()


def _mps_ok() -> bool:
    import torch

    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(preference: str) -> str:
    """Return a usable device name. An unavailable explicit choice falls back down the chain."""
    if preference not in DEVICES:
        raise ValueError(f"unknown device {preference!r}; expected one of {DEVICES}")
    if preference == "cpu":
        return "cpu"
    if preference in ("auto", "cuda") and _cuda_ok():
        return "cuda"
    if preference in ("auto", "cuda", "mps") and _mps_ok():
        return "mps"
    return "cpu"


def device_from_config(cfg: dict[str, Any]) -> str:
    return resolve_device(cfg["train"]["device"])
