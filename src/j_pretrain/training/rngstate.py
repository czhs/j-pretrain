"""Capture and restore all RNG streams for byte-exact resume.

A resumable checkpoint must let training continue as if never interrupted. That
requires restoring Python's ``random``, NumPy's global RNG, and Torch's CPU (and,
when present, CUDA) generators. The data order in this project is a *pure function*
of an integer cursor (see :mod:`j_pretrain.training.dataplan`), so RNG restoration
here only affects model-side stochasticity (dropout is 0 for Fig 3a, but we restore
anyway for strict determinism and future-proofing).
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_cpu_byte_tensor(state["torch_cpu"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_cpu_byte_tensor(s) for s in state["torch_cuda"]]
        # Only restore as many device states as we have; guards host/GPU-count drift.
        torch.cuda.set_rng_state_all(cuda_states[: torch.cuda.device_count()])


def _as_cpu_byte_tensor(x: Any) -> torch.Tensor:
    """torch.save round-trips rng_state as a ByteTensor already; be defensive anyway."""
    if isinstance(x, torch.Tensor):
        return x.cpu().to(torch.uint8)
    return torch.as_tensor(x, dtype=torch.uint8)
