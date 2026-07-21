"""Build the 135M SmolLM2-style model from the frozen scientific config.

We construct from config with RANDOM initialization (the reproduction must NOT
start from a pretrained public checkpoint). All Stage-1 conditions must start
from byte-identical weights; use :func:`build_model` with a fixed seed and save
the result as the permanent init checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from j_pretrain.config.schemas import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "smollm2-135m.json"


def load_model_config(path: Optional[str | Path] = None) -> ModelConfig:
    return ModelConfig.from_json(path or DEFAULT_MODEL_CONFIG)


def build_hf_config(mc: ModelConfig, attn_implementation: str = "sdpa") -> LlamaConfig:
    cfg = LlamaConfig(**mc.to_hf_kwargs())
    cfg._attn_implementation = attn_implementation
    return cfg


def build_model(
    mc: Optional[ModelConfig] = None,
    seed: Optional[int] = None,
    attn_implementation: str = "sdpa",
    dtype: Optional[torch.dtype] = None,
) -> LlamaForCausalLM:
    """Construct a randomly-initialized model. Deterministic given ``seed``."""
    mc = mc or load_model_config()
    if seed is not None:
        torch.manual_seed(seed)
    cfg = build_hf_config(mc, attn_implementation=attn_implementation)
    model = LlamaForCausalLM(cfg)
    if dtype is not None:
        model = model.to(dtype)
    return model


def count_parameters(model: LlamaForCausalLM) -> tuple[int, int]:
    """(total, unique) param counts. With tied embeddings total==unique because the
    tied weight is a single storage shared by embed + lm_head."""
    seen: dict[int, int] = {}
    total = 0
    for p in model.parameters():
        total += p.numel()
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen[ptr] = p.numel()
    unique = sum(seen.values())
    return total, unique
