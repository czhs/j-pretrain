"""AdamW construction with correct weight-decay param groups.

Weight decay must NOT apply to biases, norm weights, or the (tied) embedding —
standard practice and what SmolLM2/Llama training uses. Betas/eps/wd come from the
frozen :class:`~j_pretrain.config.schemas.StageConfig`; the ``fused`` kernel is an
execution choice (CUDA only) and never changes optimizer math.
"""
from __future__ import annotations

import torch

from j_pretrain.config.schemas import StageConfig


def _decay_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No WD on 1-D params (biases, RMSNorm gains) or any embedding weight.
        if p.ndim < 2 or "norm" in name.lower() or "embed" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_adamw(model: torch.nn.Module, cfg: StageConfig, *, fused: bool = False,
                lr: float | None = None) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        _decay_groups(model, cfg.weight_decay),
        lr=lr if lr is not None else cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        fused=fused,
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
