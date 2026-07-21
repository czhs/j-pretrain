"""Resumable training loop (one stage) with exact per-source token accounting.

Precision is **BF16 mixed** (Table 5): fp32 master parameters, bf16 autocast compute
on CUDA (no GradScaler — bf16 has fp32 dynamic range). On CPU (tests) it runs plain
fp32. Optimizer = AdamW with frozen betas/wd/clip from the :class:`StageConfig`.

Data order is a pure function of the integer cursor ``windows_consumed`` (see
:mod:`.dataplan`), so a resumable checkpoint stores just that cursor plus the exact
fp32 model/optimizer state and all RNG streams — resume is byte-exact.

The loop core (:meth:`train_steps`) is deliberately small and side-effect free so it
can be unit-tested for deterministic resume; milestone checkpointing/eval scheduling
is layered on top by the stage driver (see :mod:`j_pretrain.orchestration`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import torch
import torch.nn.functional as F

from j_pretrain.config.schemas import ExecConfig, StageConfig
from j_pretrain.training.loader import PlanLoader
from j_pretrain.training.optim import build_adamw, set_lr
from j_pretrain.training.rngstate import capture_rng, restore_rng
from j_pretrain.training.schedule import cosine_lr


@dataclass
class TokenCounters:
    total: int = 0
    c4: int = 0
    mp: int = 0
    chempile: int = 0

    def add(self, counts: Mapping[str, int], seq_len: int) -> None:
        self.c4 += counts.get("c4", 0) * seq_len
        self.mp += counts.get("mp", 0) * seq_len
        self.chempile += counts.get("chempile", 0) * seq_len
        self.total = self.c4 + self.mp + self.chempile

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total, "c4": self.c4, "mp": self.mp,
                "chempile": self.chempile}


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        stage_cfg: StageConfig,
        exec_cfg: ExecConfig,
        loader: PlanLoader,
        total_steps: int,
        device: str = "cpu",
        fused_adamw: bool = False,
    ):
        self.model = model.to(device)
        self.cfg = stage_cfg
        self.exec = exec_cfg
        self.loader = loader
        self.device = device
        self.seq_len = stage_cfg.seq_len
        self.total_steps = int(total_steps)
        self.grad_accum = stage_cfg.grad_accum(exec_cfg.microbatch_size)
        self.microbatch = exec_cfg.microbatch_size
        self.opt = build_adamw(self.model, stage_cfg, fused=fused_adamw)
        # bf16 autocast only on CUDA; CPU path is plain fp32 (tests / determinism).
        self._use_autocast = device.startswith("cuda") and exec_cfg.dtype == "bfloat16"

        # mutable training state
        self.opt_step = 0
        self.windows_consumed = 0
        self.tokens = TokenCounters()
        self.best_val: Optional[float] = None
        self.no_improve_evals = 0

    # ---- core loop -------------------------------------------------------
    def _autocast(self):
        if self._use_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    def _forward_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, labels=input_ids)
        return out.loss

    def train_steps(self, n_steps: int,
                    on_step: Optional[Callable[["Trainer", float], None]] = None) -> None:
        """Advance ``n_steps`` optimizer steps (each = grad_accum microbatches)."""
        self.model.train()
        for _ in range(n_steps):
            if self.opt_step >= self.total_steps:
                break
            lr = cosine_lr(self.opt_step, self.cfg.peak_lr, self.cfg.min_lr,
                           self.cfg.warmup_steps, self.total_steps)
            set_lr(self.opt, lr)
            self.opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _micro in range(self.grad_accum):
                ids, counts = self.loader.windows(self.windows_consumed, self.microbatch)
                ids = ids.to(self.device)
                with self._autocast():
                    loss = self._forward_loss(ids)
                (loss / self.grad_accum).backward()
                step_loss += loss.item() / self.grad_accum
                self.windows_consumed += self.microbatch
                self.tokens.add(counts, self.seq_len)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.opt_step += 1
            if on_step is not None:
                on_step(self, step_loss)

    # ---- evaluation ------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, val: "PackedLike", max_windows: Optional[int] = None) -> float:
        """Token-weighted mean cross-entropy (nats) over a packed validation set."""
        self.model.eval()
        n = len(val) if max_windows is None else min(len(val), max_windows)
        loss_sum = 0.0
        tok_sum = 0
        i = 0
        while i < n:
            b = min(self.microbatch, n - i)
            import numpy as np
            rows = np.stack([np.asarray(val[j], dtype=np.int64) for j in range(i, i + b)])
            ids = torch.from_numpy(rows).to(self.device)
            with self._autocast():
                logits = self.model(input_ids=ids).logits
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
            shift_labels = ids[:, 1:].reshape(-1)
            ls = F.cross_entropy(shift_logits, shift_labels, reduction="sum",
                                 ignore_index=-100)
            loss_sum += ls.item()
            tok_sum += int((shift_labels != -100).sum().item())
            i += b
        self.model.train()
        return loss_sum / max(1, tok_sum)

    # ---- resumable state -------------------------------------------------
    def training_state(self) -> dict:
        """Exact resumable blob: fp32 model+optimizer, cursor, counters, all RNG."""
        return {
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.opt.state_dict(),
            "opt_step": self.opt_step,
            "windows_consumed": self.windows_consumed,
            "tokens": self.tokens.as_dict(),
            "best_val": self.best_val,
            "no_improve_evals": self.no_improve_evals,
            "total_steps": self.total_steps,
            "rng": capture_rng(),
        }

    def load_training_state(self, blob: dict) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in blob["model"].items()})
        self.opt.load_state_dict(blob["optimizer"])
        self.opt_step = int(blob["opt_step"])
        self.windows_consumed = int(blob["windows_consumed"])
        t = blob["tokens"]
        self.tokens = TokenCounters(total=t["total"], c4=t["c4"], mp=t["mp"],
                                    chempile=t["chempile"])
        self.best_val = blob.get("best_val")
        self.no_improve_evals = int(blob.get("no_improve_evals", 0))
        self.total_steps = int(blob.get("total_steps", self.total_steps))
        if "rng" in blob:
            restore_rng(blob["rng"])


# structural typing hint only
class PackedLike:  # pragma: no cover - documentation alias
    def __len__(self) -> int: ...
    def __getitem__(self, i: int): ...
