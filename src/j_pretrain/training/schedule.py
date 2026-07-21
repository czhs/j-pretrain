"""Learning-rate schedule and checkpoint-milestone schedules.

* :func:`cosine_lr` — linear warmup to ``peak_lr`` over ``warmup_steps`` then
  cosine decay to ``min_lr`` at ``total_steps``. Deterministic, pure.
* :func:`milestone_token_marks` — the exact per-stage token thresholds at which an
  analysis snapshot or a full resumable checkpoint must be written, per the mission
  document's fixed schedules. Used by the trainer to decide when to checkpoint.
"""
from __future__ import annotations

import math
from typing import Iterable


def cosine_lr(step: int, peak_lr: float, min_lr: float, warmup_steps: int,
              total_steps: int) -> float:
    """LR at optimizer ``step`` (0-indexed) for linear-warmup + cosine-decay."""
    if step < warmup_steps:
        # linear warmup from 0 -> peak (step 0 => first nonzero after +1 convention)
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cos


def _crossings(marks: Iterable[int], prev_tokens: int, new_tokens: int) -> list[int]:
    """Which ``marks`` fall in ``(prev_tokens, new_tokens]`` (i.e. just crossed)."""
    return [m for m in marks if prev_tokens < m <= new_tokens]


# --- Analysis-snapshot token schedules (cumulative tokens of the stage's driver source) ---
# Stage 1 driver = total tokens; Stage 2 driver = MusicPile tokens; Stage 3 = ChemPile tokens.

def analysis_marks(stage: str, max_tokens: int) -> list[int]:
    """Sorted cumulative-token marks at which an analysis snapshot is required."""
    M = 1_000_000
    if stage == "stage1":
        base = [1 * M, 3 * M, 10 * M, 30 * M, 100 * M]
        step = 250 * M
    elif stage == "stage2":
        base = [1 * M, 3 * M, 10 * M, 30 * M, 100 * M]
        step = 50 * M
    elif stage == "stage3":
        base = [1 * M, 3 * M, 10 * M]
        step = 10 * M
    else:
        raise ValueError(stage)
    marks = [m for m in base if m <= max_tokens]
    k = (marks[-1] if marks else 0) + step
    while k <= max_tokens:
        marks.append(k)
        k += step
    return sorted(set(marks))


def resumable_marks(stage: str, max_tokens: int) -> list[int]:
    """Sorted cumulative-token marks at which a full resumable checkpoint is required."""
    M = 1_000_000
    if stage == "stage1":
        step = 500 * M
    elif stage == "stage2":
        step = 100 * M
    elif stage == "stage3":
        step = 50 * M
    else:
        raise ValueError(stage)
    marks, k = [], step
    while k <= max_tokens:
        marks.append(k)
        k += step
    return marks


def crossed_analysis(stage: str, max_tokens: int, prev_tokens: int,
                     new_tokens: int) -> list[int]:
    return _crossings(analysis_marks(stage, max_tokens), prev_tokens, new_tokens)


def crossed_resumable(stage: str, max_tokens: int, prev_tokens: int,
                      new_tokens: int) -> list[int]:
    return _crossings(resumable_marks(stage, max_tokens), prev_tokens, new_tokens)
