"""Stage-1 lambda mixture schedule (Figure 3a policy).

Stage 1 shows a **fixed** C4 budget (8.7B tokens) plus ``lambda * |D_post|``
MusicPile tokens *added on top* (NOT compute-matched Figure 3b, NOT C4
replacement). The MusicPile tokens are the first ``lambda`` fraction of the
frozen 300M subset — a prefix, so exposure is nested across lambda and each
selected token is shown at most once.

Interleaving uses an even (Bresenham) distribution so MusicPile windows are
spread uniformly across the whole of Stage 1 — never front- or back-loaded — and
the relative order of C4 windows is byte-identical across every lambda (only the
inserted MusicPile windows differ). Everything here is pure and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class LambdaPlan:
    lambda_frac: float
    seq_len: int
    n_c4_windows: int
    n_mp_windows: int

    @property
    def c4_tokens(self) -> int:
        return self.n_c4_windows * self.seq_len

    @property
    def mp_tokens(self) -> int:
        return self.n_mp_windows * self.seq_len

    @property
    def total_windows(self) -> int:
        return self.n_c4_windows + self.n_mp_windows

    @property
    def total_tokens(self) -> int:
        return self.total_windows * self.seq_len


def lambda_plan(
    lambda_frac: float,
    c4_budget_tokens: int,
    subset_tokens: int,
    seq_len: int,
) -> LambdaPlan:
    """Exact per-source window/token allocation for one lambda condition."""
    if not (0.0 <= lambda_frac <= 1.0):
        raise ValueError(f"lambda {lambda_frac} out of [0,1]")
    n_c4 = c4_budget_tokens // seq_len
    subset_windows = subset_tokens // seq_len
    n_mp = round(lambda_frac * subset_windows)
    return LambdaPlan(lambda_frac, seq_len, n_c4, n_mp)


def interleave_schedule(n_c4: int, n_mp: int) -> Iterator[str]:
    """Yield ``"c4"`` / ``"mp"`` tags, evenly interleaved.

    Emits exactly ``n_c4`` ``"c4"`` tags and ``n_mp`` ``"mp"`` tags. MusicPile
    window ``j`` is centered in its 1/n_mp band at slot ``floor((j+0.5)*T/n_mp)``,
    so the exposures are uniform (never front- or back-loaded, and the final Stage-1
    window is C4, not MusicPile). Because C4 tags simply fill the remaining slots in
    order, the relative C4 order is identical for every ``n_mp`` (every lambda).
    """
    total = n_c4 + n_mp
    if total == 0:
        return
    if n_mp == 0:
        for _ in range(total):
            yield "c4"
        return
    # floor((j+0.5)*total/n_mp); strictly increasing (total>=n_mp) => n_mp distinct slots.
    mp_positions = {((2 * j + 1) * total) // (2 * n_mp) for j in range(n_mp)}
    for s in range(total):
        yield "mp" if s in mp_positions else "c4"


def schedule_source_indices(n_c4: int, n_mp: int) -> Iterator[tuple[str, int]]:
    """Yield ``(source, window_index)`` pairs following the even interleave.

    ``window_index`` is the position within that source's own ordered stream
    (C4 windows 0..n_c4-1 in order; MusicPile subset windows 0..n_mp-1 in order).
    """
    ci = mi = 0
    for tag in interleave_schedule(n_c4, n_mp):
        if tag == "c4":
            yield ("c4", ci)
            ci += 1
        else:
            yield ("mp", mi)
            mi += 1
