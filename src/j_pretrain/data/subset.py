"""Fixed MusicPile D_post subset construction (300M tokens).

D_post is defined as a token-count prefix of the *deterministically ordered*,
packed MusicPile training stream. Because every requested size is a prefix of the
same ordering, the sizes are strictly nested (30M ⊂ 150M ⊂ 300M) — matching the
paper's nested-subset procedure — even though this reproduction only uses 300M.

The subset is a contiguous window range ``[0, n_windows)`` of the packed
MusicPile-train ``PackedDataset``; the same window range is used for both Stage-1
early exposure and Stage-2 post-training, so they are byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubsetSpec:
    target_tokens: int
    seq_len: int

    @property
    def n_windows(self) -> int:
        """Whole windows whose total token count does not exceed ``target_tokens``."""
        return self.target_tokens // self.seq_len

    @property
    def exact_tokens(self) -> int:
        return self.n_windows * self.seq_len


def subset_window_range(target_tokens: int, seq_len: int, pool_windows: int) -> range:
    """Return ``range(0, n)`` of window indices for the first ``target_tokens`` tokens.

    Raises if the packed pool is too small to satisfy the requested size.
    """
    spec = SubsetSpec(target_tokens, seq_len)
    if spec.n_windows > pool_windows:
        raise ValueError(
            f"pool has {pool_windows} windows ({pool_windows * seq_len} tokens); "
            f"need {spec.n_windows} for {target_tokens} tokens"
        )
    return range(0, spec.n_windows)


def is_nested(smaller_tokens: int, larger_tokens: int, seq_len: int) -> bool:
    """True if the smaller subset's window range is a prefix of the larger's."""
    s = SubsetSpec(smaller_tokens, seq_len).n_windows
    l = SubsetSpec(larger_tokens, seq_len).n_windows
    return s <= l
