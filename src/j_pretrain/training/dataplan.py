"""Deterministic, index-addressable, resumable sample plans.

A *plan* maps a global window index ``g`` (0-based, monotonically increasing over
the whole stage) to a concrete ``(source, local_window_index)`` to fetch from a
:class:`~j_pretrain.data.shards.PackedDataset`. Because a plan is a **pure
function of g**, resuming training needs only a single integer cursor
(``windows_consumed``): the exact same windows are re-derived, so data order is
byte-identical across an interruption — no dataloader worker state to serialize.

Two plan kinds:

* :class:`Stage1Plan` — the fixed-8.7B-C4 + interleaved-MusicPile schedule for
  Stage 1 (single pass, no shuffle: C4 in packed order identical across lambda,
  MusicPile as the nested prefix of the 300M subset).
* :class:`ShuffledSourcePlan` — a single source cycled over multiple epochs with a
  per-epoch deterministic permutation (Stage 2 over the 300M subset; Stage 3 over
  ChemPile). Set ``shuffle=False`` for a fixed packed order.
"""
from __future__ import annotations

import bisect
from typing import Protocol

import numpy as np

from j_pretrain.data.interleave import interleave_schedule


class SamplePlan(Protocol):
    def at(self, g: int) -> tuple[str, int]: ...
    def __len__(self) -> int: ...


class Stage1Plan:
    """Interleaved C4 + MusicPile plan (single pass over ``n_c4 + n_mp`` windows).

    Mirrors :func:`j_pretrain.data.interleave.schedule_source_indices` exactly but is
    O(log n_mp) per lookup instead of O(g), so the trainer can address any window.
    """

    def __init__(self, n_c4: int, n_mp: int):
        self.n_c4 = int(n_c4)
        self.n_mp = int(n_mp)
        self.total = self.n_c4 + self.n_mp
        # The set of global slots that carry a MusicPile window (sorted, strictly
        # increasing), computed with the same centered-even formula as interleave.
        if self.n_mp > 0:
            self._mp_slots = [((2 * j + 1) * self.total) // (2 * self.n_mp)
                              for j in range(self.n_mp)]
        else:
            self._mp_slots = []

    def __len__(self) -> int:
        return self.total

    def at(self, g: int) -> tuple[str, int]:
        if not (0 <= g < self.total):
            raise IndexError(g)
        k = bisect.bisect_left(self._mp_slots, g)
        if k < self.n_mp and self._mp_slots[k] == g:
            return ("mp", k)          # k-th MusicPile subset window (packed prefix order)
        return ("c4", g - k)          # g minus (#mp slots before g) == C4 packed index

    def verify_against_reference(self) -> bool:
        """Assert O(log n) lookups match the O(n) reference generator (test aid)."""
        ci = mi = 0
        for g, tag in enumerate(interleave_schedule(self.n_c4, self.n_mp)):
            if tag == "c4":
                if self.at(g) != ("c4", ci):
                    return False
                ci += 1
            else:
                if self.at(g) != ("mp", mi):
                    return False
                mi += 1
        return True


class ShuffledSourcePlan:
    """One source cycled over epochs, each epoch a deterministic permutation.

    Epoch ``e`` (``e = g // pool_windows``) uses ``default_rng(seed_base + e)``.
    With ``shuffle=False`` the source is traversed in packed order every epoch.
    ``__len__`` is one epoch; the plan is defined for arbitrarily large ``g``.
    """

    def __init__(self, source: str, pool_windows: int, seed: int, shuffle: bool = True):
        if pool_windows <= 0:
            raise ValueError("pool_windows must be positive")
        self.source = source
        self.pool_windows = int(pool_windows)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._perm_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.pool_windows

    def _perm(self, epoch: int) -> np.ndarray:
        p = self._perm_cache.get(epoch)
        if p is None:
            rng = np.random.default_rng(self.seed * 1_000_003 + epoch)
            p = rng.permutation(self.pool_windows)
            self._perm_cache[epoch] = p
        return p

    def at(self, g: int) -> tuple[str, int]:
        if g < 0:
            raise IndexError(g)
        epoch, pos = divmod(g, self.pool_windows)
        local = int(self._perm(epoch)[pos]) if self.shuffle else pos
        return (self.source, local)


def stage1_plan_from_lambda(n_c4_windows: int, n_mp_windows: int) -> Stage1Plan:
    return Stage1Plan(n_c4_windows, n_mp_windows)
