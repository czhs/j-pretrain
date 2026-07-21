"""Assemble training microbatches from a plan + packed shards, addressed by cursor.

The loader is deterministic and stateless beyond the cursor the trainer passes in:
``windows(g_start, n)`` returns the next ``n`` windows starting at global index
``g_start`` as an int64 tensor ``[n, seq_len]`` plus exact per-source window counts.
No worker threads, no internal cursor — resumability lives entirely in the trainer's
integer cursor because the plan is a pure function (see :mod:`.dataplan`).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from j_pretrain.data.shards import PackedDataset


class PlanLoader:
    SOURCES = ("c4", "mp", "chempile")

    def __init__(self, plan, sources: Mapping[str, PackedDataset], seq_len: int):
        self.plan = plan
        self.sources = dict(sources)
        self.seq_len = int(seq_len)
        for name, ds in self.sources.items():
            if ds.seq_len != self.seq_len:
                raise ValueError(f"source {name} seq_len {ds.seq_len} != {self.seq_len}")

    def windows(self, g_start: int, n: int) -> tuple[torch.Tensor, dict[str, int]]:
        """Return ``(input_ids[n, seq_len] int64, per_source_window_counts)``."""
        rows = np.empty((n, self.seq_len), dtype=np.int64)
        counts = {s: 0 for s in self.SOURCES}
        for i in range(n):
            source, local = self.plan.at(g_start + i)
            ds = self.sources.get(source)
            if ds is None:
                raise KeyError(f"plan referenced unknown source {source!r}")
            rows[i] = np.asarray(ds[local], dtype=np.int64)
            counts[source] += 1
        return torch.from_numpy(rows), counts

    @property
    def plan_len(self) -> int:
        return len(self.plan)
