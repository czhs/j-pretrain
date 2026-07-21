"""Deterministic document packing into fixed-length sequences.

Documents (already tokenized, each ending in EOS) are concatenated into a single
stream and cut into non-overlapping windows of exactly ``seq_len`` tokens. This is
standard packed causal-LM preprocessing:

* Each emitted window is a training example of exactly ``seq_len`` tokens.
* Windows never overlap, so no token is shown twice within one pass ("at most one
  copy of each selected token").
* The EOS between documents lets the model see document boundaries; loss is still
  computed over every position.
* A trailing remainder shorter than ``seq_len`` is dropped (and counted) so every
  window is full — this keeps per-source token counts exact multiples of seq_len.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

TOKEN_DTYPE = np.uint16  # vocab 49152 < 65536


@dataclass
class PackStats:
    n_windows: int = 0
    n_tokens_emitted: int = 0  # == n_windows * seq_len
    n_tokens_consumed: int = 0  # total tokens seen (incl. dropped remainder)
    n_documents: int = 0
    n_tokens_dropped: int = 0  # trailing remainder < seq_len


def pack_documents(
    docs: Iterable[list[int]],
    seq_len: int,
    stats: PackStats | None = None,
) -> Iterator[np.ndarray]:
    """Yield ``uint16`` windows of length ``seq_len`` from a stream of token lists.

    ``stats`` (if given) is updated in place with exact counts; on generator
    exhaustion the trailing remainder is recorded as dropped.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    buf: list[int] = []
    st = stats if stats is not None else PackStats()
    for ids in docs:
        st.n_documents += 1
        st.n_tokens_consumed += len(ids)
        buf.extend(ids)
        while len(buf) >= seq_len:
            window = np.asarray(buf[:seq_len], dtype=TOKEN_DTYPE)
            del buf[:seq_len]
            st.n_windows += 1
            st.n_tokens_emitted += seq_len
            yield window
    st.n_tokens_dropped = len(buf)


def count_packed_windows(docs: Iterable[list[int]], seq_len: int) -> PackStats:
    """Consume a stream and return exact packing statistics (no arrays kept)."""
    st = PackStats()
    for _ in pack_documents(docs, seq_len, stats=st):
        pass
    return st
