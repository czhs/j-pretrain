"""Frozen tokenizer access.

The tokenizer is the single frozen SmolLM2-135M BPE, saved into the repo at
``configs/data/tokenizer`` (revision recorded in ``configs/data/datasets.json``).
Loading from the local frozen copy — never from the network — guarantees byte
identical tokenization across every stage, run and machine.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_DIR = REPO_ROOT / "configs" / "data" / "tokenizer"

# SmolLM2 uses <|endoftext|> (id 0) as both BOS and EOS; there is no pad token.
EOS_ID = 0


@lru_cache(maxsize=1)
def load_tokenizer() -> PreTrainedTokenizerBase:
    """Load the frozen local tokenizer (cached)."""
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    assert tok.vocab_size == 49152, f"unexpected vocab_size {tok.vocab_size}"
    assert tok.eos_token_id == EOS_ID, f"unexpected eos {tok.eos_token_id}"
    return tok


def tokenizer_sha256() -> str:
    """Stable hash of the frozen tokenizer directory (all files, sorted)."""
    h = hashlib.sha256()
    for f in sorted(TOKENIZER_DIR.iterdir()):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def encode_document(text: str, tok: PreTrainedTokenizerBase | None = None,
                    add_eos: bool = True) -> list[int]:
    """Deterministically encode one document, optionally appending the EOS separator.

    Special tokens are NOT auto-added by the BPE; the EOS we append is the only
    document boundary marker, matching standard packed-LM preprocessing.
    """
    tok = tok or load_tokenizer()
    ids = tok(text, add_special_tokens=False).input_ids
    if add_eos:
        ids = ids + [EOS_ID]
    return ids


def encode_stream(texts: Iterable[str], tok: PreTrainedTokenizerBase | None = None,
                  add_eos: bool = True) -> Iterable[list[int]]:
    """Yield per-document token-id lists for a stream of texts."""
    tok = tok or load_tokenizer()
    for t in texts:
        yield encode_document(t, tok, add_eos=add_eos)
