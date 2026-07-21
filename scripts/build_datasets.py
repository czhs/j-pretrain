#!/usr/bin/env python
"""Deterministic streaming tokenize + pack driver for C4 / MusicPile / ChemPile.

Streams each corpus (never materialising it in RAM), tokenizes with the frozen
SmolLM2 tokenizer, assigns documents to train/tune/val, packs into fixed
``seq_len`` windows and writes memory-mapped ``uint16`` shards under the artifact
root. Also records the 300M MusicPile subset reference and builds fixed probes.

Design notes
------------
* C4 uses its native ``train`` / ``validation`` splits (no hash carve needed);
  MusicPile and ChemPile are single-split, so val/tune are carved by document
  hash (disjoint by construction — see j_pretrain.data.splits).
* Only as many C4 shards as needed to cover the 8.7B train budget (+val margin)
  are downloaded — the full C4 corpus is never mirrored.
* Resumable at the (source, split) granularity: an existing complete manifest is
  skipped. Partial shard writes use atomic rename so a crash never leaves a
  corrupt shard in a finalized manifest.

Run detached (tmux) with the env python. ``--smoke`` uses tiny budgets to
validate the end-to-end path in seconds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from datasets import load_dataset

from j_pretrain.data.packing import PackStats, pack_documents
from j_pretrain.data.shards import MANIFEST_NAME, PackedDataset, ShardWriter
from j_pretrain.data.splits import SPLIT_TRAIN, SplitPolicy, doc_key, split_of
from j_pretrain.data.tokenizer import encode_document, load_tokenizer, tokenizer_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_CFG = REPO_ROOT / "configs" / "data" / "datasets.json"
DEFAULT_ARTIFACT_ROOT = "/home/hshi-j-4090/Desktop/j-pretrain-artifacts"


def artifact_root() -> Path:
    return Path(os.environ.get("J_PRETRAIN_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))


def load_cfg() -> dict:
    return json.loads(DATASETS_CFG.read_text())


# ---------------------------------------------------------------- doc streams
def _c4_stream(cfg: dict, split: str) -> Iterator[tuple[str, str]]:
    s = cfg["sources"]["c4"]
    ds = load_dataset(s["hf_path"], s["hf_name"], split=split,
                      streaming=True, revision=s["revision"])
    for i, ex in enumerate(ds):
        yield (str(ex.get(s["id_key"]) or i), ex[s["text_key"]])


def _musicpile_stream(cfg: dict) -> Iterator[tuple[str, str]]:
    s = cfg["sources"]["musicpile"]
    ds = load_dataset(s["hf_path"], split="train", streaming=True, revision=s["revision"])
    for i, ex in enumerate(ds):
        did = ex.get(s["id_key"]) if s["id_key"] else None
        yield (str(did if did is not None else i), ex[s["text_key"]])


def _chempile_stream(cfg: dict) -> Iterator[tuple[str, str]]:
    s = cfg["sources"]["chempile"]
    for conf in s["configs"]:  # deterministic config order
        ds = load_dataset(s["hf_path"], conf, split="train",
                          streaming=True, revision=s["revision"])
        for i, ex in enumerate(ds):
            yield (f"{conf}:{i}", ex[s["text_key"]])


# ---------------------------------------------------------------- packing driver
def _pack_split(
    docs: Iterable[tuple[str, str]],
    out_dir: Path,
    seq_len: int,
    max_tokens: int | None,
    base_meta: dict,
) -> dict:
    """Tokenize a document stream, pack, write shards, stop at ``max_tokens``."""
    if (out_dir / MANIFEST_NAME).exists():
        print(f"[skip] {out_dir} already complete")
        return json.loads((out_dir / MANIFEST_NAME).read_text())
    tok = load_tokenizer()
    stats = PackStats()
    n_docs = [0]

    def _token_docs() -> Iterator[list[int]]:
        for did, text in docs:
            n_docs[0] += 1
            yield encode_document(text, tok, add_eos=True)

    writer = ShardWriter(out_dir, seq_len=seq_len, meta=base_meta)
    for window in pack_documents(_token_docs(), seq_len, stats=stats):
        writer.add(window)
        if max_tokens is not None and stats.n_tokens_emitted >= max_tokens:
            break
    manifest = writer.finalize(extra={
        "packstats": {
            "n_documents_read": n_docs[0],
            "n_tokens_consumed": stats.n_tokens_consumed,
            "n_tokens_emitted": stats.n_tokens_emitted,
        },
    })
    print(f"[done] {out_dir}: {manifest['n_seqs']} seqs, {manifest['n_tokens']} tokens")
    return manifest


def _hash_filtered(docs: Iterable[tuple[str, str]], source: str, revision: str,
                   policy: SplitPolicy, keep_split: str) -> Iterator[tuple[str, str]]:
    for did, text in docs:
        if split_of(doc_key(source, revision, did), policy) == keep_split:
            yield (did, text)


def build(smoke: bool) -> None:
    cfg = load_cfg()
    seq_len = cfg["seq_len"]
    root = artifact_root() / ("smoke" if smoke else "datasets")
    root.mkdir(parents=True, exist_ok=True)
    b = cfg["budgets"]
    scale = 1
    if smoke:
        c4_train = 2_000_000
        mp_train = 3_000_000
        chem_train = 1_000_000
        val_tokens = 200_000
    else:
        c4_train = b["stage1_c4_tokens"]
        mp_train = b["d_post_tokens"] + 20_000_000  # subset 300M + margin so val/pool safe
        chem_train = b["stage3_chempile_tokens"] + 20_000_000
        val_tokens = 20_000_000
    pol = lambda src: SplitPolicy(val_pm=cfg["split_policy"]["val_pm"],
                                  tune_pm=cfg["split_policy"]["tune_pm"], salt=src)
    common = dict(tokenizer_sha256=tokenizer_sha256(), seq_len=seq_len)

    # C4: native splits
    c4 = cfg["sources"]["c4"]
    _pack_split(_c4_stream(cfg, "train"), root / "c4" / "train", seq_len, c4_train,
                {**common, "source": "c4", "split": "train", "revision": c4["revision"]})
    _pack_split(_c4_stream(cfg, "validation"), root / "c4" / "val", seq_len, val_tokens,
                {**common, "source": "c4", "split": "val", "revision": c4["revision"]})

    # MusicPile: hash carve
    mp = cfg["sources"]["musicpile"]
    _pack_split(_hash_filtered(_musicpile_stream(cfg), "musicpile", mp["revision"], pol("musicpile"), SPLIT_TRAIN),
                root / "musicpile" / "train", seq_len, mp_train,
                {**common, "source": "musicpile", "split": "train", "revision": mp["revision"]})
    _pack_split(_hash_filtered(_musicpile_stream(cfg), "musicpile", mp["revision"], pol("musicpile"), "val"),
                root / "musicpile" / "val", seq_len, val_tokens,
                {**common, "source": "musicpile", "split": "val", "revision": mp["revision"]})

    # ChemPile: hash carve over concatenated configs
    ch = cfg["sources"]["chempile"]
    _pack_split(_hash_filtered(_chempile_stream(cfg), "chempile", ch["revision"], pol("chempile"), SPLIT_TRAIN),
                root / "chempile" / "train", seq_len, chem_train,
                {**common, "source": "chempile", "split": "train", "revision": ch["revision"]})
    _pack_split(_hash_filtered(_chempile_stream(cfg), "chempile", ch["revision"], pol("chempile"), "val"),
                root / "chempile" / "val", seq_len, val_tokens,
                {**common, "source": "chempile", "split": "val", "revision": ch["revision"]})

    print("[build_datasets] complete under", root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny budgets for path validation")
    build(ap.parse_args().smoke)


if __name__ == "__main__":
    main()
