# DATA_AUDIT.md — Deterministic data construction (Figure 3a reproduction)

Status: **pipeline implemented + unit-tested + smoke-validated end-to-end**; full
tokenization not yet run (launched detached after the pre-training readiness audit).
Frozen config: `configs/data/datasets.json`. Code: `src/j_pretrain/data/`. Tests:
`tests/test_data.py` (29 tests, all passing). Driver: `scripts/build_datasets.py`.

## 1. Datasets, roles, pinned revisions
| Role | Corpus | HF id (name) | Revision (SHA) | Provenance |
|---|---|---|---|---|
| D_pre | C4 | `allenai/c4` (`en`) | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | EXPLICIT_PAPER corpus; HF id INFERRED |
| D_post | MusicPile | `m-a-p/MusicPile` | `5930bd7199ddead76ab540901e2187ea4e5fc6a7` | EXPLICIT_PAPER corpus; HF id INFERRED |
| D_ft | ChemPile | `jablonkagroup/chempile-education` | `c653c3c7bbf599030b80f117b6748ebae406043b` | EXPLICIT_PAPER corpus; HF id + config set INFERRED/LOCAL |

Revisions resolved via `HfApi.repo_info` and frozen; the driver passes `revision=`
on every `load_dataset` so the exact commit is pinned.

## 2. Tokenizer (frozen)
- SmolLM2-135M BPE, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, saved **into the
  repo** at `configs/data/tokenizer/` (5 files, 4.79 MB) and loaded only from that local
  copy — never the network. Dir SHA-256 `658eb5b9…af525` (canonical `tokenizer_sha256`,
  name+bytes; the identical frozen files also bytes-only-hash to `b4ec3f78…a301c4` — reconciled
  to the canonical function, DECISIONS.md).
- vocab 49152, bos=eos=0 (`<|endoftext|>`), pad=None. One tokenizer across all stages.
- Token dtype `uint16` (vocab < 65536). Determinism covered by `test_...tokenizer` +
  `encode_document` (add_special_tokens=False; a single EOS appended per document).
- Provenance: INFERRED_FROM_SMOLLM2 (paper only says "SmolLM2-style").

## 3. Packing (`data/packing.py`)
Documents are tokenized, EOS-terminated, concatenated and cut into **non-overlapping**
windows of exactly `seq_len=1024`. Trailing remainder < seq_len is dropped and counted,
so every per-source token count is an exact multiple of 1024. No window straddles a split
(splitting is per-document, before packing). Tests: exact counts, boundary/remainder,
determinism, no-overlap.

## 4. Shards (`data/shards.py`)
Each `(source, split)` is a directory of `shard-NNNNN.npy` (2-D uint16 `[n_seqs,1024]`)
+ `manifest.json` recording per-shard SHA-256, revision, tokenizer hash, and exact token
counts. Writes are atomic (tmp → fsync → rename). Reads memory-map shards (corpus never
resident in RAM). Tests: roundtrip, mmap indexing, checksum verify + corruption detection.

## 5. Splits (train / tune / val) — disjoint by construction
- **C4**: native `train` and `validation` HF splits (cleanest; no overlap possible).
- **MusicPile, ChemPile**: single-split corpora → carve by **document-hash per-mille bands**
  (`data/splits.py`): `val=[0,5)`, `tune=[5,10)`, `train=[10,1000)` on
  `sha256(salt‖source@rev:doc_id)`. Deterministic, disjoint, salted per corpus.
- Tests: determinism, disjointness, band ratios (~0.5%/0.5%/99%), salt sensitivity.
- **No MusicPile-val leakage**: the 300M subset is a prefix of the *train* pool only; val
  is a disjoint hash band. **No ChemPile leakage into Stage 1/2**: the Stage-1 schedule
  only references `c4`/`mp` sources (test `test_no_chempile_in_stage1_schedule`); ChemPile
  is used solely in Stage 3.

## 6. D_post = 300M MusicPile subset (`data/subset.py`)
- Subset = window prefix `[0, N)` of the deterministically-ordered packed MusicPile-train
  stream. `N = 300_000_000 // 1024 = 292_968` windows = **299,999,232 tokens** (floored to
  whole windows; exact count recorded). Same window range used for Stage-1 exposure and
  Stage-2 post-training → byte-identical.
- Sizes are **nested prefixes** (30M ⊂ 150M ⊂ 300M) though only 300M is used here.
- Requires MusicPile train pool ≫ 300M tokens (MusicPile total ≈ 4.16B; ample headroom for
  a disjoint val split). Driver targets 320M train tokens + 20M val margin; fails loudly if
  the pool is short.
- LOCAL_REPRODUCTION_CHOICE (matches paper's "300M = full pool used"; nesting UNKNOWN in paper).

## 7. Lambda scheduling (`data/interleave.py`) — Figure 3a ADD policy
- Stage 1 = **fixed 8.7B C4** (8,496,093 windows = 8,699,999,232 tokens, identical for every
  lambda) **+ λ·300M MusicPile added on top** (NOT compute-matched 3b, NOT C4 replacement).
- Realized exact MusicPile exposure per lambda (whole windows):

  | λ | MP windows | MP tokens | total upstream tokens |
  |---|---|---|---|
  | 0.00 | 0 | 0 | 8,699,999,232 |
  | 0.25 | 73,242 | 74,999,808 | 8,774,999,040 |
  | 0.50 | 146,484 | 149,999,616 | 8,849,998,848 |
  | 0.75 | 219,726 | 224,999,424 | 8,924,998,656 |
  | 1.00 | 292,968 | 299,999,232 | 8,999,998,464 |

  (λ=1 total ≈ 9.0B upstream, consistent with the paper's loose "~10B" prose; both figures
  disclosed.)
- MusicPile windows exposed = **prefix** `[0, n_mp)` of the 300M subset → nested across λ,
  each selected token shown **at most once** in Stage 1.
- Interleave = **centered even distribution**: MP window `j` at slot `floor((j+0.5)·T/n_mp)`.
  Uniform across all of Stage 1 (never front/back-loaded; first & last Stage-1 windows are
  C4). **C4 relative order is byte-identical across every lambda** (test
  `test_interleave_c4_order_invariant_across_lambda`) — the only difference between λ
  conditions is the inserted MusicPile windows. Front-loading vs uniform: paper is silent;
  uniform is the documented LOCAL choice (mission default), flagged as a limitation.
- Tests: exact per-λ token allocation, C4 budget fixed across λ, exact source counts,
  uniform spacing, C4-order invariance, prefix consumption.

## 8. Probes (`data/probes.py`)
- First **256 windows** of each corpus's **val** split → frozen tiny packed datasets under
  `artifacts/probes/{c4,musicpile,chempile}/` + `probe_manifest.json` (per-shard SHA-256).
  Same probe token IDs for every checkpoint (deterministic prefix). Test: determinism +
  size guard. Interpretability analysis on probes is out of scope (preserve only).

## 9. Open items / limitations
- MusicPile & ChemPile pool sizes confirmed only at smoke scale so far; the full driver
  asserts sufficiency and fails loudly (→ BLOCKED) if a pool is short of budget.
- ChemPile config set = all 4 `chempile-education` configs concatenated (deterministic
  order) — LOCAL choice; paper only says "ChemPile, 0.3B chemistry".
- λ MusicPile exposure = uniform interleave, prefix subset — LOCAL choices where the paper
  is silent; both disclosed here and to be re-listed in FINAL_REPORT limitations.

## 10. Independent audit
This document + the data code/tests are to be independently reviewed as part of
`reports/PRETRAIN_READINESS_AUDIT.md` before full Stage-1 training begins.
