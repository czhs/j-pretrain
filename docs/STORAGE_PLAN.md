# STORAGE_PLAN.md

Status: **MEASURED** (real 135M checkpoints serialized; see `scripts/bench_storage.py`).
Absolute-threshold disk gate applies (host drive large + mostly full by unrelated data).

## Filesystem
`/dev/sda2` (ext4-class), mounted `/`. **581 GB free** / 3.5 TB total / 83% used at measurement.
Artifact root: `J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`
(same filesystem as repo). No second storage destination configured →
backup status = `unreplicated_local_copy` (recorded truthfully; no external upload without user auth).

## Measured checkpoint sizes (real SmolLM2-135M, 134,515,008 params)
| class | contents | on-disk |
|-------|----------|---------|
| **Analysis snapshot** | bf16 safetensors weights + config.json + metadata + checksums | **310.6 MB** |
| **Full resumable** | analysis contents + `training_state.pt` (fp32 model + AdamW exp_avg/exp_avg_sq + RNG + cursor) | **1850.3 MB** |

Notes:
- Analysis weights are 310.6 MB not the naive 269 MB because `safetensors` materialises the **tied**
  embedding twice (`embed_tokens.weight` + `lm_head.weight`, +56 MB). Kept as-is: load-independence
  and "complete unquantized weights" outweigh a 56 MB/snapshot saving; disclosed here.
- Resumable = fp32 model (538 MB) + fp32 AdamW state (2×538 MB) + bf16 analysis copy (310 MB) ≈ 1.85 GB.

## Serialization overhead (per checkpoint, estimate)
Analysis write+fsync+sha256+load-test ≈ 2–4 s; resumable ≈ 15–25 s. Temp-dir atomic write doubles the
transient footprint of the *one* checkpoint being written (≤ ~2 GB transient), never of the archive.

## Permanent-artifact projection (locked scope: 5 λ, D_post=300M)
Counts follow the **mandatory** analysis + resumable schedules (mission doc), never weakened.

| bucket | count | unit | subtotal |
|--------|-------|------|----------|
| Analysis snapshots (Stage1 per-λ marks+init+LR-bounds+final; Stage2 marks+best+incoming+restored; Stage3 marks+incoming+final) | 579 | 310.6 MB | **175.6 GB** |
| Permanent resumables (shared init + 7/run × 5: s1-final, s2-{incoming,best,final,restored}, s3-{incoming,final}) | 36 | 1850 MB | 65.1 GB |
| Rolling resumables (keep ≥2 most-recent per active run; 1 active at a time) | 2 | 1850 MB | 3.6 GB |
| Tokenized datasets (C4 8.7B + MP 320M + ChemPile 220M + vals, uint16) | — | — | ~19 GB |
| Logs / metrics / inventories / probes | — | — | ~5 GB |
| **Subtotal** | | | **~268 GB** |
| **+15% safety headroom** | | | **~308 GB** |

## Gate result
Projected permanent storage **~308 GB (with headroom) < 581 GB free** → **STORAGE GATE: PASS**.
Post-experiment free space ≈ 581 − 308 ≈ **273 GB** (> 250 GB planning threshold).

Disk-pressure protocol (absolute free-space thresholds, per mission): recompute at 250 GB; gate new
runs at 150 GB; pause + `BLOCKED.md` at 75 GB. Never delete analysis snapshots, permanently-retained
resumables, logs, metrics, manifests, probes, or checksums to relieve pressure — only regenerable
package/download caches, each removal recorded.

## Resumable retention policy (recap; enforced by the inventory + orchestrator)
Permanently retain: init (shared), each run's Stage-1 final, Stage-2 {incoming, best, final,
restored}, Stage-3 {incoming, final}. Prune other intermediates only once a newer load-validated
resumable exists (keep ≥2 most-recent per active run); every prune is an append-only superseding
inventory record. If an intermediate also carries an analysis milestone, its bf16 weights are saved
as an analysis snapshot before the optimizer state is pruned.
