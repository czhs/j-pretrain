# COMPUTE_PLAN.md

Status: **PRELIMINARY** (early fail-fast benchmark). Final numbers require the real training loop
(data loader, checkpoint serialization, eval) and will be folded into `reports/FEASIBILITY.md`
before scope lock. Recorded here so the feasibility direction is durable.

## Hardware (see docs/ENVIRONMENT.md)
RTX 4090 24 GB (24564 MiB), driver 535.309.01, CUDA driver 12.2. i9-13900K (32 threads), 62 GiB RAM.
torch 2.5.1+cu121, transformers 4.46.3. Root fs /dev/sda2: 590 GB free / 83% used (absolute-threshold
storage gate applies; 590 GB > 250 GB checkpoint-planning threshold).

## Model (verified this iteration)
SmolLM2-135M config, random init (LlamaForCausalLM). Local build = **134.52M params** (tied
embeddings). Matches "~135M" spec. Training seq len 1024.

## Throughput benchmark (scripts/bench_throughput.py; random data, seq_len=1024)
Measured over 40 steps after 15 warmup. Memory is dominated by the fp32 cross-entropy over the
49152-token vocab (~1.74 GB/microbatch in eager).

| mode | microbatch | tok/s | s/step | peak alloc GB | peak reserved GB |
|------|-----------|-------|--------|---------------|------------------|
| eager (SDPA) | 4  | 48,260 | 0.085 | 8.90  | 9.51  |
| eager (SDPA) | 8  | 52,497 | 0.156 | 15.84 | 16.58 |
| eager (SDPA) | 12 | 51,283 | 0.240 | 22.91 | 24.38 (near cap) |
| eager (SDPA) | 16 | OOM    | —     | —     | — |
| **compile (default)** | **8** | **94,255** | **0.087** | **9.39** | **9.65** |

Key findings:
- `torch.compile` gives **1.80× throughput** (52.5k→94.3k tok/s) AND cuts reserved memory 16.6→9.65 GB
  (it fuses the large-vocab CE, avoiding the fp32 logits blow-up). Chosen as the default execution
  mode; eager SDPA is the stable fallback.
- With compile at only 9.65 GB reserved for mb=8, there is ample headroom to raise microbatch or add
  eval memory; effective global batch will be set by grad-accumulation to the frozen scientific value.
- ~1.5–2 GB headroom rule easily satisfied by compile@mb8 (≈14 GB free).

## Preliminary experiment-duration projection (pure compute, before overhead)
Token totals under the locked scope (5 λ conditions, D_post=300M):
- Stage 1: 5 × 8.7B C4 + (0+0.25+0.5+0.75+1.0)×300M MusicPile = 43.5B + 0.75B = **44.25B tokens**.
- Stage 2: ≤ 5 × 2B (early stopping usually less) = **≤10B tokens** (worst case).
- Stage 3: 5 × 200M = **1.0B tokens**.
- Grand total ≤ **55.25B tokens**.

| throughput | Stage1 | Stage2 (≤) | Stage3 | total (pure) |
|-----------|--------|-----------|--------|--------------|
| 94k tok/s (compile) | 5.45 d | 1.23 d | 0.13 d | **~6.8 d** |
| 52k tok/s (eager fallback) | 9.85 d | 2.22 d | 0.22 d | **~12.3 d** |

Add ~20–30% for checkpoint serialization (frequent per schedule), periodic eval sweeps, data-loader
warmup, and orchestration gaps → realistic **~8–16 days**. Both bracket ends are **< 21-day gate**.

## Autotuning scope (execution-only, identical scientific HPs across λ)
Confirmed defaults: bf16 autocast, SDPA attention, fused AdamW, torch.compile(default). To finalize
in real impl: microbatch (≥8), grad-accumulation to hit scientific global batch, dataloader workers,
persistent workers, prefetch, memory-mapped shards. Gradient checkpointing NOT needed (memory ample
under compile). If real-loop compile is unstable, fall back to eager@mb8 (still passes the gate).

## Open items before final COMPUTE_PLAN / FEASIBILITY
1. Re-benchmark with the real data loader + periodic eval + checkpoint writes (end-to-end tok/s).
2. Measure real analysis-snapshot and full-resumable checkpoint sizes (→ STORAGE_PLAN.md).
3. Confirm compile stability over long runs; record recompilation triggers.
