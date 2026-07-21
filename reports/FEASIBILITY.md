# FEASIBILITY.md — pre-scope-lock gate

Status: **PASS on both axes** → proceed to lock scope and continue autonomously.
This gate exists because the loop runs unattended; the multi-week commitment is a number the user has
now seen (below). A large-but-sub-threshold projection is NOT a reason to reduce scope.

## Inputs (all measured this project, not guessed)
- Throughput (`scripts/bench_throughput.py`, real 135M, seq 1024): **compile default 94,255 tok/s**
  (9.65 GB reserved); **eager SDPA fallback 52,497 tok/s** @ mb=8. ~14 GB headroom under compile.
- Checkpoint sizes (`scripts/bench_storage.py`, real 135M): analysis **310.6 MB**, resumable **1850 MB**.
- Free disk: **581 GB** on the artifact filesystem.

## Locked-scope token budget (5 λ conditions, D_post=300M)
- Stage 1: 5 × 8.7B C4 + (0+0.25+0.5+0.75+1.0)×300M MP = 43.5B + 0.75B = **44.25B**.
- Stage 2: ≤ 5 × 2B (early stopping usually less) = **≤ 10B** worst case.
- Stage 3: 5 × 200M = **1.0B**.
- **Total ≤ 55.25B tokens.**

## Wall-clock projection (incl. overhead)
Pure compute: 55.25B / 94,255 = **6.8 d** (compile) … 55.25B / 52,497 = **12.3 d** (eager).
Overhead — checkpoint serialization (579 analysis ≈ 0.5 h + 38 resumable ≈ 0.2 h), periodic eval
sweeps on small val sets, data-loader warmup, orchestration/usage-limit gaps — add ~15–30%:

| execution mode | projected total |
|----------------|-----------------|
| compile (default) | **~8 days** |
| eager (fallback)  | **~14–16 days** |

Both ends are **< 21-day gate**. (Usage-limit idle time is orchestration overhead only; training is
detached and survives gaps — never counted against the compute budget.)

## Storage projection (see docs/STORAGE_PLAN.md)
Permanent artifacts **~268 GB**; **~308 GB with 15% headroom** < **581 GB free** → fits, leaving
~273 GB (> 250 GB threshold).

## Gate decision
```
wall_clock_projection (<= 21 days)  : PASS  (~8 d compile / ~14–16 d eager)
permanent_storage (fits + 15% head) : PASS  (~308 GB < 581 GB free)
=> LOCK SCOPE and proceed autonomously. No BLOCKED.md required.
```

## Disclosed reductions / limitations (also in every report)
- **Reduced Fig 3a**: only the **300M** MusicPile subset (paper sweeps 30M/150M/300M). User-authorized
  (`state/SCOPE_LOCK.json`). This is Fig 3a (early-exposure), **not** compute-matched Fig 3b.
- One seed per condition (label as one-seed reproduction; no invented CIs).
- Several Stage-1/2/3 optimizer knobs are LOCAL_REPRODUCTION_CHOICE (no official code found); the fixed
  Stage-2 Fig-3a triple is provisional pending a λ=0 pilot. All labeled in docs/PAPER_SPEC.md.
