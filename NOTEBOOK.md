# Scientific Notebook — j-pretrain

Running log for the Figure-3a reproduction **and** the J-space characterization built on top
of it. This file is the project's durable memory: what we are asking, what we have measured,
what broke, and how it was fixed. Append; do not rewrite history.

Machine-readable state stays in `state/experiment_state.json`. This file is the human story.

---

## 1. The question

Two questions, one nested inside the other.

### 1.1 The reproduction (paper Figure 3a)

Paper: **"Early Data Exposure Improves Robustness to Subsequent Fine-Tuning"**
(Feng, Ghosal, Springer, Zhong, Raghunathan — CMU, arXiv:2605.12705v1, 12 May 2026).

Three-stage pipeline on SmolLM2-135M:

| Stage | Data | Measures |
|---|---|---|
| 1 — pretrain | 8.7B C4 **+ λ × 300M MusicPile interleaved** | — |
| 2 — post-train | full 300M MusicPile to convergence (early stop) | `L_im = L(θ_post; D_post)` |
| 3 — fine-tune | 200M ChemPile @ lr 5e-5 | `L_ret = L(θ_ft; D_post)`, `L_ft`, `L_pre` |

λ ∈ {0, 0.25, 0.5, 0.75, 1.0}, seed 1234, one seed per condition.

**The result we are recreating (the "shocking" one).** Figure 3a, left panel: `L_im` is
**nearly flat** in λ — mixing buys you essentially nothing measurable at handoff. Right panel:
`L_ret` **decreases monotonically** in λ — after an unrelated ChemPile fine-tune, the mixed
models have retained substantially more of the MusicPile capability.

The benefit of mixing is **latent**. It is invisible at the moment the capability is acquired
and only materializes after a later, unrelated training stage. Two checkpoints that look
identical on the target domain diverge sharply once someone else fine-tunes them.

This is a deliberately reduced Fig 3a: only the 300M subset (paper sweeps 30M/150M/300M),
one seed. User-authorized, recorded in `state/SCOPE_LOCK.json`. It is **not** Fig 3b
(compute-matched).

### 1.2 The mechanism question (J-space) — the new work

The paper's Section 5 does not stop at the phenomenon; it proposes a *geometric* mechanism,
proved in a two-layer linear model. Input space splits into three feature blocks:

- **invariant** features — identical singular values across D_pre, D_post, D_ft
- **inconsistent** features — shared dimensions where the tasks *disagree*
- **specialized** features — active **only** on D_post, zero covariance under D_pre and D_ft

The three theorems:

- **5.1** Only early exposure learns the specialized features. Linear nets learn features in
  descending singular-value order; without exposure the specialized features have the lowest
  effective singular value and never cross the learning threshold. Mixing an α-fraction boosts
  their effective singular value to ≈ αβ, which can clear the threshold even for small α.
- **5.2** Post-training from θ^mixed *reuses* the specialized features to reduce `L_im`.
  θ^unmixed cannot — those features are absent at init and stay absent — so it can only reduce
  `L_im` by **distorting the inconsistent features**, which are shared with D_pre.
- **5.3** `Δ_unmixed ≥ Δ_mixed`. Because D_ft has **zero covariance along the specialized
  directions**, fine-tuning gradients have **zero projection** there, so that capability
  survives. The inconsistent features overlap D_ft and get overwritten.

So the paper's own explanation is a claim about **where in representation space the capability
lives, and whether the fine-tuning gradient can reach it.** That is a directly measurable
geometric statement, and nobody has measured it in a real transformer — the theory is proved
in a two-layer linear toy.

**Our question:** *does the J-space of a real 135M transformer reorganize with λ in the way
Theorems 5.1–5.3 predict?*

### 1.3 The J-lens / J-space instrument

From Anthropic's "workspace" work (transformer-circuits.pub/2026/workspace). The **J-lens** is
the averaged first-order causal map from a layer to the output:

```
J_ℓ = E_{t, t' ≥ t, prompt} [ ∂h_final,t' / ∂h_ℓ,t ]        (d_model × d_model per layer)
lens(h_ℓ) = softmax( W_U · norm( J_ℓ h_ℓ ) )
```

The **J-lens dictionary** at layer ℓ is `D_ℓ = W_U J_ℓ ∈ R^{n_vocab × d_model}` — one vector per
vocabulary item, the direction at layer ℓ that most promotes that token at the output. It is
overcomplete (49152 ≫ 576). The **J-space** is the set of points expressible as a *sparse
nonnegative* combination of ≤ k dictionary vectors; decomposition is by gradient pursuit.

**Honesty note.** The original work studies a much larger model and reports workspace
phenomenology (k ≈ 25 concurrent concepts, onset near layer 38, ~10% of activation variance)
that we should **not** expect to transfer literally to a 30-layer, d=576 model. We therefore use
the J-lens as a **measurement operator** — a principled, causally-grounded readout basis — and
define the J-space empirically for this model. We are not claiming to find a global workspace
in a 135M model. Any layer indices, k, and variance fractions are re-derived here, not assumed.

### 1.4 Planned measurements and their predictions

Each maps to a theorem. All run on already-saved permanent checkpoints — inference plus one
batched backward, **no retraining**.

| # | Measurement | Tests | Prediction if the theory holds |
|---|---|---|---|
| M1 | **Domain subspace overlap.** Principal angles between the J-space spanned by MusicPile-active components and by C4/ChemPile-active components. | 5.1 | Overlap **decreases** with λ — MusicPile gets its own directions. |
| M2 | **New vs. reused components.** Do λ>0 stage-2 checkpoints activate J-lens components that λ=0 never activates, or reweight shared ones? | 5.2 | λ>0 recruits **new** components; λ=0 **reweights shared** ones. |
| M3 | **Gradient projection.** Norm of the ChemPile stage-3 gradient projected onto the MusicPile-specialized J-directions, at θ_post. | 5.3 (most direct) | Projection **→ 0** as λ increases. |
| M4 | **Survival through stage 3.** Subspace alignment of the MusicPile J-space between θ_post and θ_ft. | 5.3 | Survival **increases** with λ; should correlate with measured `L_ret`. |
| M5 | **Capacity / sparsity.** Active-k and variance explained by the MusicPile J-space. | descriptive | Establishes the model's own workspace scale. |

M3 is the crux: it is a *direct* measurement of the theory's load-bearing assumption
("D_ft has no covariance along the specialized features") in a real transformer, and it is
computable at θ_post **before** stage 3 runs — i.e. it is a genuine **prediction**, not a
post-hoc fit. If M3 lands, we can predict retention from geometry alone.

**Compute note.** One backward pass seeded with basis vector `e_j` at *all* output positions
yields, by causality, `Σ_{t'≥t} ∂h_final,t'[j]/∂h_ℓ,t` for **every** source position t and
**every** layer ℓ at once. So the full J_ℓ for all 30 layers costs d_model = 576 backward
passes per prompt batch, not 576 × layers × positions. Tractable.

**Prompt corpora.** The existing `probes/` sets are exactly right and require no new data:
256 fixed windows per corpus (c4 / musicpile / chempile), taken as a deterministic prefix of
each frozen val split, with pinned dataset revisions and recorded shard sha256. The *same*
probe tokens are used for every checkpoint of every run, so cross-condition J-lens
comparisons are apples-to-apples by construction.

### 1.5 Implementation status

`src/j_pretrain/analysis/jlens.py` — built and tested (commit `398f04e`):
`jlens_batch` / `compute_jlens` (J_ℓ), `jlens_dictionary` (W_U J_ℓ), `gradient_pursuit`
(sparse **nonnegative** decomposition), `principal_angles` / `subspace_overlap` (M1 primitive).

Both compute shortcuts above are easy to get subtly wrong, so `jlens_batch` is verified
against `jlens_batch_reference`, a naive one-backward-per-output-dim implementation, in
`tests/test_jlens.py` (11 tests, all passing). Also asserted: chunk size doesn't change the
result, all-layers-in-one-pass equals per-layer calls, and the lens never pollutes `.grad`.

---

## 2. Where things stand

### 2.1 Results so far

**λ = 0 complete and audited** (on the RTX 4090):

| quantity | value |
|---|---|
| `L_im` | 2.2311 |
| `L_ret` | 2.3948 |
| `L_ft` | 2.1574 |
| `L_pre` | 3.7238 |
| forgetting (`L_ret − L_im`) | 0.1637 |

λ = 0.25 stage-1 reached ~5.8% before being shelved to free the GPU. λ = 0.5 / 0.75 / 1.0 not
started. **One point on a five-point curve — no trend yet.**

### 2.2 Measured cost (not projected)

From λ=0's actual wall-clock on the 4090, end-to-end including eval and checkpointing:

| stage | tokens | wall-clock | effective tok/s |
|---|---|---|---|
| stage 1 | 8.70B | 46.79 h | 51,650 |
| stage 2 | 1.60B (early-stopped) | 10.37 h | 42,900 |
| stage 3 | 0.20B | 2.27 h | 24,500 |
| **per condition** | | **≈ 59.5 h** | |

Remaining work ≈ 42B tokens ≈ **~10 days serial on the 4090**, which is also busy with another
project. Hence the move to PSC.

Note the orchestrator runs **eager** (`torch_compile=False`, `run.py:164`) while the repo's own
benchmark measured 94k tok/s compiled vs 52k eager. We are leaving ~1.8× on the table; see
decision D-4.

---

## 3. Infrastructure — the PSC port

**Design constraint:** the orchestrator is heavily tested (101 tests) and enforces mission
invariants (one GPU job at a time; byte-identical shared init; no cross-run checkpoint reuse).
Refactoring it for multi-GPU was the wrong risk. Instead:

> **One private working copy per λ, all sharing one artifact root.**

`OrchestratorConfig` already parameterizes `state_dir` and `inventory_dir`, and checkpoints /
run_metrics are already namespaced by `run_id`. So five independent copies, each with its own
`state/`, `gpu.lock`, DAG and inventory, run the *unmodified* orchestrator concurrently on five
GPUs. **Zero orchestrator code changes.** `PYTHONPATH=<copy>/src` makes `REPO_ROOT`
(`Path(__file__).parents[3]`) resolve per-copy.

Each copy's inventory is seeded with **only** the `shared-init` records, so
`ensure_init_checkpoint` finds the transferred init and every stage-1 starts from
byte-identical weights.

Layout:

```
/ocean/projects/cis220039p/hshi6/
  j-pretrain-psc/{miniforge,conda-env,repo,runs/<run_id>/,slurm,logs,state}
  j-pretrain-artifacts/{datasets,checkpoints,run_metrics}      # SHARED
```

Slurm: `ROBO` partition (8×H100 nodes), 1 GPU + 12 cores + 100G per job.
`--dependency=singleton` on a per-condition job name guarantees only one job per condition is
ever alive, which is what makes clearing a stale `gpu.lock` at job start safe, and lets us
submit to two partitions as a hedge without any double-run risk. Each job queues its own
successor up front so a hard walltime kill is still picked up; successors exit immediately on
seeing the `DONE` marker.

---

## 4. Decisions

| id | decision | reasoning |
|---|---|---|
| D-1 | **ROBO, not GPU-shared.** | ROBO QOS allows **7-day** walltime (vs 48h) so one job runs a whole condition with no chaining, and ROBO is not SU-gated for this account — it preserves all 20,059 remaining GPU SU. |
| D-2 | **Restart λ=0.25 from scratch** rather than resuming the 4090's 5.8% checkpoint. | Resuming would make that one condition a hardware chimera (half 4090, half H100). Costs ~3 h of redone work; buys a clean per-condition hardware provenance. |
| D-3 | **Add a λ=0 replication on PSC** (`music-300m_lambda-0.0-ctl`). | Doubles as (a) a hardware control for the cross-machine confound and (b) an end-to-end validation of the port: if it reproduces the 4090's `L_im`/`L_ret`, the port is faithful. Explicitly **not** a Fig-3a grid point. Submitted last so real conditions get GPUs first. |
| D-4 | **Keep `torch_compile=False`.** | H100 already gives ~2×; changing hardware *and* execution mode at once compounds numerical confounds against the finished λ=0. Not worth 1.8× here. |
| D-5 | **Disable wandb** (`--no-wandb`). | PSC compute nodes have no outbound internet. The `run_metrics/*.jsonl` stream is the source of truth anyway. |
| D-6 | **Keep the `env` dict (and so `environment_hash`) unchanged.** | It records *package* versions, which are pinned identically on PSC. Accurate as-is. The hardware change belongs in this notebook and `docs/ENVIRONMENT.md`, not in a package-version hash. |

---

## 5. Issues encountered, and how they were fixed

### I-1 — Orchestrator DAG orphaned a shelved node *(fixed, committed 2c0e1f6)*
A node left in a non-terminal status (`running` / `failed_retryable`) was never rescheduled,
because `dag.is_ready` only ever schedules `planned`. So the manually shelved λ=0.25 stage-1 was
orphaned and a relaunch **silently skipped ahead to λ=0.5** (killed after ~3 min, wrote no
artifacts). Fix: `reclaim_stale_nodes()` runs at startup while the GPU lock is held — holding
the lock proves no live trainer owns any node, so any non-terminal node is stale by definition —
and returns them to `planned`. Retry counts persist in `experiment_state.json` (limit 3, then
`failed_blocked`). Two regression tests; full suite 101 pass.

### I-2 — PSC rejects the 4090's SSH key even though it is in `authorized_keys` *(worked around)*
Fingerprints confirmed identical (`SHA256:yjmT01Zh…` present in `~/.ssh/authorized_keys` on
PSC), yet the 4090 gets `Permission denied (publickey,…)` on both the login node and the data
mover, while the Mac authenticates fine with *its* key. The 4090 additionally reports
`No Kerberos credentials available`.

Most likely PSC honors keys registered through their portal rather than a hand-appended
`authorized_keys` line (the file has a `.bak` from a recent rewrite, consistent with
center-managed regeneration).

**Impact:** no direct 4090→PSC path, even though the 4090 uplinks at **38 MB/s** while this
Mac uplinks at **~3 MB/s**. Workaround: two resumable rsync hops staging through the Mac
(4090 → Mac at 18 MB/s, Mac → PSC at 3 MB/s).
**Worth fixing properly:** registering the 4090's key via the PSC portal would make future
transfers ~10× faster. Needs the account owner.

### I-3 — rsync broken on both ends *(fixed)*
PSC login nodes: `/opt/packages/rsync/3.2.3/bin/rsync` fails with
`libxxhash.so.0: cannot open shared object file`, and there is no rsync on `PATH`.
Fixed by installing rsync 3.4.4 from conda-forge into the project env and pointing
`--rsync-path` at it.

macOS ships **openrsync** ("protocol version 29, rsync 2.6.9 compatible"), which rejects
`--info=progress2` and `--inplace`. Reduced to `-a --partial`, which both ends understand.

### I-4 — Naive tar-over-double-ssh ran at 0.7 MB/s *(fixed)*
First transfer attempt piped `ssh 4090 tar | ssh psc tar`. Two nested SSH encryption layers on
an M1 Mac gave 0.7 MB/s — 17 GB of C4 would have taken ~7 h. Plain `dd | ssh` measured 3 MB/s
on the same link, so the pipe itself, not the network, was the bottleneck. Replaced with two
separate rsync hops staging on local disk: each leg runs at its natural speed and is resumable.

### I-5 — ROBO `--mem` cap is 120000M per GPU, and the error is misleading *(fixed)*
`--mem=200G` failed with two lines: `Allocation requested mem higher than maximum of
120000M/gpu` **and** `allocation failure: Access/permission denied`. The second line looks like
an entitlement problem — given the account's Robo GPU balance sits at −222,377 SU, it is easy to
misread as "you are out of SUs and blocked". It is not: it is a cascade of the first error.
`--mem=100G` submits fine. **ROBO is usable for this account.**

### I-10 — Login-node daemons silently die; PSC round-robins login nodes *(fixed)*
The validation gate was first written as a `setsid nohup` daemon on a login node. It got
through the env-build wait, started the test suite, reached 65 of 101 tests, and **stopped** —
no error, no log entry, process gone.

Two compounding causes:
1. PSC **reaps long-running / CPU-heavy processes on login nodes**. A full test suite (which
   runs real training steps) is exactly what that policy targets. Running it there was my
   mistake, not a quirk.
2. `ssh bridges2` is **load-balanced across login nodes** (br012 / br013 / br014 observed).
   So a follow-up `ps` can land on a different node and show nothing, making a live process
   look dead and a dead one impossible to confirm. The `.ib` hostnames are internal, so you
   cannot check a specific login node from outside either.

Fix: the gate is now a **Slurm batch job** (`slurm/gate.sbatch`, RM-shared, 4 cores) — it is
scheduled, not reaped, its stdout is a real job log, and its state is visible via `squeue`.
Cost is ~1 SU against the 5,816 remaining RM balance.

**General rule for this project: nothing long-running goes on a PSC login node.** Batch job or
it does not exist. The bulk transfers are the exception only because they are driven from the
Mac and the 4090, with PSC merely receiving.

### I-7 — Queued a 184.9 GB transfer where 0.98 GB was needed *(caught and fixed)*
Staging the finished λ=0 condition for J-space analysis, the whole
`checkpoints/music-300m_lambda-0.0` tree was queued without looking at its composition:

| class | size | purpose |
|---|---:|---|
| **resumable** | **142.6 GB** | optimizer + RNG state, only for restarting training |
| analysis | 42.3 GB | weights only (~127 snapshots @ 0.33 GB) |

λ=0 is **complete** — nothing will ever resume it — and the J-lens reads weights only. Worse,
of the analysis snapshots the measurements need exactly three: θ_pre (stage1 `final`),
θ_post (stage2 `restored_best`), θ_ft (stage3 `final`). **0.98 GB, not 184.9 GB** — a ~190×
overshoot, ~7 hours of transfer avoided.

General lesson for this project: checkpoint *class* matters as much as run/stage when moving
artifacts. The permanent-retention policy means resumables dominate on-disk size (142.6 of
184.9 GB here) and are almost never what analysis wants. It also reframes co-location: each
condition's analysis needs ~1 GB, so all five conditions move ~5 GB total and the J-space
comparison can run on either machine. There was never a reason to move 700 GB.

### I-8 — Bulk rsync starves interactive SSH on the same ControlMaster *(worked around)*
Once the Mac started pushing shards, every `ssh bridges2` status check began timing out at
120 s. Cause: connection sharing multiplexes all channels over **one** TCP connection, so
interactive commands queue behind rsync's bulk data. Fix: pass `-o ControlPath=none` for
status checks (the Mac's key authenticates fresh connections fine). Only the *4090* is
dependent on its master socket.

### I-9 — PSC ingest is single-stream capped; parallel senders fix it *(fixed)*
Direct 4090→PSC rsync sustained **2.8 MB/s** — no better than the Mac relay — despite the
4090 uplinking at 38 MB/s to a public speed test. So the limit is per-TCP-connection ingest
into PSC (81 ms RTT, single stream), not the source uplink.

Parallel streams are the standard fix but are blocked here: only the ControlMaster socket
authenticates, and its channels share one TCP connection. Workaround: use **two different
senders** — the Mac (which still had 71 of 85 C4 shards staged from the earlier relay) and
the 4090 — splitting the shard range. They land on different PSC login nodes (br012/br013)
and get independent caps: **2.8 → 6.6 MB/s**. The Mac works top-down from shard 70 while the
4090 works bottom-up, so they never collide; rsync skips already-matching files at the seam.

### I-6 — Slurm `--test-only` start times are alarming but not predictive *(understood)*
`--test-only` reported starts 8–10 days out for 48h GPU jobs. That is the *guaranteed* start
assuming no backfill. Empirically, over 7 days of accounting (33k GPU jobs), H100 jobs
requesting ≥24h waited a **median of 0.74 h** (p75 4.3 h). Use the empirical distribution for
planning, not `--test-only`.

---

## 6. Environment / access facts worth not rediscovering

- PSC login nodes have outbound HTTPS (GitHub, PyPI, HF all reachable). Compute nodes do not —
  hence `--no-wandb` and pre-staged datasets.
- No conda on PSC login nodes; the project uses a self-contained env at
  `j-pretrain-psc/conda-env`, prepended to `PATH` (no `conda activate`).
- `/jet/home` is 25 GB — never put checkpoints or caches there. Everything goes to `/ocean`.
- ROBO nodes are x86 + H100 despite the `robo-gh*` hostnames. Not Grace-Hopper.
- **V100 and P100 are disqualified for this project**: sm_70 / sm_60 have no bf16, and the
  trainer is bf16-autocast with no GradScaler and bf16 safetensors payloads. That rules out 176
  of Bridges-2's 192 non-H100 GPUs, and every AirLab `gpu.p100.*` flavor.
- Storage: one *completed* condition is **173 GB** of permanent checkpoints. Five conditions
  ≈ 700 GB (fits `/ocean`; would have been tight on the 4090's remaining 770 GB). The old
  `reports/FEASIBILITY.md` estimate of 268 GB total was a ~2.6× underestimate.

---

## 7. Launch state — 2026-08-11 ~20:00 UTC

Five conditions submitted, ten Slurm jobs:

| condition | λ | job name | GPU-shared (heads chain) | ROBO (overflow) |
|---|---|---|---|---|
| music-300m_lambda-0.25 | 0.25 | `jp-l025` | 43366617 | 43366618 |
| music-300m_lambda-0.5 | 0.5 | `jp-l05` | 43366619 | 43366620 |
| music-300m_lambda-0.75 | 0.75 | `jp-l075` | 43366621 | 43366622 |
| music-300m_lambda-1.0 | 1.0 | `jp-l10` | 43366623 | 43366624 |
| music-300m_lambda-0.0-ctl | 0 (control) | `jp-l00c` | 43366625 | 43366626 |

**Why GPU-shared heads each chain (revision of D-1).** The original plan was ROBO-only (free,
7-day walltime). Two facts changed it: (a) the account's Robo balance is −222,377 SU, which
tanks fairshare — a 10-minute ROBO probe sat `PENDING (Priority)` for 25+ min with 8 GPUs
visibly free; (b) `--dependency=singleton` waits for the prior same-named job to *terminate*,
not to *start*, so submitting to both partitions serializes rather than races, and a true
multi-partition job is impossible here (ROBO nodes expose no features, and ROBO/GPU-shared have
incompatible partition QOS). So GPU-shared heads each chain for a predictable start (~1,800 SU,
9% of the 20,059 balance, expiring 2026-09-30 anyway), with the free ROBO job queued behind it
as genuine overflow if GPU-shared hits its 48 h walltime first.

**Nothing trains until the gate opens.** `gate.sh` on PSC releases
`$ART/datasets/.C4_READY` only after: env build done → repo test suite passes on PSC →
all six packed datasets open with `len()` matching the manifest **and** every shard sha256
verified → shared-init checkpoint loads. Jobs poll for that marker and exit cleanly (successor
retries) rather than training on a truncated corpus. Stage 1 consumes 8,496,093 of C4's
8,496,094 windows, so a partial transfer would otherwise fail deep into the run.

## 8. Open items

- [x] PSC env built — torch 2.5.1+cu121, transformers 4.46.3, datasets 3.1.0,
      tokenizers 0.20.3, safetensors 0.4.5, numpy 2.1.3 (exactly the pinned set)
- [x] Probe sets (256 windows × 3 corpora) staged to PSC — the J-space prompt corpora
- [x] J-lens instrument implemented and verified (`398f04e`)
- [ ] C4 transfer to PSC (in flight, two senders)
- [ ] Confirm gate released and first stage-1 steps logging
- [ ] **Measure real H100 tok/s and replace the estimated 1.8–2.2× speedup with a number**
- [ ] Build the M1–M5 measurement driver on top of `jlens.py`; validate against the λ=0
      θ_pre/θ_post/θ_ft snapshots (0.98 GB, queued behind C4)
- [ ] Register the 4090's key with the PSC portal (owner action, see I-2) — would make future
      transfers ~10× faster
- [ ] AirLab is quota-blocked by an ERROR'd instance from another project — see §9

---

## 9. AirLab Cloud — checked 2026-08-11, not usable for this project

Live query via `openstack` (app credential works from the Mac without VPN; only the floating
IPs need CMU network).

Quota headroom is **40 vCPU / 102,400 MB RAM** — 160/200 vCPU and 614,400/716,800 MB are
consumed, 128 vCPU / 491,520 MB of that by `sdft-8b-a100x4`, which is in **ERROR** state and
still holding both its quota and the `sdft-8b-boot` volume (`reserved`).

Against that headroom, exactly one GPU flavor can launch:

| flavor | vCPU | RAM (MB) | fits 40 / 102,400? | usable here? |
|---|---:|---:|---|---|
| `gpu.a100.1` | 32 | 122,880 | ✗ RAM over by 20,480 | — |
| `gpu.rtx6000ada.1` | 42 | 163,840 | ✗ both | — |
| `gpu.rtxpro5000.1` | 38 | 56,320 | ✓ | **yes** (1 GPU) |
| `gpu.p100.1` | 20 | 56,320 | ✓ | **no** — P100 is sm_60, no bf16 |

So AirLab currently offers **one** usable GPU, which buys no parallelism over the 4090 and adds
a VPN dependency. Not worth porting to.

Freeing it would mean deleting another project's ERROR'd instance (`sdft-8b-a100x4`), which
would release 128 vCPU + 480 GB and make `gpu.a100.4` launchable. **Not done** — that is not
this project's instance to delete. Flagging for the owner.
