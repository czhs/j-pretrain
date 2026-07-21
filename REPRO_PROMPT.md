# Mission

Reproduce the pretraining-mixing-lambda result in Figure 3a of the attached paper,
"Early Data Exposure Improves Robustness to Subsequent Fine-Tuning."

You are running autonomously inside a Ralph loop on this machine:

- One NVIDIA RTX 4090
- 24 GB GPU memory
- 64 GB system RAM
- Linux
- The paper is available as `paper.pdf` in the project root

Continue until the experiment is genuinely complete. Do not declare success merely
because the code exists, a smoke test passes, or the expected trend appears in a partial
run. The completion condition refers to completing the experiment, whether or not the
paper's result replicates.

Only output:

<promise>EXPERIMENTS_COMPLETE</promise>

after every mandatory completion criterion near the end of this prompt has been
verified programmatically.

Do not optimize, discard, rerun, or selectively report conditions based on whether
they support the paper's conclusion.

# Scientific question

Test whether exposing a language model to part of its eventual post-training corpus
during Stage 1 pretraining changes how well the post-trained capability survives a
subsequent Stage 3 fine-tuning update.

The main paper result to reproduce is:

1. Stage 1: Pretrain a SmolLM2-style approximately 135M-parameter causal LM on C4.
   During pretraining, expose it to a lambda fraction of a fixed MusicPile subset.
2. Stage 2: Post-train the resulting checkpoint on that same MusicPile subset until
   validation loss converges.
3. Stage 3: Fine-tune the post-trained checkpoint on ChemPile.
4. Measure:
   - Immediate MusicPile validation loss after Stage 2, L_im.
   - Retained MusicPile validation loss after Stage 3, L_ret.
   - ChemPile validation loss after Stage 3, L_ft.
   - C4 validation loss after Stage 3, L_pre.
5. Vary lambda over:
   {0.0, 0.25, 0.5, 0.75, 1.0}.
6. Use MusicPile subset sizes:
   {30M, 150M, 300M tokens}.
7. The paper reports that L_im is nearly flat across lambda while L_ret generally
   decreases as lambda increases.

This is Figure 3a, not the compute-matched allocation experiment in Figure 3b.
Do not accidentally reserve the remaining `(1-lambda)` fraction for Stage 2. In the
Figure 3a experiment, Stage 2 trains on the complete chosen MusicPile subset for every
lambda and is allowed to train to convergence. Early exposure therefore adds prior
exposure rather than reallocating a fixed one-pass budget.

# First actions: inspect, research, and freeze the specification

Before implementing training:

1. Read `paper.pdf` carefully, especially:
   - Sections 3.3, 4.1, and Appendix A.
   - Architecture, optimizer, precision, token budgets, tokenizer, dataset construction,
     stopping criteria, evaluation intervals, and seeds.
2. Extract all relevant details into `docs/PAPER_SPEC.md`.
3. Clearly distinguish:
   - Explicitly stated facts.
   - Facts recovered from official code or project materials.
   - Necessary implementation choices not specified by the authors.
4. Use available browser, GitHub, web-search, MCP, and command-line tools to look for:
   - The paper's official project page.
   - An official code repository.
   - Public configurations or checkpoints.
   - Exact MusicPile, ChemPile, C4, tokenizer, and architecture identifiers.
5. Prefer official author code and configurations when available. Record URLs, commit
   hashes, dataset revisions, and license information in `docs/REFERENCES.md`.
6. Never treat a third-party reimplementation as authoritative without documenting it.
7. Have a dedicated research subagent independently inspect the paper and available
   official materials. Have a second subagent audit the extracted specification.
8. Resolve discrepancies using this priority:
   official experiment code > paper appendix tables > main paper prose >
   closest documented SmolLM2 configuration > explicitly documented local choice.

Do not begin expensive training until the frozen specification has been independently
reviewed and committed.

# Use Claude Code's relevant capabilities

At the start, inspect the tools, plugins, MCP servers, skills, agents, hooks, and
background-task facilities available in this Claude Code session.

Use relevant capabilities rather than attempting to use every installed plugin
gratuitously:

- Ralph loop: remain active until all mandatory runs and analyses are complete.
- Subagents:
  - Paper/configuration extraction.
  - Data-pipeline review.
  - Training-code review.
  - Final independent reproducibility audit.
- Background tasks or Monitor:
  - Run long training jobs without filling the main context with logs.
  - React to failures and milestone messages.
- Hooks:
  - A lightweight post-edit hook for formatting and fast tests.
  - A PreCompact hook that verifies durable state files are current.
  - Do not interfere with the Ralph Stop hook.
- Skills:
  - Create a small project-local `experiment-status` skill if useful.
  - Keep large procedural instructions out of `CLAUDE.md`.
- Git and Claude checkpoints:
  - Commit each stable milestone.
  - Use Git as the durable history of code and configuration.
- Browser/GitHub MCP:
  - Use it for official repositories or documentation when configured.
- Auto memory:
  - Store only durable engineering lessons.
  - Never use auto memory as the canonical experiment-status database.

Do not use `--dangerously-skip-permissions`. Ask only for narrowly necessary,
repeatable permissions. Do not publish the repository or upload datasets/checkpoints
without explicit authorization.

# Context and memory discipline

The experiment may run for days or weeks. The conversation is not reliable long-term
memory. Establish durable state before doing substantial work.

Create:

- `CLAUDE.md`
  - Fewer than 150 lines.
  - Invariant project rules, key commands, directory layout, and completion rule.
  - It must tell future compacted contexts to read the state files below first.
- `state/experiment_state.json`
  - Canonical machine-readable state.
  - Current phase, run IDs, statuses, retries, checkpoint paths, metrics, and next action.
- `state/NEXT_ACTION.md`
  - At most 50 lines.
  - Exact current status and next concrete action.
- `state/DECISIONS.md`
  - Concise scientific and engineering decisions with rationale.
- `state/SCOPE_LOCK.json`
  - Immutable experiment scope after the benchmark phase.
- `runs/manifest.jsonl`
  - One immutable record per planned run and state transition.
- `results/results.csv`
  - One row per completed experimental condition.
- `logs/`
  - Raw logs. Never load an entire long log into context.
- `docs/FAILURES.md`
  - Repeated failures, diagnosis, and remedies.
- `docs/HANDOFF.md`
  - Concise reconstruction instructions for a fresh Claude session.

At the beginning of every Ralph iteration:

1. Read `CLAUDE.md`.
2. Read `state/experiment_state.json`.
3. Read `state/NEXT_ACTION.md`.
4. Inspect only the relevant tail or filtered lines of logs.
5. Perform the next useful action.
6. Atomically update the state files before attempting to stop.

Keep raw command output out of context:

- Redirect full training output to timestamped files.
- Emit concise structured heartbeat lines.
- Inspect at most the last 100–200 relevant log lines unless deeper diagnosis is needed.
- Summarize old decisions periodically instead of allowing status files to grow forever.
- Before compaction, update `NEXT_ACTION.md`, `HANDOFF.md`, and the JSON state.
- Never rely on an earlier conversation statement that is absent from durable files.

# Ralph-loop behavior

You are already running inside the Ralph loop. Do not invoke another nested Ralph loop.

Do not busy-poll training:

- Prefer Claude Code's Monitor/background-task functionality.
- Otherwise use a watcher that blocks and emits only on:
  - Evaluation completion.
  - Checkpoint creation.
  - Process failure.
  - Run completion.
  - A low-frequency heartbeat, no more than once every 10 minutes.
- Do not repeatedly call `nvidia-smi`, `tail`, or status commands every few seconds.
- A long training process is not evidence that the task is stuck.
- Keep the training orchestrator independent of the current Claude context so an
  automatic compaction does not terminate it.

Use `tmux`, a process supervisor, or a robust detached subprocess for the experiment
runner. Record PIDs, tmux session names, commands, and log paths in state.

# Repository and implementation requirements

Build a clean Python project with:

- A pinned environment and reproducible install command.
- PyTorch and a mature causal-LM stack.
- `pyproject.toml` or equivalent dependency lock.
- Typed configuration files.
- Unit tests for:
  - Lambda data allocation.
  - Token counting.
  - Data leakage prevention.
  - Checkpoint/resume.
  - Evaluation-loss calculation.
  - Pareto/statistical utilities where applicable.
- Integration smoke tests.
- Deterministic run identifiers.
- Resumable stages.
- Atomic checkpoint writes.
- Machine-readable metrics.
- A single experiment orchestrator that executes the run DAG.

Prefer stable optimized primitives:

- BF16 training.
- PyTorch SDPA or FlashAttention when installation and numerical tests are stable.
- Fused AdamW where supported.
- Gradient accumulation.
- Gradient checkpointing only when needed.
- `torch.compile` only if a benchmark demonstrates stable improvement.
- Memory-mapped or streaming tokenized data.
- Persistent workers and pinned-memory data loading when beneficial.

Do not spend hours compiling fragile optional dependencies for a marginal speedup.
Always retain a stable fallback.

# Hardware benchmark and feasibility calibration

Before the real runs:

1. Record:
   - GPU model and memory.
   - Driver and CUDA versions.
   - PyTorch version.
   - CPU model.
   - System RAM.
   - Free disk space.
2. Run a representative training benchmark at the exact model size and sequence length.
3. Autotune only hardware-execution parameters:
   - Per-device microbatch.
   - Gradient accumulation needed to preserve the frozen global batch.
   - Number of data-loader workers.
   - SDPA/FlashAttention choice.
4. Do not tune scientific hyperparameters separately by lambda.
5. Measure:
   - Tokens/second after warmup.
   - Peak allocated and reserved VRAM.
   - Data-loader wait.
   - Checkpoint time.
   - Estimated duration per run and for the complete DAG.
6. Leave at least approximately 1.5 GB of VRAM headroom during the benchmark.
7. If an OOM occurs:
   - Reduce microbatch.
   - Increase gradient accumulation to preserve global batch.
   - Then enable activation checkpointing if needed.
   - Do not change the effective batch or scientific condition silently.
8. Save the benchmark and estimated schedule in `docs/COMPUTE_PLAN.md`.

After this benchmark, write `state/SCOPE_LOCK.json`. The mandatory scope is the full
Figure 3a lambda grid at all three MusicPile sizes unless an objective blocker exists,
such as unavailable data, inadequate disk capacity even after safe pruning, or an
estimated duration exceeding 45 uninterrupted days.

If full Figure 3a is objectively infeasible, lock the largest scientifically meaningful
scope in this order:

1. Full lambda grid for the 30M-token MusicPile condition.
2. Lambda endpoints and midpoint `{0, 0.5, 1.0}` for 150M and 300M.
3. At least one additional confirmation seed for `{lambda=0, lambda=1}` at 30M.

Document the reason and quantitative estimates. Do not silently downscale. Once the
scope is locked, do not reduce it merely because the result is inconvenient or training
is slow.

# Data construction

Use the exact datasets and revisions from the paper or official code whenever possible:

- D_pre: C4.
- D_post: MusicPile.
- D_ft: ChemPile.

Requirements:

1. Use one tokenizer for all three stages, matching the paper.
2. Create deterministic held-out validation splits that are never trained on.
3. Prevent document or token-window overlap between train and validation.
4. Tokenize once and reuse memory-mapped shards across lambda runs.
5. Avoid loading the full corpora into the 64 GB RAM.
6. Record SHA-256 hashes or equivalent content manifests.
7. Construct fixed, nested MusicPile training subsets:
   - First deterministic 30M selected training tokens.
   - A 150M-token superset.
   - A 300M-token superset.
   Use author methodology if available.
8. Use the same MusicPile subset in:
   - Lambda exposure during Stage 1.
   - Stage 2 post-training.
9. Lambda means the fraction of the chosen MusicPile subset seen during Stage 1:
   - 30M, lambda 0.25 => 7.5M MusicPile tokens during Stage 1.
   - 150M, lambda 0.25 => 37.5M.
   - 300M, lambda 0.25 => 75M.
10. Expose each selected MusicPile token at most once during Stage 1.
11. Distribute MusicPile exposure through pretraining rather than placing it all at the
    end, unless official code specifies another schedule.
12. Keep the C4 sequence, random initialization, batching logic, and all other conditions
    identical across lambda wherever possible.
13. Determine from the paper/code whether MusicPile tokens are added to or replace C4
    tokens in the fixed Stage 1 budget. Follow official behavior. If unspecified, keep
    the C4 exposure fixed and add the lambda-selected MusicPile tokens, then document
    this choice prominently.
14. Do not accidentally include ChemPile during Stages 1 or 2.
15. Write tests that inspect realized token counts and source proportions.

Create a data audit report before training any real model.

# Model

Use the paper's approximately 135M-parameter SmolLM2-style decoder-only architecture.

Recover the exact 135M configuration from official sources if available. Otherwise use
the closest official SmolLM2-135M configuration and record every field:

- Parameter count.
- Hidden size.
- Number of layers.
- Attention heads and query groups.
- MLP width.
- Vocabulary size.
- Context/training sequence length.
- RMSNorm details.
- RoPE base.
- Weight tying.
- Initialization.
- Dropout.

Initialize every Stage 1 condition from the same initial weights and random seed.
Do not initialize from a publicly pretrained 135M checkpoint, because the experiment
tests when the target domain is first encountered during pretraining.

# Stage 1: pretraining

Use the paper's Stage 1 budget and optimizer configuration as exactly as recoverable.

The paper describes approximately 10B pretraining tokens and lists 8.7B C4 training
tokens for its 135M setup. Resolve the intended Figure 3a budget from official code or
the most specific paper table and document the resolution.

For every MusicPile size and nonzero lambda, pretrain a corresponding Stage 1 model.
The lambda=0 Stage 1 checkpoint is independent of MusicPile subset size and should be
reused across the 30M, 150M, and 300M Stage 2 conditions.

Reuse work only when it is mathematically identical. Do not use interpolation,
checkpoint surgery, or sequential lambda training.

All Stage 1 runs must:

- Begin from identical initialization.
- Use identical C4 token order.
- Use the same frozen optimizer schedule.
- Differ only in intended MusicPile exposure.
- Save resumable checkpoints periodically.
- Save the final Stage 1 checkpoint.
- Record exact token-source counts.

Prune old optimizer checkpoints safely:

- Keep the latest resumable checkpoint.
- Keep the final checkpoint.
- Keep any checkpoint required to verify recovery.
- Never delete a checkpoint while its replacement is still being written.
- Do not retain dozens of obsolete multi-GB checkpoints.

# Stage 2: MusicPile post-training

Starting from each corresponding Stage 1 checkpoint:

1. Post-train on the complete selected MusicPile subset.
2. Use full-parameter fine-tuning, not LoRA.
3. Use AdamW with the paper's shared settings, including:
   - beta1 = 0.9.
   - beta2 = 0.95.
   - gradient clipping at max norm 1.0.
   - BF16 mixed precision.
   - weight decay 0.1 where specified.
   - linear warmup and cosine decay.
4. Recover the fixed Figure 3a learning rate, global batch, evaluation interval,
   patience, and stopping rule from official code if possible.
5. If the exact fixed configuration is unavailable:
   - Choose it once using a small, preregistered pilot on lambda=0.
   - Use a tuning split distinct from final validation.
   - Select only from the paper's reported 135M search space.
   - Freeze the selected configuration before evaluating the lambda grid.
   - Never tune separately for different lambdas.
6. Continue until MusicPile validation loss stops improving, up to the paper's 2B-token
   maximum.
7. Define the stopping rule precisely in advance, including minimum improvement,
   patience, and evaluation frequency.
8. Restore the best validation checkpoint, not simply the final step.
9. Evaluate token-weighted MusicPile validation loss and store it as L_im.
10. Also evaluate C4 and ChemPile validation loss at the Stage 2 checkpoint for
    diagnostics, but do not use them for model selection.

# Stage 3: ChemPile fine-tuning

Starting from each best Stage 2 checkpoint:

1. Fine-tune on ChemPile for the paper's fixed 200M-token budget.
2. Use the paper's fixed Stage 3 learning rate of 5e-5 for the Figure 3a comparison.
3. Recover the remaining Stage 3 optimizer and schedule details from official sources.
4. Use the same Stage 3 data order and configuration for every lambda.
5. Do not early-stop Stage 3 based on MusicPile retention.
6. At the final Stage 3 checkpoint evaluate:
   - L_ret: MusicPile validation loss.
   - L_ft: ChemPile validation loss.
   - L_pre: C4 validation loss.
7. Evaluation must be deterministic, token-weighted, and performed on identical
   validation examples for every condition.

# Run scheduling

Represent the experiment as a dependency DAG.

Potential reusable nodes include:

- Tokenized C4, MusicPile, and ChemPile shards.
- Shared random initialization.
- Shared lambda=0 Stage 1 checkpoint.
- Frozen data splits.
- Frozen Stage 2 configuration.
- Frozen Stage 3 configuration.

Run one GPU-training process at a time. Do not start competing CUDA jobs.

The orchestrator must:

- Detect completed valid artifacts and skip them.
- Detect partial runs and resume them.
- Validate checkpoint metadata before resuming.
- Mark runs failed only after collecting an error report.
- Retry transient failures at most three times.
- Never restart a 10B-token run from scratch if a valid checkpoint exists.
- Verify that no other process occupies substantial VRAM before launching.
- Record every launch command and environment.
- Shut down data-loader workers cleanly.
- Continue to the next DAG node automatically.

For repeated CUDA OOM:

1. Confirm no unrelated GPU process exists.
2. Lower microbatch and preserve global batch with accumulation.
3. Resume from the latest valid checkpoint.
4. Record the execution-only change.

For NaNs or divergence:

1. Stop the affected run.
2. Preserve logs and checkpoint.
3. Reproduce on a short deterministic segment.
4. Check data corruption, loss scaling, LR schedule, and optimizer state.
5. Fix the underlying issue.
6. Apply the same correction to every scientifically comparable condition.
7. Document the change.

# Health and resource management

Monitor without flooding context:

- GPU utilization and memory.
- Training throughput.
- Loss and gradient norm.
- CPU RAM.
- Disk space.
- Checkpoint age.
- Process liveness.

Do not alter GPU voltage, clocks, firmware, or power limits.

If disk becomes constrained:

1. Stop before disk exhaustion.
2. Delete only regenerated caches or superseded checkpoints.
3. Preserve final checkpoints, current resumable checkpoints, metrics, manifests, and
   logs needed for auditing.
4. Record every deletion.

Use atomic writes and checksums so a power loss cannot make an incomplete checkpoint
look valid.

# Analysis

After all locked-scope runs complete, generate:

- `results/results.csv`
- `results/results.json`
- `figures/figure3a_replication.png`
- `figures/figure3a_replication.pdf`
- `reports/FINAL_REPORT.md`
- `reports/REPRODUCIBILITY.md`
- `reports/COMPUTE_ACCOUNTING.md`

The main figure should mirror Figure 3a conceptually:

Panel 1:
- X-axis: lambda.
- Y-axis: immediate MusicPile validation loss, L_im.
- One line for each MusicPile subset size.

Panel 2:
- X-axis: lambda.
- Y-axis: retained MusicPile validation loss after ChemPile fine-tuning, L_ret.
- One line for each MusicPile subset size.

Include uncertainty only where independent seeds or valid evaluation resampling support
it. Do not present minibatch variance as seed uncertainty.

Compute and report:

- Absolute and relative L_ret improvement from lambda=0 to each lambda.
- Change in L_im over lambda.
- Spearman correlation between lambda and L_ret for each dataset size.
- Linear and monotonic trend summaries.
- Forgetting:
  `delta_forgetting = L_ret - L_im`.
- ChemPile L_ft at every lambda.
- C4 L_pre at every lambda.
- Tokens and wall-clock time per stage.
- Total GPU-hours.
- Peak VRAM.
- Number and cause of retries.
- Exact number of C4 and MusicPile tokens seen.

The conclusion must distinguish:

- Replicated.
- Directionally replicated but weaker.
- Mixed/inconclusive.
- Did not replicate.

Do not require the expected trend in order to call the experiment complete.
An honest null result is a completed experiment.

# Independent final audit

Before completion, invoke a fresh-context review subagent that has not participated in
implementation. Give it the paper, frozen specification, run manifest, configurations,
tests, results, and figure.

The auditor must verify:

1. Every locked-scope condition exists.
2. Lambda token allocations match the intended definition.
3. Validation data was not trained on.
4. All lambda conditions use the same non-lambda hyperparameters.
5. Stage 2 used the full MusicPile subset for every lambda.
6. Stage 3 used the same ChemPile budget and LR for every lambda.
7. Metrics correspond to the correct checkpoints and datasets.
8. No failed or inconvenient run was omitted.
9. The figure is generated from the published CSV.
10. Commands can reproduce the analysis.
11. Results are not overstated.
12. Checkpoint and dataset manifests are internally consistent.

Resolve every audit failure and rerun affected analysis or experiments.

Create `reports/AUDIT.md` containing the auditor's findings and resolutions.

# Mandatory completion criteria

Do not emit the completion promise until a script such as
`scripts/verify_completion.py` exits successfully and verifies all of the following:

- `state/SCOPE_LOCK.json` exists and is valid.
- Every mandatory run in the locked scope has status `complete`.
- Every run has a frozen config, log, environment record, and valid metrics.
- Required checkpoints exist and pass integrity checks.
- `results/results.csv` contains all locked conditions exactly once.
- No required metric is missing, infinite, or NaN.
- Dataset and token-allocation audits pass.
- Unit and integration tests pass.
- The main PNG and PDF figures exist and regenerate from `results.csv`.
- Final, reproducibility, compute-accounting, and audit reports exist.
- The independent audit has no unresolved critical findings.
- The repository is clean or all remaining generated files are intentionally ignored.
- A final Git commit records the completed experiment.
- `docs/HANDOFF.md` contains exact commands for reproducing analysis and resuming any
  archived training.
- The final report states honestly whether the result replicated.

Completion means the planned experiment actually ran. It does not mean that the code
is ready to run, that a pilot finished, or that a trend was observed in partial data.

If an external blocker makes completion impossible, do not emit the completion promise.
Instead, create `BLOCKED.md` containing:

- The exact blocker.
- Evidence.
- Actions attempted.
- Current durable state.
- Minimal human action needed.
- Exact command to resume afterward.

When, and only when, all mandatory verification checks pass, output exactly:

<promise>EXPERIMENTS_COMPLETE</promise>
