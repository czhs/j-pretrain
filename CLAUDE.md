# j-pretrain — Figure 3a reproduction (Ralph autonomous loop)

## Scientific objective
Reproduce **Figure 3a** of "Early Data Exposure Improves Robustness to Subsequent Fine-Tuning".
Test whether showing a fraction `lambda` of the eventual post-training corpus (MusicPile) during
Stage-1 C4 pretraining improves retention of the post-trained capability after Stage-3 ChemPile
fine-tuning. Primary grid: **1 subset size (300M MusicPile) × lambda ∈ {0,0.25,0.5,0.75,1.0}**.
This is a deliberately reduced Fig 3a (paper sweeps 30M/150M/300M; only 300M here). Reduction is
user-authorized — recorded in `state/SCOPE_LOCK.json`, disclosed in every report. NOT Fig 3b.

Three stages per lambda: (1) C4 pretrain w/ early MusicPile exposure → (2) full 300M MusicPile
post-train to convergence (measure L_im) → (3) ChemPile FT @ lr 5e-5, 200M tokens
(measure L_ret, L_ft, L_pre on MusicPile/ChemPile/C4 val). Key claim: L_im ~flat across lambda;
L_ret improves as lambda increases.

## Canonical state (READ THESE FIRST every iteration)
1. `CLAUDE.md` (this file — invariants only)
2. `state/experiment_state.json` — machine-readable canonical state
3. `state/NEXT_ACTION.md` — what to do right now (<50 lines)
4. last line of `state/iteration_ledger.jsonl`
5. `state/DECISIONS.md`, `state/SCOPE_LOCK.json` (once locked)
Do NOT scan whole repo or reload full logs each iteration.

## Repo layout
- `src/j_pretrain/` — {config,data,models,training,evaluation,orchestration,artifacts,analysis}
- `configs/{data,model,stage1,stage2,stage3,experiments}` — frozen YAML/JSON configs
- `scripts/` — CLI entrypoints incl. `verify_completion.py`
- `tests/` — unit + integration (resume) tests
- `docs/` — PAPER_SPEC, REFERENCES, COMPUTE_PLAN, STORAGE_PLAN, DATA_AUDIT, FAILURE_PLAYBOOK, FAILURES, HANDOFF, ENVIRONMENT
- `reports/` — FINAL_REPORT, REPRODUCIBILITY, COMPUTE/STORAGE_ACCOUNTING, AUDIT, FEASIBILITY, audits
- `runs/manifest.jsonl`, `results/`, `figures/`, `artifacts/` (weights — gitignored)

## Artifact retention (CRITICAL)
Artifact root: env `J_PRETRAIN_ARTIFACT_ROOT` (resolved path in experiment_state.json).
Every valid checkpoint is PERMANENT. Never delete/overwrite/quantize/dedup. Atomic writes
(tmp dir → fsync → checksum → load-test → rename). Two classes: analysis snapshots + full
resumable. Append-only inventories under `artifacts/`.

## Commands
- Env: conda env `jpre` (pinned; see docs/ENVIRONMENT.md / pyproject.toml)
- Tests: `python -m pytest tests/ -q`
- Verify completion: `python scripts/verify_completion.py`
- Orchestrator: `python -m j_pretrain.orchestration.run` (detached via tmux; ONE GPU job at a time)

## Git
- branch `main`, remote `git@github.com:czhs/j-pretrain.git`.
- Commit code/config/state/docs/small results ONLY. Never weights/datasets/logs/secrets.
- NEVER: `git push --force*`, `git reset --hard`, `git clean -fd`, history rewrite.
- Push to origin/main after each stable milestone.

## Ralph rules
- Reconstruct state from files each iteration; context is ephemeral. No busy-poll (~10 min health
  checks). Training runs DETACHED (tmux/durable runner), survives usage-limit gaps — never restart a
  healthy run. Usage-limit pauses are NOT blockers. External blocker → `BLOCKED.md` (commit+push).
- Do NOT reduce scope for slowness/unfavorable results. Scope changes only per SCOPE_LOCK rules.

## Completion
Emit `<promise>EXPERIMENTS_COMPLETE</promise>` ONLY after `python scripts/verify_completion.py`
exits 0, all audits pass, final commit pushed to origin/main. Never on partial progress.
