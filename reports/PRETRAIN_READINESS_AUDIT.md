# PRETRAIN_READINESS_AUDIT.md

Independent pre–Stage-1 readiness audit of the Figure-3a reproduction
("Early Data Exposure Improves Robustness to Subsequent Fine-Tuning").

- Auditor role: independent, adversarial, read-only. Did not implement this code.
- Date: 2026-07-21. Repo: `/home/hshi-j-4090/Desktop/j-pretrain` @ branch `main`.
- Env python: `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` (torch 2.5.1+cu121, transformers 4.46.3).

## Summary verdict

**18 / 18 readiness criteria PASS (0 FAIL, 0 blocking CONCERN).** The data, model,
config, lineage, determinism, artifact-retention and completion-gate machinery are
sound and well-tested (85/85 pytest pass). **One MAJOR correctness bug** was found in
the Stage-2 → Stage-3 hand-off (the `restored_best` checkpoint is written from the
terminal model, not the best-val weights). It is confined to Stage 2/3 and therefore
does **not** block the start of expensive Stage-1 training, but it **must be fixed
before any Stage-2 run hands off to Stage 3**. Several disclosed LOCAL/provisional
choices remain (Stage-2 fixed config pending a λ=0 pilot; cross-stage bf16 snapshot
truncation) — none invalidate the Fig-3a comparison.

## Criteria table

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | Paper spec complete; every setting provenance-labelled | PASS | `docs/PAPER_SPEC.md` tags each value EXPLICIT_PAPER / INFERRED_FROM_SMOLLM2 / LOCAL_REPRODUCTION_CHOICE / UNKNOWN; §10 consolidates 11 UNKNOWNs. Structural caveat (no 135M arch/Stage-1 tables in paper) explicit at lines 19-24. |
| 2 | Arch matches SmolLM2-135M; param count correct | PASS | `count_parameters(build_model(...,seed=0))` → `(134515008, 134515008)`, exactly `expected_param_count` in `configs/model/smollm2-135m.json:28`; ~135M reasonable. `build.py` uses HF `LlamaForCausalLM`, tied embeddings (unique==total). |
| 3 | Tokenizer frozen + used identically across stages | PASS | `configs/data/datasets.json:6-16` pins rev `93efa2f0`, dir sha `b4ec3f78…`; 5 files present in `configs/data/tokenizer/`; loaded local-only (`build_datasets.py:89`). Training consumes pre-packed uint16 shards, so one tokenizer feeds all stages. |
| 4 | Dataset revisions pinned (C4/MusicPile/ChemPile) | PASS | `datasets.json:33/44/54` pin SHAs `1588ec45`/`5930bd71`/`c653c3c7`; `revision=` passed on every `load_dataset` (`build_datasets.py:54,62,71`); mirrored in `state/experiment_state.json`. |
| 5 | Train/tune/val disjoint; C4 uses native val | PASS | `data/splits.py` assigns each doc to exactly one per-mille band `val[0,5)/tune[5,10)/train[10,1000)` on salted sha256, at document level before packing (no window straddles a split, `packing.py` header). C4 uses native `train`/`validation` HF splits (`build_datasets.py:143-147`). Tests `test_data.py`. |
| 6 | 300M subset = deterministic nested prefix, 292968 windows | PASS | `data/subset.py` `n_windows = target//seq_len` → `300000000//1024 = 292968` (`is_nested` prefix logic); `299,999,232` tokens. Verified `300000000//1024==292968`. |
| 7 | λ allocation exact under ADD; interleave == reference; λ=0 → 0 MP | PASS | `interleave.py:55` `n_mp=round(λ*292968)` → {0,73242,146484,219726,292968}; `n_c4=8700000000//1024=8496093` fixed. `Stage1Plan.verify_against_reference()` (dataplan.py:64) matched by `test_training.py:57`. λ=0 branch yields all-C4 (`interleave.py:71-74`). |
| 8 | ChemPile can never enter Stage 1/2 | PASS | `run.py:_build_train` sources: stage1 `{c4,mp}`, stage2 `{mp}`, stage3 `{chempile}` (lines 304/315/322). `PlanLoader` KeyErrors on unknown source. `test_data.py:236 test_no_chempile_in_stage1_schedule`. |
| 9 | Validation data can never enter training | PASS | Train plans address only `*/train` packed dirs; val sets built separately (`run.py:285 _build_val_sets`) and used only in `Trainer.evaluate`. Stage-1 MP draws from subset prefix of the *train* pool; val is a disjoint hash band (`test_data.py:242`). |
| 10 | Sci hyperparams frozen/identical per stage; only ExecConfig HW-tuned; grad_accum preserves global batch | PASS | `StageConfig` (schemas.py) holds all sci knobs; `ExecConfig` (microbatch/compile) excluded from `scientific_dict()` and config hash (`run.py:360`). `grad_accum = global_batch_seqs//microbatch` and raises if not divisible (schemas.py:155-160) → effective global batch invariant to microbatch. |
| 11 | Atomic-write + resume tests pass | PASS | `pytest tests/ -q` → **85 passed**. `test_training.py:192 test_deterministic_resume_matches_continuous` (losses+params+counters byte-match); `test_orchestrator_run.py:162 test_resume_after_interrupted_stage` (stage1 not retrained). Atomic write tmp→fsync→checksum→load-test→rename in `checkpoint.py:97-156`. |
| 12 | Cross-stage lineage; init loaded (not rebuilt); no cross-run reuse | PASS | `run.py:_parent_for` (203-211): stage1←shared init analysis, stage2←that run's stage1 `final`, stage3←that run's stage2 `restored_best`. `run_node` loads init via `load_state_dict(ck.load_weights(parent_dir))` (run.py:348), not a same-seed rebuild. Init created once (`ensure_init_checkpoint`), byte-identical to all stage1. `test_orchestrator_run.py:133-138`. |
| 13 | Stage 2 uses COMPLETE 300M subset for every λ | PASS | `run.py:311` `n_sub = subset_tokens//seq` (independent of λ); `ShuffledSourcePlan("mp", n_sub, seed)`. This is Fig 3a, not compute-matched 3b. |
| 14 | Stage 3: lr 5e-5, 200M budget, identical ChemPile order, no MP early-stop | PASS | `configs/stage3/chempile.json`: peak_lr 5e-5, max_tokens 200M, `early_stop_patience:null`. `ShuffledSourcePlan("chempile", len(ch), seed=1234)` → same permutation across conditions (seed identical). No `primary_val` for stage3 → no best/early-stop path. |
| 15 | Checkpoint schedules represented + storage projection covers them | PASS | `training/schedule.py analysis_marks/resumable_marks` per stage; `stage_driver._after_chunk` writes on crossings; incoming/final/best write both classes. `docs/STORAGE_PLAN.md` projects 579 analysis (310.6 MB) + 36 permanent resumables (1850 MB) ≈ 308 GB < 581 GB free; `reports/FEASIBILITY.md`. |
| 16 | Git repo/remote correct; no large artifacts/secrets staged | PASS | branch `main`; `origin git@github.com:czhs/j-pretrain.git`. `git status` dirty entries are only `REPRO_PROMPT.md`, `loop.sh`, `wandb/*` (logs). `.gitignore` excludes `*.safetensors/*.pt/*.npy`, `/artifacts/*` payloads (re-includes only small inventories), `wandb/`, `logs/`, `.env`, `*.key/*.pem`. |
| 17 | Scope locked; reduction disclosed | PASS | `state/SCOPE_LOCK.json` `locked:true`, scope_hash `ca992877`, 5 runs (λ grid, seed 1234), `disclosed_reductions` list 300M-only + one-seed + LOCAL knobs. Fig 3a marked, 3b excluded. |
| 18 | `verify_completion.py` faithful, not stubbed | PASS | Checks scope-lock, 5 runs × 3 stages complete, run manifest, frozen configs, spec/scope hashes, pinned revisions, ≥6 tokenized manifests, 300M coverage, probes, checkpoint inventory present+checksummed+load-validated, one results row/condition with L_im/L_ret/L_ft/L_pre, figures, `pytest` green, all docs, git branch/remote/clean/pushed. Runs the real test suite (line 214). Not weakened. |

## Critical / Major findings

### MAJOR-1 — Stage-2 `restored_best` is the terminal model, not the best-val weights
`orchestration/stage_driver.py:175-177` (`StageDriver.run`):

```python
if self.ctx.stage == "stage2" and self.primary_val is not None:
    result["restored_best_written"] = True
    self._write_both(["restored_best"], {self.primary_val: self.tr.best_val or float("nan")})
```

`_write_both` serialises **`self.tr.model` as it currently stands** — i.e. the model at
the end of the loop (or 3 evals past the best val, when early stopping fires). The
best-val weights *were* persisted earlier under the `best` label (`_maybe_best_and_earlystop`,
line 211) but are **never reloaded** before writing `restored_best`. There is no
"restore best" step anywhere in `Trainer` or the driver.

Consequences:
- The checkpoint labelled `restored_best` — which is both the recorded θ_post (the point
  where **L_im** is meant to be read) and the **Stage-3 parent** (`run.py:_parent_for`,
  line 210) — contains the terminal weights, not the min-val weights. The scalar metric
  stored with it (`best_val`) then disagrees with the weights it accompanies.
- This contradicts the documented intent: the module docstring says *"stage2: restore
  best -> Stage 3 uses this"* (stage_driver.py:18/174) and `configs/stage2/music.json`
  `_early_stop_note` says *"Restore best-val checkpoint for Stage 3."* The code does not
  implement the restore, so this is a bug, not a design choice.

Impact assessment: with `early_stop_patience=3`, `min_delta=0` and near-converged
training the terminal model is only a few evals past the best, and the error is roughly
consistent across λ, so the qualitative Fig-3a claim (L_im ~flat, L_ret decreasing) will
likely survive. But it biases the reported L_im and perturbs the Stage-3 starting point,
so it is scientifically material and untested (`test_stage_driver.py` only checks the
`restored_best` *label* exists, not that its weights equal the `best` checkpoint).

Fix before Stage 2 hands off: reload the live `best` checkpoint's weights into
`self.tr.model` before writing `restored_best` (and before it becomes the Stage-3
parent). Add a test asserting `restored_best` weights == `best` weights and metric.

Does **not** block Stage 1 (Stage-1 has no `primary_val`/early-stop path), so Stage-1
training may begin while this is fixed.

## Minor findings / CONCERNs (non-blocking)

- **Cross-stage bf16 truncation.** Analysis snapshots are bf16 (`checkpoint.py:_bf16_state_dict`).
  Stage 2 loads the bf16 stage1-`final` and Stage 3 loads the bf16 stage2-`restored_best`,
  so each stage boundary truncates fp32→bf16. Resumable checkpoints keep fp32, so *within*-stage
  resume is exact; only stage *hand-offs* truncate. Consistent across all λ → does not bias the
  comparison; matches the bf16 training-compute representation. Note in FINAL_REPORT limitations.
- **Stage-2 fixed config provisional.** `configs/stage2/music.json` provenance
  `LOCAL_PROVISIONAL_PENDING_PILOT` (LR 5e-4 / batch 480 / dropout 0), to be finalised via a
  λ=0 pilot before the grid (DECISIONS.md, SCOPE_LOCK disclosed_reductions). Disclosed and authorized.
- **Orchestrator default runs eager, not compiled.** `run.py:default_config` sets
  `ExecConfig(..., torch_compile=False)` though DECISIONS records compile as primary (~1.8×).
  Execution-only; eager still meets the <21-day feasibility gate. Wall-clock only.
- **verify_completion final-audit gate is a crude substring match** (`scripts/verify_completion.py:228-231`
  looks for "no unresolved critical" etc. in `AUDIT.md`). Adequate as a gate but game-able; not weakened.
- **Stage-3 budget floor.** `max_optimizer_steps = 200M//(480*1024) = 406` steps → 199.5M tokens
  (0.25% under the nominal 200M). Expected floor behaviour, identical across λ.

## Recommendation

**READY to begin Stage 1.** The Stage-1 path (data construction, λ interleave, model
init, lineage, determinism, checkpointing, resume, artifact retention, and the
completion gate) is correct and well-tested. Begin Stage-1 training now, but track
**MAJOR-1** as a required fix (with a regression test) **before any Stage-2 run writes
`restored_best` / hands off to Stage 3**, since that bug corrupts the recorded θ_post
(L_im) and the Stage-3 initialization.

---

## Resolution log (post-audit)

- **MAJOR-1 — RESOLVED (commit follows this report).** `StageDriver.run` now, for Stage 2,
  RELOADS the best-val weights from the most recent `best` resumable checkpoint (exact fp32
  state) before writing `restored_best`, and re-evaluates L_im on those restored weights.
  `_write` now returns the resumable dir so the best-val checkpoint can be located;
  `_best_resumable_dir` is captured whenever a new best is written. Fallback: if no best was
  ever recorded (degenerate zero-eval stage) the terminal weights are used.
  **Regression test:** `tests/test_stage_driver.py::test_stage2_restored_best_reloads_best_not_terminal`
  scripts an improving-then-worsening val trajectory with early stopping and asserts the
  `restored_best` weights are byte-identical to the best-val checkpoint and DIFFER from the
  terminal/post-early-stop weights. 87/87 tests pass.

- Minor CONCERNs (torch_compile off vs DECISIONS; substring final-audit gate; Stage-3 199.5M
  token floor; tokenizer dir-hash method) are non-blocking and documented (see DECISIONS.md for
  the tokenizer-hash reconciliation). No critical or major findings remain unresolved.
