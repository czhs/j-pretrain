# NEXT ACTION

**Phase:** spec_frozen_impl_pending (spec extracted + independently audited; benchmark done)

**Process running?** No training/GPU job. No detached runner. Nothing to babysit.

**Last verified (2026-07-21):**
- Env `jpre` fully built (torch 2.5.1+cu121, transformers 4.46.3, wandb 0.18.7). GPU free.
- PAPER_SPEC.md extracted + independently audited → 11/11 confirmed, 0 errors (reports/SPEC_AUDIT.md).
- REFERENCES.md consolidated. No official code exists → key unknowns are documented LOCAL choices.
- Model builds to 134.52M params (SmolLM2-135M config, verified via curl).
- Throughput bench: compile mb8 = 94.3k tok/s @9.65GB; eager mb8 = 52.5k tok/s. Feasibility PASS
  (~7d compile / ~12-16d eager incl overhead; both < 21d gate). See docs/COMPUTE_PLAN.md (prelim).
- Key decisions FROZEN in state/DECISIONS.md: arch, 8.7B C4, λ=ADD policy, exec=compile.

**Next exact action (fresh session):**
1. `cat state/experiment_state.json state/NEXT_ACTION.md state/DECISIONS.md`
2. Build the typed codebase under `src/j_pretrain/`:
   - `config/` — dataclasses + deterministic config hashing + run-id derivation.
   - `models/` — SmolLM2-135M builder from frozen config; param-count assert; save init checkpoint.
   - `data/` — tokenizer freeze; C4/MusicPile/ChemPile streaming+tokenize to mmap shards; deterministic
     splits (train/tune/val, no overlap); 300M MusicPile subset; λ interleave scheduler; probes.
   - `training/` — resumable trainer (AdamW, cosine, grad-accum, bf16, compile, atomic ckpt, RNG+cursor).
   - `evaluation/` — token-weighted CE eval on frozen val sets (L_im/L_ret/L_ft/L_pre).
   - `artifacts/` — atomic checkpoint writer, inventory (jsonl), checksums, load-validation.
   - `orchestration/` — DAG, GPU lock, run queue, tmux durable runner, resume logic.
   - `analysis/` — results tables + figure3a regen.
3. Write `configs/{model,data,stage1,stage2,stage3,experiments}` frozen YAML/JSON.
4. Write tests (see CLAUDE.md required-tests list) incl. deterministic resume integration test.
5. THEN: real data pipeline + DATA_AUDIT.md (independently audited) + storage bench.
6. THEN: PRETRAIN_READINESS_AUDIT → SCOPE_LOCK.json + FEASIBILITY.md gate → Stage 1.

**Must NOT repeat:** preflight, env build, spec extraction/audit, prelim benchmark (all done+committed).
**Must NOT:** commit weights/datasets/logs; reduce scope; start training before readiness audit + scope lock.

**Command to resume orchestrator:** none yet (no runs launched; orchestrator not built).
