# NEXT ACTION

**Phase:** impl_foundation_done (config+model built & tested; data access validated)

**Process running?** No training/GPU job. No detached runner. Nothing to babysit.

**CRITICAL OPS NOTE:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY.
Do NOT use `conda run -n jpre` (it swallows heredoc stdout → false "hang"). Set
`HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false` for HF ops.

**Last verified (2026-07-21):**
- Spec frozen + audited (11/11). Env jpre ready. Benchmark: compile mb8=94k tok/s. Feasibility PASS.
- Config system + 135M model builder: 12 tests pass. Model = 134,515,008 params, tied embeds.
  Commits: 3b40459 (spec freeze), d4a1bd4 (config+model).
- Data access validated (streaming): C4(en).text OK; MusicPile.text OK (Human/Assistant fmt);
  ChemPile needs config (4 available). Tokenizer SmolLM2-135M OK (vocab 49152, bos=eos=0, pad None).

**Next exact action (fresh session):**
1. `cat state/experiment_state.json state/NEXT_ACTION.md state/DECISIONS.md`; `pytest tests/ -q`.
2. Build the DATA PIPELINE (`src/j_pretrain/data/`) + tests + `docs/DATA_AUDIT.md`:
   - Freeze tokenizer (save to configs/data/ or artifacts; record revision).
   - Resolve HF dataset revisions (pin commit hashes); write configs/data/*.
   - DECIDE ChemPile config set (likely all 4 education configs) → record in DATA_AUDIT.
   - Deterministic download+tokenize+pack to memory-mapped shards (seq_len 1024). DO NOT load full
     corpora into RAM. Only fetch C4 shards needed for 8.7B + val margin.
   - Deterministic train/tune/val splits with NO document overlap; no ChemPile in Stage1/2; no
     MusicPile-val leakage. Record doc IDs, seeds, manifests, hashes.
   - 300M MusicPile subset construction (full pool); λ interleave scheduler (C4 fixed 8.7B +
     λ·300M added, uniform interleave). Exact per-source token counting.
   - Fixed probes (C4/MusicPile/ChemPile) → artifacts/probes/.
   - Tests: λ allocation, per-source token counts, subset construction, split disjointness,
     determinism, tokenizer determinism, no-leakage, probe determinism.
3. THEN: training/artifacts modules (resumable trainer, atomic ckpt, inventory) + resume test.
4. THEN: orchestration (DAG, GPU lock, tmux runner) + storage bench + STORAGE_PLAN.md.
5. THEN: PRETRAIN_READINESS_AUDIT → SCOPE_LOCK.json + FEASIBILITY.md gate → Stage 1 training.

**Must NOT repeat:** preflight, env, spec, benchmark, config/model impl, data-access probe (all done).
**Must NOT:** commit weights/datasets/logs; use `conda run`; reduce scope; train before readiness audit.

**Command to resume orchestrator:** none yet (orchestrator not built).
