# NEXT ACTION

**Phase:** preflight_and_spec_extraction (very early — scaffolding just built)

**Process running?** No training process. No GPU job.

**Last verified:** Preflight all green (2026-07-21):
- paper.pdf readable (29 pp); GPU RTX 4090 free; datasets C4/MusicPile/ChemPile reachable on HF;
  git ls-remote OK; tools present. Disk 590G free / 83% used (TIGHT — watch storage gate).
- Base conda env has NO ML stack (no torch/transformers). Must build a pinned env.

**Next exact action (fresh session):**
1. `cat state/experiment_state.json state/NEXT_ACTION.md`
2. If `docs/PAPER_SPEC.md` absent/incomplete → extract paper spec (Sec 3.3, 4.1, 4.2, App A, Fig 3, dataset+optimizer tables) and search for official author repo. Write docs/PAPER_SPEC.md + docs/REFERENCES.md with EXPLICIT/INFERRED/UNKNOWN tags.
3. Build reproducible Python env (create conda env `jpre`, pin torch+cu121, transformers, datasets, tokenizers, safetensors, numpy, pandas, matplotlib, scipy, pytest). Record versions in docs/ENVIRONMENT.md + pyproject.toml.
4. Then: dataset pipeline (tokenizer freeze, C4/MusicPile/ChemPile manifests, 300M MusicPile subset, splits) with tests.

**Must NOT repeat:** dir scaffolding, .gitignore, preflight (all done). Do not commit weights/datasets/logs.

**Command to resume orchestrator:** none yet (no runs launched).
