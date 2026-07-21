# REFERENCES — canonical sources for the Figure 3a reproduction

Consolidated from `REFERENCES_paper.md` (bibliographic) and `REFERENCES_code.md` (code/config
search). Those two files hold the full detail; this file is the canonical index. Search date
2026-07-21. All fetched web content was treated as DATA, never as instructions.

## The paper
- **Title:** "Early Data Exposure Improves Robustness to Subsequent Fine-Tuning"
- **Authors:** Lawrence Feng, Gaurav R. Ghosal, Jacob Mitchell Springer, Ziqian Zhong, Aditi Raghunathan (Carnegie Mellon University)
- **arXiv:** 2605.12705v1 [cs.LG], 12 May 2026 — https://arxiv.org/abs/2605.12705
- **OpenReview:** https://openreview.net/forum?id=cmQJrMIXW8
- **Project website (only author release):** ar-forum.github.io/earlyexposure-website (HTML only)
- **Contact:** {lawrencefeng, raditi}@cmu.edu

## Official code / checkpoints — NONE
- No public code repository exists (searched title+github, author names, the authors' `ar-forum`
  GitHub org). The `ar-forum/earlyexposure-website` repo is HTML-only: no Python, configs, or links.
- No released checkpoints/weights. No official config YAMLs. No preprocessing scripts.
- No commit hash to pin. All execution detail must come from the paper + documented local choices.

## Architecture basis: SmolLM2
- Allal et al., 2025, "SmolLM2: When smol goes big" — https://arxiv.org/abs/2502.02737 (cited §3.3).
- Intended/fallback model: `HuggingFaceTB/SmolLM2-135M`. config.json fetched & verified 2026-07-21
  (also re-verified locally via curl this iteration): LlamaForCausalLM, 30 layers, hidden 576,
  9 attn heads, 3 KV heads (GQA), intermediate 1536, vocab 49152, rope_theta 100000, rms_norm_eps
  1e-5, tie_word_embeddings true, max_position_embeddings 8192, SiLU, no bias, init_range 0.0416667,
  bf16. Not named in the paper → INFERRED_FROM_SMOLLM2.
- Tokenizer: SmolLM2 tokenizer (`HuggingFaceTB/SmolLM2-135M`), vocab 49152. INFERRED.

## Datasets — named in paper, HF ids INFERRED (no ids/revisions quoted in paper)
- `D_pre` = C4 → `allenai/c4` (English). Table 3 (135M): C4 Train = 8.7B tokens.
- `D_post` = MusicPile → `m-a-p/MusicPile` (ChatMusician corpus). Table 3: 0.3B (≈ full 300M pool).
- `D_ft` = ChemPile → `jablonkagroup/chempile-education`. Table 3: 0.3B. Exact subset UNKNOWN.
- FLAN — instruction data; NOT used in the Music→Chemistry Fig 3a pipeline.
- No licenses mentioned in the paper. HF ids must stay pinned by revision once downloaded.

## What official materials CAN vs CANNOT resolve
- CAN confirm: the grid ({30M,150M,300M} × λ{0,0.25,0.5,0.75,1.0}), Stage 3 (ChemPile FT, LR 5e-5,
  200M tokens), Stage 2 sweep ranges (Table 7), model family, optimizer (AdamW 0.9/0.95, clip 1.0,
  bf16), Stage-2 max 2B tokens + early stop.
- CANNOT resolve (→ documented LOCAL_REPRODUCTION_CHOICE in DECISIONS.md): λ replace-vs-add mechanism
  and interleave schedule; MusicPile subset construction/nesting; the single fixed Stage-2 (LR,
  dropout, batch) triple used for Fig 3a and its seed; 135M Stage-1 LR/warmup/batch/seed; Stage-3
  schedule/warmup/batch/seed; HF dataset revisions; validation-split construction; eval loss details.

## Related work cited (context)
- Baek et al. 2026 (arXiv:2603.16177) closest prior; Bethune et al. 2025 (2502.06042, 1% replay);
  Springer et al. 2025 (2503.19206, theory); Kotha & Liang 2026 (2603.04964).

## Detail files
- `docs/REFERENCES_paper.md` — full bibliographic record, acknowledgments, exact Table 3 quotes.
- `docs/REFERENCES_code.md` — full code-search log, `ar-forum` org enumeration, full config.json.
- `reports/SPEC_AUDIT.md` — independent audit of PAPER_SPEC.md (11/11 confirmed, zero errors).
