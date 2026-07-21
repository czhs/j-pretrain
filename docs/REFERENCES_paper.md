# REFERENCES_paper.md — Bibliographic record for the reproduced paper

## The paper
- **Title:** "Early Data Exposure Improves Robustness to Subsequent Fine-Tuning"
- **Authors:** Lawrence Feng, Gaurav R. Ghosal, Jacob Mitchell Springer, Ziqian Zhong, Aditi Raghunathan
- **Affiliation:** Carnegie Mellon University
- **Contact (EXPLICIT, p.1):** `{lawrencefeng, raditi}@cmu.edu`
- **Venue / status:** Preprint. arXiv:2605.12705v1 [cs.LG], dated 12 May 2026.
- **Project website (EXPLICIT, p.1):** `ar-forum.github.io/earlyexposure-website`
- **Contributions (p.15):** Lawrence Feng led the project and conducted all main experiments; the
  other authors contributed to direction, experimental design, analysis, and writing.
- **Acknowledgments (p.15):** Apple, Google, Jane Street, NSF, and the FLAME cluster at CMU;
  NSF Graduate Research Fellowship Grant No. DGE2140739. Thanks to Christina Baek and Kevin Li.

## No explicit code / dataset URLs
- The PDF gives NO GitHub repository URL and NO HuggingFace dataset identifiers in text. The only
  author URL is the project website above. Dataset HF ids below are inferred, NOT quoted from paper.

## SmolLM2 reference (architecture basis)
- Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo,
  Lewis Tunstall, ... Leandro von Werra, Thomas Wolf. **"SmolLM2: When smol goes big — data-centric
  training of a small language model", 2025.** URL: https://arxiv.org/abs/2502.02737
- Cited in Section 3.3 as the source of the "SmolLM2-style architecture (Allal et al., 2025)".
- Inferred official model for the 135M reproduction: HuggingFace `HuggingFaceTB/SmolLM2-135M`
  (config + tokenizer). NOT named in the paper — INFERRED.

## Datasets — identities and inferred HF ids
The paper names datasets only by short name (Table 1, Table 3). No HF ids are quoted. Inferred ids:
- **C4** (`D_pre`) — "General web text pretraining corpus" (Table 3, EXPLICIT). Inferred HF id:
  `allenai/c4` (English). Underlying: Raffel et al. C4. Not quoted in paper.
- **MusicPile** (`D_post`) — "Music-domain text corpus" (Table 3, EXPLICIT). Inferred HF id:
  `m-a-p/MusicPile` (ChatMusician corpus). Not quoted in paper.
- **ChemPile** (`D_ft`) — "Chemistry-domain text corpus" (Table 3, EXPLICIT). Inferred: the ChemPile
  corpus (Jablonka group). Exact HF id / revision not quoted — UNKNOWN.
- **FLAN** — "Instruction-tuning dataset" (Table 3, EXPLICIT). Not used in the Fig 3a pipeline.

## Licenses
- No licenses are mentioned anywhere in the paper for models, datasets, or code.

## Key related works cited (for context; arXiv ids as printed)
- Baek et al., 2026 — "The finetuner's fallacy: When to pretrain with your finetuning data" — arXiv:2603.16177 (closest prior; mixing post-training data into pretraining).
- Bethune et al., 2025 — "Scaling laws for forgetting during finetuning with pretraining data injection" — arXiv:2502.06042 (source of the 1% replay fraction).
- Springer et al., 2025 — "Overtrained language models are harder to fine-tune" — arXiv:2503.19206 (theory basis, Appendix C).
- Kotha & Liang, 2026 — "Replaying pre-training data improves fine-tuning" — arXiv:2603.04964.
- Biderman et al., 2024 — "LoRA learns less and forgets less" — arXiv:2405.09673.
- Gidel et al., 2019 — "Implicit regularization of discrete gradient dynamics in linear neural networks" — arXiv:1904.13262 (sequential feature-learning theory).
- Gadre et al., 2024 (arXiv:2403.08540), Du et al., 2024 (NeurIPS), Chen et al., 2025 (arXiv:2410.08527) — loss-as-capability-proxy justification.

## Exact-quote notes (dataset identifiers)
- Table 3 verbatim rows (135M): "C4 | Train | 8.7B | General web text pretraining corpus";
  "MusicPile | Train | 0.3B | Music-domain text corpus";
  "ChemPile | Train | 0.3B | Chemistry-domain text corpus";
  "FLAN | Train | 0.3B | Instruction-tuning dataset".
- No HuggingFace identifier strings appear anywhere in the paper text — all HF ids above are inferred
  and must be confirmed against official code before freezing data sources.
