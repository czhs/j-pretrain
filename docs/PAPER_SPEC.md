# PAPER_SPEC.md — Frozen scientific specification for Figure 3a reproduction

Paper: "Early Data Exposure Improves Robustness to Subsequent Fine-Tuning"
(Feng, Ghosal, Springer, Zhong, Raghunathan; Carnegie Mellon University).
arXiv:2605.12705v1 [cs.LG], 12 May 2026. Project site: ar-forum.github.io/earlyexposure-website

Target reproduction: **Figure 3a** (left half of Figure 3, "Varying mixture fraction lambda").
NOT Figure 3b (the compute-matched setting).

## Provenance tags
- `EXPLICIT_PAPER` — stated verbatim in paper text/table/figure.
- `OFFICIAL_CODE` — from authors' released code (NOT consulted here; none available at spec time).
- `OFFICIAL_AUTHOR_MATERIAL` — author website/appendix material beyond the PDF (none used).
- `INFERRED_FROM_SMOLLM2` — supplied from the official SmolLM2-135M config because the paper is
  silent; MUST be verified against HuggingFace `HuggingFaceTB/SmolLM2-135M` before freezing.
- `LOCAL_REPRODUCTION_CHOICE` — a choice this reproduction makes where the paper is silent/ambiguous.
- `UNKNOWN` — not determinable from the paper; would need official code/authors to resolve.

> CRITICAL STRUCTURAL FACT: **The paper gives NO 135M architecture table and NO 135M Stage-1
> pretraining hyperparameter table.** Appendix A.1 Table 2 is the *1B* architecture ("custom config
> interpolated from SmolLM2 family"); Appendix A.4 Table 6 is the *1B* Stage-1 config. All 135M
> architecture numbers below the vocab/context/RoPE that appear in the 1B table are therefore
> `INFERRED_FROM_SMOLLM2` or `UNKNOWN`, not `EXPLICIT_PAPER`. Do not represent them as reported.

---

## 1. Model architecture (135M, SmolLM2-style)

Section 3.3 (EXPLICIT): "we use a SmolLM2-style architecture (Allal et al., 2025) at two scales:
our primary experiments use a 135M-parameter model, and we additionally run a 1B-parameter variant".
No 135M architecture table is provided. Values below are the official SmolLM2-135M
(`HuggingFaceTB/SmolLM2-135M`) config unless a paper-wide value applies.

| Field | Value | Provenance |
|---|---|---|
| Parameter count | ~135M (SmolLM2-135M ≈ 134.5M) | EXPLICIT_PAPER (scale name); count INFERRED_FROM_SMOLLM2 |
| Layers | 30 | INFERRED_FROM_SMOLLM2 |
| Hidden dim | 576 | INFERRED_FROM_SMOLLM2 |
| Attention heads (query) | 9 | INFERRED_FROM_SMOLLM2 |
| KV heads / query groups (GQA) | 3 KV heads → GQA (3 query groups) | INFERRED_FROM_SMOLLM2 (NB: paper's 1B uses MHA per Table 2; 135M official config uses GQA) |
| Head dim | 64 (576/9) | INFERRED_FROM_SMOLLM2 |
| MLP intermediate dim | 1536 | INFERRED_FROM_SMOLLM2 |
| Activation | SiLU (gated / SwiGLU MLP) | INFERRED_FROM_SMOLLM2 |
| Vocab size | 49,152 | EXPLICIT_PAPER (Table 2, 1B) + INFERRED_FROM_SMOLLM2 (same for 135M) |
| Training sequence length | 1,024 | EXPLICIT_PAPER (Table 2 "1,024 (training)"; assumed same for 135M) |
| Max context length | 8,192 | EXPLICIT_PAPER (Table 2 "8,192 (max)"); INFERRED_FROM_SMOLLM2 for 135M |
| Normalization | RMSNorm | EXPLICIT_PAPER (Table 2); INFERRED_FROM_SMOLLM2 for 135M |
| RMSNorm eps | 1e-5 | INFERRED_FROM_SMOLLM2 |
| Position encoding / RoPE base | RoPE, base=100,000 | EXPLICIT_PAPER (Table 2 "RoPE (base=100,000)"); INFERRED_FROM_SMOLLM2 for 135M |
| Weight tying (embed↔LM head) | Tied (`tie_word_embeddings=true`) | INFERRED_FROM_SMOLLM2 |
| Attention/MLP bias | No bias | INFERRED_FROM_SMOLLM2 |
| Initialization | Normal; `initializer_range` per SmolLM2-135M config (verify exact value in HF config) | INFERRED_FROM_SMOLLM2 |
| Dropout (architecture) | 0.0 in the model; dropout only added as a Stage-2 intervention | INFERRED_FROM_SMOLLM2 / EXPLICIT_PAPER (Table 7 dropout is a Stage-2 sweep knob) |

Note: the paper's 1B Table 2 explicitly uses MHA (27 heads = 27 query groups, head dim 64,
hidden 1728, 24 layers, intermediate 4608). This confirms the authors build "custom configs
interpolated from the SmolLM2 family" and do NOT necessarily copy the official per-scale config.
Whether their 135M used official GQA(3-KV) or an MHA variant is `UNKNOWN`; default to official
SmolLM2-135M (GQA) as `LOCAL_REPRODUCTION_CHOICE` and flag for verification.

## 2. Tokenizer

| Field | Value | Provenance |
|---|---|---|
| Tokenizer | SmolLM2 tokenizer (GPT-2-style BPE; `HuggingFaceTB/SmolLM2-135M`) | INFERRED_FROM_SMOLLM2 (paper only says "SmolLM2-style") |
| Vocab size | 49,152 | EXPLICIT_PAPER (Table 2) |

---

## 3. Datasets

### Identities (Table 1, Table 3; Section 3.2)
- `D_pre` = **C4** — "General web text pretraining corpus" (EXPLICIT_PAPER). Fixed across all
  pipelines. Exact HF id not given; presumed `allenai/c4`, English split. HF id + revision = UNKNOWN.
- `D_post` = **MusicPile** — "Music-domain text corpus" (EXPLICIT_PAPER). For Fig 3a, D_post ⊂
  MusicPile. Exact HF id not given; MusicPile is `m-a-p/MusicPile` (from ChatMusician). Id = UNKNOWN/INFERRED.
- `D_ft` = **ChemPile** — "Chemistry-domain text corpus" (EXPLICIT_PAPER). Exact HF id not given = UNKNOWN.
- FLAN — instruction dataset; NOT used in the Fig 3a (Music→Chemistry) pipeline.

### Dataset statistics, 135M experiments (Table 3, EXPLICIT_PAPER)
| Dataset | Split | Tokens |
|---|---|---|
| C4 | Train | 8.7B |
| MusicPile | Train | 0.3B |
| ChemPile | Train | 0.3B |
| FLAN | Train | 0.3B |

(1B experiments, Table 4: C4 Train = 19.7B; others 0.3B. Not used for Fig 3a.)

### D_post = 300M subset & nested subsets
- Section 4.1 (EXPLICIT): "We study three post-training dataset sizes |D_post| ∈ {30M, 150M, 300M}
  where D_post ⊂ MusicPile." Figure 3a plots all three sizes; **our reduced reproduction uses only
  the 300M line** (recorded in `state/SCOPE_LOCK.json`; user-authorized).
- Table 3 lists MusicPile Train = 0.3B, i.e. the full available MusicPile pool ≈ 300M tokens; the
  30M/150M subsets are drawn from it.
- **Nesting procedure: UNKNOWN.** The paper never states whether 30M⊂150M⊂300M are strictly nested
  prefixes, nor the sampling/seed. For the 300M-only reproduction this is moot (300M = full pool),
  but if smaller sizes are ever added it must come from official code. LOCAL choice for 300M:
  use the entire MusicPile 300M train pool.

---

## 4. Stage 1 — Upstream pretraining (C4 + early MusicPile exposure)

### Token budget — the ~10B vs 8.7B conflict (RESOLVED)
- Section 3.3 (EXPLICIT): "For the 135M experiments, we pretrain on approximately 10B tokens from
  D_pre, optionally with early exposure to D_post during Stage 1."
- Table 3 (EXPLICIT): C4 Train tokens = **8.7B**.
- **Resolution (LOCAL_REPRODUCTION_CHOICE):** Treat **8.7B C4 tokens** as the authoritative Stage-1
  C4 training budget — Table 3 is the precise, dataset-specific figure; "approximately 10B" is loose
  prose rounding (and, with up to +0.3B MusicPile at lambda=1, total upstream ≈ 9.0B, still described
  loosely as "~10B"). Do NOT scale C4 to a literal 10B. Disclose both figures in every report.

### Lambda early-exposure scheduling: ADD vs REPLACE
- Section 3.1 (EXPLICIT): "the upstream developer additionally mixes a fraction lambda ∈ [0,1] of
  the post-training dataset D_post into this stage. Here, lambda=0 denotes no exposure ... lambda is
  at most one, denoting at most one pass over D_post during pretraining." → lambda-fraction means
  **lambda · |D_post| MusicPile tokens** mixed into Stage 1 (lambda=1 → one full 300M pass;
  lambda=0.25 → 75M tokens; etc.).
- Section 4.2 (EXPLICIT, key disambiguator): the Fig 3a experiments "do not isolate whether the
  benefit comes from *when* D_post is introduced or simply from the model seeing *more total* D_post
  tokens." The compute-matched Fig 3b is the one that "fix[es] the total number of D_post tokens seen
  across Stage 1 pretraining and Stage 2 post-training." → **Therefore in Fig 3a, MusicPile exposure
  is ADDED (total MusicPile seen grows with lambda); it is NOT compute-matched.**
- Word "from D_pre" in Section 3.3 ("10B tokens from D_pre, optionally with early exposure to
  D_post") reads as: C4 budget is fixed and MusicPile is added on top. → **Policy (LOCAL choice):
  Stage 1 = fixed 8.7B C4 tokens + interleaved lambda·300M MusicPile tokens.** The exact interleaving
  schedule (uniform mixing throughout vs. front-loaded "early", and shuffling seed) is `UNKNOWN` and
  would need official code. "Early exposure" in the title/Section 3.1 suggests mixing throughout
  Stage 1 rather than a dedicated early block; default to uniform random interleave at the target
  mixture ratio unless code says otherwise.
- Grid: **lambda ∈ {0, 0.25, 0.5, 0.75, 1.0}** (EXPLICIT_PAPER, Section 4.1).

### Stage-1 optimizer / schedule (135M) — mostly UNKNOWN, 1B values shown for reference
Paper gives NO 135M Stage-1 table. Table 5 (all experiments) + Table 6 (1B only) give:
| Field | 135M value | Provenance |
|---|---|---|
| Optimizer | AdamW | EXPLICIT_PAPER (Table 5, all experiments) |
| Betas | beta1=0.9, beta2=0.95 | EXPLICIT_PAPER (Table 5) |
| Grad clip | 1.0 (max norm) | EXPLICIT_PAPER (Table 5) |
| Precision | bf16-mixed | EXPLICIT_PAPER (Table 5) |
| LR schedule | linear warmup + cosine decay | EXPLICIT_PAPER (Section 3.3 states this for Stage 2; Table 6 shows cosine for 1B Stage 1; assume same shape for 135M Stage 1) — LOCAL for 135M |
| Peak LR | UNKNOWN (1B used 5e-4) | UNKNOWN / 1B ref EXPLICIT (Table 6) |
| Min LR | UNKNOWN (1B used 5e-5) | UNKNOWN / 1B ref EXPLICIT |
| Warmup steps | UNKNOWN (1B used 1,000) | UNKNOWN / 1B ref EXPLICIT |
| Global batch size | UNKNOWN (1B used 512) | UNKNOWN / 1B ref EXPLICIT |
| Weight decay | UNKNOWN (Stage-2 used 0.1) | UNKNOWN |
| Seed | UNKNOWN (1B Stage-1 seed=42) | UNKNOWN / 1B ref EXPLICIT |

---

## 5. Stage 2 — Upstream post-training on FULL 300M MusicPile (measure L_im)

- Runs on the FULL D_post subset (300M for our reproduction), NOT the lambda-fraction.
- Section 3.3 (EXPLICIT): "we perform Stage 2 post-training on D_post using AdamW with linear warmup
  and cosine decay. In all but the compute-matched experiments, training proceeds exclusively on
  D_post, with no restriction on dataset repetitions: we apply early stopping and continue training
  as long as validation loss on D_post improves (up to a maximum budget of 2B tokens)."
- Section 4.1 (EXPLICIT): Fig 3a "fix[es] the Stage 2 post-training procedure and vary only how much
  of D_post is seen during Stage 1 pretraining" and post-trains "on D_post until convergence using a
  fixed hyperparameter configuration." → **Fig 3a uses ONE fixed Stage-2 config, NOT the sweep.**
- L_im := L(theta_post; D_post) = MusicPile validation loss (Section 3, EXPLICIT).

### Stage-2 hyperparameters (135M) — Table 7 is a SWEEP; Fig 3a's single fixed config is UNKNOWN
Table 7 (EXPLICIT, "FFT hyperparameter search space for Stage 2 post-training (135M)"):
| Field | Value(s) | Provenance |
|---|---|---|
| Learning rate (peak) | search {1e-4, 2e-4, 5e-4, 1e-3, 5e-3} | EXPLICIT_PAPER (range) |
| Min LR | 5e-5 (cosine decay target) | EXPLICIT_PAPER |
| Dropout | search {0.0, 0.02, 0.05} (embed/attn/resid/mlp) | EXPLICIT_PAPER (range) |
| Weight decay | 0.1 (fixed) | EXPLICIT_PAPER |
| Warmup steps | 500 (fixed) | EXPLICIT_PAPER |
| Global batch size | search {192, 480, 896} | EXPLICIT_PAPER (range) |
| Max tokens | 2B (with early stopping) | EXPLICIT_PAPER |
| Early-stopping patience (135M) | UNKNOWN (1B Table 8 used 3) | UNKNOWN / 1B ref |
| Eval interval (135M) | UNKNOWN (1B Table 8 used 100 steps) | UNKNOWN / 1B ref |
| Early-stopping metric/threshold | "as long as val loss on D_post improves"; exact min-delta = UNKNOWN | EXPLICIT (metric) / UNKNOWN (threshold) |
| Seed (135M) | UNKNOWN (1B Stage-2 seed=40) | UNKNOWN / 1B ref |
| **Fixed config used FOR FIG 3a** | one specific (LR, dropout, batch) triple from the above ranges — NOT specified | **UNKNOWN — needs official code** |

LOCAL_REPRODUCTION_CHOICE (until code obtained): a reasonable fixed Fig-3a Stage-2 config is
LR=5e-4, dropout=0.0, global batch=480, warmup=500, WD=0.1, min LR=5e-5, max 2B tokens with early
stopping on MusicPile val loss (patience 3, eval every 100 steps). Flag as a choice, not the paper's.

---

## 6. Stage 3 — Downstream fine-tuning on ChemPile (measure L_ret, L_ft, L_pre)

- Section 3.3 (EXPLICIT): "We then fine-tune each post-trained checkpoint theta_post on D_ft for a
  fixed token budget of 200M tokens with various learning rates."
- Section 4.1 (EXPLICIT, Fig 3a specific): "we then fine-tune on D_ft ⊂ ChemPile, and report both
  the immediate post-training loss L_im and the retained post-training loss L_ret at a fixed Stage 3
  learning rate of 5×10^-5 (Figure 3a)." → **Fig 3a Stage-3 LR = 5e-5, single fixed value** (the LR
  sweep is for the frontier plots in Fig 2 / Section 4.3, not Fig 3a).
| Field | Value | Provenance |
|---|---|---|
| Token budget | 200M tokens (fixed) | EXPLICIT_PAPER (Section 3.3) |
| Learning rate | 5e-5 (fixed for Fig 3a) | EXPLICIT_PAPER (Section 4.1) |
| Fine-tuning method | full FFT, standard supervised fine-tuning | EXPLICIT_PAPER (Section 1: "applying standard supervised fine-tuning") |
| LR schedule / warmup / global batch / weight decay / dropout / seed | NOT separately specified for Stage 3 | UNKNOWN |
| Optimizer / betas / clip / precision | AdamW, 0.9/0.95, clip 1.0, bf16-mixed (Table 5 is "across all experiments") | EXPLICIT_PAPER (Table 5) |

LOCAL_REPRODUCTION_CHOICE: hold all Stage-3 knobs fixed across lambda (same schedule/batch/seed);
cosine decay with short warmup, global batch matching Stage 2. Everything except lambda held fixed —
that is the experimental control the paper relies on.

### Measured losses (Section 3.1, EXPLICIT)
- L_im := L(theta_post; D_post) — MusicPile val loss, measured after Stage 2.
- L_ft := L(theta_ft; D_ft) — ChemPile val loss, after Stage 3.
- L_ret := L(theta_ft; D_post) — MusicPile val loss, after Stage 3 (retention).
- L_pre := L(theta_ft; D_pre) — C4 val loss, after Stage 3.
Figure 3a plots only L_im (Before FT / Immediate) and L_ret (After FT / Retained) vs lambda.

---

## 7. Evaluation methodology

- Section 3 (EXPLICIT): "All losses are computed on held-out splits of the corresponding datasets."
- Section 3.1 (EXPLICIT): "We use validation loss as our evaluation metric." Rationale: loss is a
  reliable scale-invariant proxy for capability (Du et al. 2024; Gadre et al. 2024; Chen et al. 2025).
- Token-weighting / padding-exclusion / context length at eval: NOT specified = UNKNOWN.
  LOCAL choice: standard mean next-token CE over non-pad target tokens at seq len 1024.
- Validation sets: held-out splits of MusicPile (L_im, L_ret), ChemPile (L_ft), C4 (L_pre). Their
  construction/size = UNKNOWN.

## 8. Seeds for Figure 3a

- No seed count, error bars, or shaded variance appear for Figure 3a. The panels show a single line
  per dataset size. → **Appears to be a single seed / single run per (lambda, size).** = EXPLICIT
  (no multi-seed shown) but exact seed value for 135M = UNKNOWN (1B tables use 42 / 40).
- A star (★) marker appears on the Fig 3a lines; its meaning is not defined in the caption
  (possibly marking the lambda that corresponds to the compute-matched point). = UNKNOWN.

## 9. Exact claim of Figure 3a (Figure 3 caption + Section 4.1, EXPLICIT)

- Figure 3 caption (Left): "As the mixture fraction lambda increases, immediate MusicPile loss after
  post-training remains nearly constant, while retained MusicPile loss after downstream fine-tuning
  on ChemPile improves. This shows that the benefits of mixing can be *latent* ... but emerge after
  subsequent fine-tuning."
- Section 4.1 Result (EXPLICIT): "L_im remains nearly flat as the mixing fraction increases. In other
  words, once post-training is allowed to run to convergence, mixed and unmixed models reach similar
  performance on D_post. However ... As lambda increases, the retained post-training loss L_ret
  consistently decreases, indicating that models with more exposure to D_post during pretraining
  forget less after subsequent downstream adaptation."
- **Reproduction success criterion (300M line):** L_im vs lambda is approximately flat/constant;
  L_ret vs lambda is monotonically (or near-monotonically) decreasing from lambda=0 to lambda=1.
  Directionality matters more than absolute loss values.

---

## 10. Consolidated list of UNKNOWNs requiring official code / authors

1. Full 135M architecture (layers/hidden/heads/KV/MLP/init/tying) — no paper table; using SmolLM2-135M.
2. 135M Stage-1 pretraining config: peak/min LR, warmup, global batch, weight decay, seed.
3. Whether 135M used GQA (official) or an MHA "interpolated" variant like the 1B model.
4. Exact lambda interleaving schedule in Stage 1 (uniform-throughout vs front-loaded; shuffling seed).
5. The single FIXED Stage-2 config used for Fig 3a (specific LR/dropout/batch from Table 7 ranges).
6. 135M Stage-2 early-stopping patience, eval interval, and improvement threshold.
7. Stage-3 schedule/warmup/global batch/weight decay/seed (only LR=5e-5 and 200M budget are given).
8. Exact HF dataset ids/revisions for C4, MusicPile, ChemPile; val-split construction.
9. Nested-subset construction for 30M/150M (moot for 300M-only reproduction).
10. Validation-loss computation details (token weighting, padding, eval context length).
11. Meaning of the ★ marker in Figure 3a.
