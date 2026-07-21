# SPEC_AUDIT.md — Independent audit of docs/PAPER_SPEC.md

Auditor: independent verification agent. Method: read paper.pdf directly
(`pdftotext -layout` + visual read of the Figure 3 page), did NOT trust the existing spec.
Paper: "Early Data Exposure Improves Robustness to Subsequent Fine-Tuning."
Line numbers below refer to the extracted text `/tmp/paper.txt`; page/table/section citations are
the authoritative reference.

## Verdict table

| # | Claim under audit | Verdict | Evidence |
|---|---|---|---|
| 1 | No 135M architecture table, no 135M Stage-1 HP table; only 1B tables exist | CONFIRMED | Table 2 title = "SmolLM2-**1B** model architecture" (A.1, p.15). Table 6 title = "Stage 1 pretraining configuration for **1B** experiments" (A.4, p.16). No 135M counterpart anywhere. Only 135M-specific HP table is Table 7 (Stage 2). |
| 2 | Stage-1 C4 budget = 8.7B (Table 3); "~10B" (Sec 3.3) is loose rounding | CONFIRMED | Table 3 "135M Parameter Experiments": C4 Train = 8.7B (p.15). Sec 3.3: "we pretrain on approximately 10B tokens from Dpre" (p.5). Spec treats 8.7B as authoritative via a labeled LOCAL_REPRODUCTION_CHOICE — sound. |
| 3 | Fig 3a is ADD (total Dpost grows with λ), NOT compute-matched; Sec 4.2 distinguishes 3a from 3b | CONFIRMED | Sec 4.2: the 4.1/Fig-3a experiments "do not isolate whether the benefit comes from *when* Dpost is introduced or simply from the model seeing more total Dpost tokens" (p.6). Sec 4.2 Setup: "We fix the total number of Dpost tokens seen across Stage 1 ... and vary only how that budget is allocated ... reserve the remaining (1−λ) fraction for post-training, so every model sees exactly one pass over Dpost in total" (Fig 3b). Spec's verbatim quote matches exactly. |
| 4 | λ = λ·|Dpost| MusicPile tokens mixed into Stage 1, at most one pass (Sec 3.1) | CONFIRMED | Sec 3.1 Stage 1: "additionally mixes a fraction λ ∈ [0,1] of the post-training corpus Dpost into this stage. Here, λ=0 denotes no exposure ... λ is at most one, denoting at most one pass over Dpost during pretraining." Spec quote verbatim. |
| 5 | Stage 2: full Dpost, AdamW warmup+cosine, max 2B tokens, early stop on Dpost val loss; Fig 3a uses ONE fixed config (not the Table 7 sweep) | CONFIRMED | Sec 3.3: "Stage 2 post-training on Dpost using AdamW with linear warmup and cosine decay ... early stopping and continue training as long as validation loss on Dpost improves (up to a maximum budget of 2B tokens)." Sec 4.1 Setup: "post-train on Dpost until convergence using a fixed hyperparameter configuration"; "these experiments fix the Stage 2 post-training procedure." |
| 6 | Table 7 (135M Stage-2) sweep values | CONFIRMED (all 7 numbers) | Table 7 (A.5, p.17): LR {1e-4,2e-4,5e-4,1e-3,5e-3}; Min LR 5e-5; Dropout {0.0,0.02,0.05}; Weight decay 0.1 (fixed); Warmup 500 (fixed); Batch {192,480,896}; Max tokens 2B (early stopping). Every value matches the spec exactly. |
| 7 | Stage 3: 200M tokens; LR 5e-5 fixed for Fig 3a | CONFIRMED | Sec 3.3: "fine-tune each post-trained checkpoint θpost on Dft for a fixed token budget of 200M tokens." Sec 4.1: "report both ... Lim and ... Lret at a fixed Stage 3 learning rate of 5×10^-5 (Figure 3a)." |
| 8 | Fig 3a: L_im ~flat; L_ret decreases as λ increases | CONFIRMED | Fig 3 caption (Left): "immediate MusicPile loss ... remains nearly constant, while retained MusicPile loss ... improves." Sec 4.1 Result: "Lim remains nearly flat"; "As λ increases, the retained post-training loss Lret consistently decreases." Visual check of the Fig 3a panels agrees (flat Immediate; downward-sloping Retained, incl. the thick 300M line). |
| 9 | Fig 3a appears single-seed (no error bars / no variance) | CONFIRMED | Visual inspection of Figure 3a: solid single line per dataset size, no error bars, no shaded band, no ± annotation. Caption/text report no seed count or variance. (Spec appropriately hedges as "Appears to be" and marks the exact 135M seed value UNKNOWN.) |
| 10 | Datasets C4 / MusicPile / ChemPile; Dpost sizes {30M,150M,300M} | CONFIRMED | Table 1 (p.4): Music→Chemistry = C4 / MusicPile / ChemPile. Sec 4.1 Setup: "|Dpost| ∈ {30M, 150M, 300M} where Dpost ⊂ MusicPile." Table 3 lists all three corpora (135M). |
| 11 | Any EXPLICIT_PAPER tag that is actually NOT in the paper (most dangerous error) | NONE FOUND | Every EXPLICIT_PAPER tag traces to real paper text/table. Architecture rows citing Table 2 (vocab 49,152; 1,024/8,192 context; RMSNorm; RoPE base 100,000) are 1B-table values, but the spec explicitly co-tags each as INFERRED_FROM_SMOLLM2 for 135M and repeats the warning in the CRITICAL STRUCTURAL FACT box. No value is misrepresented as 135M-reported. |

## 135M Stage-1 LR / batch / warmup — truly absent?

CONFIRMED ABSENT. Searched the full text and all appendix tables. The only place these appear is
Table 6, which is titled "Stage 1 pretraining configuration for **1B** experiments" (LR 5e-4,
warmup 1,000, global batch 512, seed 42, 20B total tokens / 21.0B C4 corpus). Table 5 (all
experiments) gives only optimizer/betas/clip/precision — no LR, batch, or warmup. No caption,
footnote, or combined table states a 135M Stage-1 LR, batch size, or warmup. The spec's UNKNOWN
labels for these are correct.

## Minor / non-blocking observations (NOT spec errors)

- Internal paper inconsistency (1B only, irrelevant to Fig 3a): Table 4 lists 1B C4 Train = 19.7B,
  while Table 6 lists "C4 corpus 21.0B tokens" / "Total tokens 20B." The spec cites only the 19.7B
  (Table 4) figure and never uses it for Fig 3a, so this does not affect the reproduction.
- The ★ marker: visually present on both Fig 3a and Fig 3b Retained panels (e.g. ~λ=0.5 on 150M,
  ~λ=1.0 on 300M in 3a) and undefined in the caption. Spec marks its meaning UNKNOWN — correct.
- "standard supervised fine-tuning" (spec §6, cited as Section 1) is indeed in Section 1 / intro
  (p.2). Attribution correct.

## Overall verdict

The spec is SOUND. All 11 audited claims: CONFIRMED. Zero WRONG, zero OVERSTATED, zero mislabeled
EXPLICIT_PAPER tags. Provenance discipline is strong: 1B-table values are never passed off as
135M-reported, and every UNKNOWN/INFERRED boundary is drawn where the paper actually stops. The
Fig 3a (ADD) vs Fig 3b (compute-matched) distinction — the highest-risk conceptual point — is
correctly grounded in the exact Sec 4.1/4.2 wording. Table 7 numbers are exact. No corrections
required before freezing.
