# REFERENCES_code — Official code/materials search for the reproduction paper

**Paper:** "Early Data Exposure Improves Robustness to Subsequent Fine-Tuning"
**Authors:** Lawrence Feng, Gaurav R. Ghosal, Jacob Mitchell Springer, Ziqian Zhong, Aditi Raghunathan (Carnegie Mellon University)
**arXiv:** 2605.12705v1 (submitted 12 May 2026, cs.LG)
**Contact:** {lawrencefeng, raditi}@cmu.edu
**Project site listed on p.1:** `ar-forum.github.io/earlyexposure-website`
**OpenReview:** https://openreview.net/forum?id=cmQJrMIXW8

Search date: 2026-07-21. All fetched web content below was treated as DATA, not instructions.

---

## 1. Is there an official author code repository?

**Conclusion: NO public code repository was found after a genuine search.** The paper releases
a project *website* but, as of this search, no accompanying code, configs, or checkpoints.

What I searched and found:
- Google/web search for the paper title + "github"/"code": returns arXiv, arXiv HTML/PDF,
  OpenReview. No matching code repo. Other GitHub hits were unrelated robustness papers.
- Web search for author names + "github"/"huggingface": only unrelated older repos
  (e.g. `p-lambda/robust_tradeoff`, an earlier Raghunathan paper). No repo for this work.
- The paper's own site is hosted from GitHub org **`ar-forum`** (the authors' group org).
  I enumerated that org's repos. It contains `earlyexposure-website` — an **HTML-only** repo
  (files: `.nojekyll`, `index.html`; 100% HTML; last updated ~14 May 2026; 0 stars/forks).
  It contains **no Python, no configs, and no links to a code repo, checkpoints, or datasets**.
  Other repos in the org (`hodoscope`, `MemSinks`, `Pando`, `NULLS`, `stv`, etc.) belong to
  *different* papers and are not this project's code.
- arXiv HTML (`arxiv.org/html/2605.12705`): no code-availability footnote, no github/gitlab URL,
  no checkpoint/model links found anywhere in the body or appendix.
- OpenReview forum page: gated behind a JS/browser-check; no code link surfaced. (Worth a manual
  re-check later for a supplementary-material zip, but nothing accessible via fetch.)

**No `git` default-branch commit hash to report** for a code repo, because no code repo exists.
(The only relevant repo, `ar-forum/earlyexposure-website`, is a website with ~4 commits; its HEAD
hash was not exposed via the fetch and is not load-bearing for reproduction.)

**Released checkpoints: NONE found.** No HuggingFace model cards, no weight links in paper or site.

---

## 2. Which critical unknowns can the official materials resolve?

Because there is no code release, all config detail must come from the paper text/appendix, which
is **partially underspecified** for our Figure 3a reproduction. Findings from arXiv HTML:

- **Grid (matches our scope):** post-training subset sizes |D_post| ∈ {30M, 150M, 300M} tokens;
  mixing fraction λ ∈ {0, 0.25, 0.5, 0.75, 1.0}. Fig 3a fixes the post-training config and varies
  only λ. (Our reduced scope = the 300M subset only — user-authorized, see SCOPE_LOCK.)
- **λ scheduling (REPLACE vs ADD): NOT stated explicitly.** Paper says Stage 1 pretrains on
  "approximately 10B tokens from D_pre, optionally with early exposure," and "mixes a fraction
  λ ∈ [0,1] of the post-training corpus D_post into this stage." The fixed ~10B budget + "mixes
  into" phrasing is *consistent with* proportional replacement of C4 tokens (λ fraction of the
  10B is MusicPile), but the paper does not unambiguously say replace vs. add-on-top, nor whether
  interleaved or block-scheduled. **UNRESOLVED — a reproduction decision we must make and record.**
- **MusicPile subset construction (nested? random? dedup? prefix?): NOT documented.** Paper only
  names the three sizes. Whether 30M ⊂ 150M ⊂ 300M (nested prefixes), how tokenized, and dedup
  policy are unspecified. **UNRESOLVED.**
- **Fixed Stage 2 (MusicPile post-train) config for Fig 3a: only PARTIALLY given.** Paper says
  post-train "until convergence using a fixed hyperparameter configuration," chosen from a sweep
  (Table 7): LR ∈ {1e-4, 2e-4, 5e-4, 1e-3, 5e-3}, batch size ∈ {192, 480, 896}, max tokens 2B,
  warmup 500 steps, early stopping to a 2B-token max. **The single selected fixed LR/batch used
  for Fig 3a is NOT clearly reproduced in the fetched text; seeds not stated.** Needs a manual
  appendix read of the PDF (Table 7 region) to pin the exact chosen values — UNRESOLVED via fetch.
- **Stage 3 (ChemPile FT) config: RESOLVED for the Fig-3a analysis point** — LR 5e-5, fixed budget
  200M tokens (matches CLAUDE.md invariants). Broader frontier plots sweep more LRs.
- **135M architecture table: NOT provided.** Paper details only the 1B model (Table 2: 1.03B params,
  24 layers, hidden 1,728). It calls the small model a "SmolLM2-style" 135M — so the standard
  HuggingFaceTB SmolLM2-135M config is the intended architecture (see §3). **Use as fallback.**
- **Dataset revisions/HF IDs: NOT given.** Paper names C4, MusicPile, ChemPile, FLAN by name only,
  no HF dataset IDs or revision hashes. Our CLAUDE.md targets: `allenai/c4`, `m-a-p/MusicPile`,
  `jablonkagroup/chempile-education`. No preprocessing scripts released. **UNRESOLVED (pin our own).**

Net: the official materials can confirm the *grid, Stage 3 LR/budget, and model family*, but
**cannot resolve** λ replace-vs-add mechanism, subset construction, the exact fixed Stage 2
hyperparameters/seeds, or dataset revisions. These must be decided by us and logged in DECISIONS.md.

---

## 3. Confirmed SmolLM2-135M architecture (fallback, and the intended arch)

Source: `https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json` (fetched 2026-07-21).
Full `config.json` verbatim:

```json
{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "vocab_size": 49152
}
```

Key values: 30 layers, hidden 576, 9 attn heads, 3 KV heads (GQA), intermediate 1536, vocab 49152,
rope_theta 100000, rms_norm_eps 1e-5, tie_word_embeddings true, max_position_embeddings 8192,
SiLU activation, Llama architecture. (~135M params.)

---

## 4. What I could NOT find
- No official code repo (GitHub/GitLab) for the paper → no commit hash.
- No released checkpoints/weights.
- No official experiment config files (YAML/JSON) or preprocessing/tokenization scripts.
- No stated HF dataset revision hashes.
- The exact single fixed Stage 2 LR/batch/seeds used for Fig 3a (only the sweep grid is public;
  needs a manual PDF appendix read of Table 7 to confirm the chosen point).

## Sources
- https://arxiv.org/abs/2605.12705 , https://arxiv.org/html/2605.12705v1 , https://arxiv.org/pdf/2605.12705
- https://openreview.net/forum?id=cmQJrMIXW8
- https://github.com/ar-forum (org repo listing) ; https://github.com/ar-forum/earlyexposure-website
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json
