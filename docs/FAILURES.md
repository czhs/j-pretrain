# FAILURES.md — running log of failures & non-obvious issues

Append-only chronological record. Each entry: date, symptom, root cause, resolution, whether any
run/artifact was invalidated. Routine status lives in the ledger, not here.

## 2026-07-21 — `conda run -n jpre` swallows heredoc/stdout
- **Symptom:** Python invoked via `conda run -n jpre python - <<'PY'` produced no stdout for data
  probes, making it look like data access failed.
- **Root cause:** `conda run` buffers/redirects child stdout in a way that drops heredoc output here.
- **Resolution:** Always invoke the interpreter directly:
  `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python`. Recorded in NEXT_ACTION as a critical ops note.
- **Impact:** none (tooling only; no run/artifact affected).

## 2026-07-21 — `hf_hub_download` hung once during preflight
- **Symptom:** a single `hf_hub_download` call hung; `curl` to the same host worked.
- **Root cause:** not fully diagnosed; likely `hf_transfer`/connection stall.
- **Resolution:** set `HF_HUB_ENABLE_HF_TRANSFER=0` for all HF/data ops; streaming + metadata access
  via `datasets`/`HfApi` works reliably. Dataset revisions were pinned successfully afterward.
- **Impact:** none (transient; datasets reachable, revisions pinned).

## 2026-07-21 — safetensors stores tied embedding twice (+56 MB/analysis snapshot)
- **Symptom:** analysis snapshot measured 310.6 MB vs naive 269 MB expectation.
- **Root cause:** `safetensors` materialises both `embed_tokens.weight` and `lm_head.weight` for the
  tied 135M model (they share storage in-memory but are written as two tensors).
- **Resolution:** accepted — load-independence and "complete unquantized weights" outweigh the 56 MB;
  folded into the storage projection (docs/STORAGE_PLAN.md). No action needed.
- **Impact:** storage projection uses the true 310.6 MB figure; gate still PASS.

<!-- New failures appended below this line. -->
