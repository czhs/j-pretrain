# DECISIONS

Concise scientific/engineering decisions. Format: date | decision | evidence | alternatives | reason | frozen?

## 2026-07-21
- **Reduced scope authorized**: Only the 300M MusicPile subset of Fig 3a (not 30M/150M/300M sweep). | Evidence: mission doc explicitly authorizes. | Alt: full 3-size sweep. | Reason: compute budget on single 4090; user pre-authorized. | FROZEN (record in SCOPE_LOCK.json).
- **Datasets**: C4=allenai/c4, MusicPile=m-a-p/MusicPile, ChemPile=jablonkagroup/chempile-education (education split reachable). | Evidence: HF dataset_info OK for all three. | Reason: match paper D_pre/D_post/D_ft. | Pending: confirm exact ChemPile subset from paper/official code before freezing.
- **Env**: Build dedicated conda env; base env lacks ML stack. | Reason: reproducibility + isolation. | Not yet frozen (env hash pending).
