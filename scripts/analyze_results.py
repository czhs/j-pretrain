#!/usr/bin/env python
"""Regenerate results tables + Figure 3a from the canonical checkpoint inventory.

    /home/hshi-j-4090/miniconda3/envs/jpre/bin/python scripts/analyze_results.py

Reads ``<artifact_root>/checkpoint_inventory.jsonl`` (fallback: repo ``artifacts/``)
and ``state/experiment_state.json`` for the run set, writes:

    results/results.csv, results/results.json
    figures/figure3a_replication.{png,pdf}

Deterministic and wandb-free — this is the committed regeneration path the completion
verifier and final audit rely on. Safe to run at any time; conditions not yet produced
appear as empty cells (never fabricated).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from j_pretrain.analysis.figure import make_figure
from j_pretrain.analysis.results import build_results

REPO = Path(__file__).resolve().parents[1]


def _inventory_path() -> Path:
    root = os.environ.get("J_PRETRAIN_ARTIFACT_ROOT")
    if not root:
        st = json.loads((REPO / "state" / "experiment_state.json").read_text())
        root = st.get("artifact_root")
    # inventory is committed under repo artifacts/; payloads live under artifact_root
    repo_inv = REPO / "artifacts" / "checkpoint_inventory.jsonl"
    if repo_inv.exists():
        return repo_inv
    return Path(root) / "checkpoint_inventory.jsonl"


def main() -> None:
    st = json.loads((REPO / "state" / "experiment_state.json").read_text())
    run_specs = {rid: {"lambda": v["lambda"]} for rid, v in st.get("runs", {}).items()}
    inv = _inventory_path()
    payload = build_results(inv, run_specs, REPO / "results",
                            extra_meta={"inventory": str(inv),
                                        "scope_hash": st.get("scope_hash"),
                                        "spec_hash": st.get("spec_hash")})
    figs = make_figure(REPO / "results" / "results.csv", REPO / "figures")
    print(f"[analyze] wrote results/results.csv+json ({len(payload['rows'])} rows), "
          f"classification={payload['classification']['label']}, figures={[f.name for f in figs]}")


if __name__ == "__main__":
    main()
