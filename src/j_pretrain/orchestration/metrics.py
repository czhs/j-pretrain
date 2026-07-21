"""Metric logging: local JSONL (canonical) + optional Weights & Biases (supplementary).

The **local** append-only JSONL under the run's artifact dir is the canonical record
that the completion verifier, analyses, figures and audits read. wandb is
supplementary observability only: any wandb error falls back to offline / disabled
and NEVER interrupts, delays, or fails a training run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

WANDB_ENTITY = "ametind-o"
WANDB_PROJECT = "j-pretrain"


class MetricLogger:
    def __init__(self, run_id: str, stage: str, metrics_path: str | Path,
                 wandb_run=None):
        self.run_id = run_id
        self.stage = stage
        self.path = Path(metrics_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wandb = wandb_run

    def log(self, record: dict[str, Any]) -> None:
        rec = {"run_id": self.run_id, "stage": self.stage, **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self._wandb is not None:
            try:
                step = record.get("opt_step")
                self._wandb.log(record, step=step)
            except Exception:
                pass  # wandb failure must never affect training

    def close(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


def init_wandb(run_id: str, stage: str, lambda_frac: float, subset_tokens: int,
               config: dict[str, Any], enabled: bool = True):
    """Best-effort wandb run init; returns a run handle or None. Never raises."""
    if not enabled:
        return None
    try:
        import wandb
        return wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT,
            name=f"{run_id}::{stage}", id=f"{run_id}__{stage}".replace(".", "_"),
            group=run_id, job_type=stage,
            tags=[stage, f"lambda-{lambda_frac}", f"subset-{subset_tokens // 1_000_000}m"],
            config={**config, "run_id": run_id, "stage": stage,
                    "lambda": lambda_frac, "subset_tokens": subset_tokens},
            resume="allow", reinit=True,
        )
    except Exception:
        # offline fallback, then give up silently (metrics still logged locally)
        try:
            os.environ.setdefault("WANDB_MODE", "offline")
            import wandb
            return wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                              name=f"{run_id}::{stage}", group=run_id, reinit=True,
                              config=config)
        except Exception:
            return None
