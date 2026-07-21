"""Append-only checkpoint & artifact inventories.

Inventories are the canonical record of every checkpoint written. They are
strictly append-only: corrections and prunings are represented by *superseding*
records, never by editing or deleting a prior line. See CLAUDE.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from j_pretrain.artifacts.checkpoint import CheckpointMeta

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_DIR = Path(os.environ.get("J_PRETRAIN_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts")))
CHECKPOINT_INVENTORY = "checkpoint_inventory.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def record_checkpoint(
    meta: CheckpointMeta,
    rel_path: str,
    byte_size: int,
    created_at_utc: str,
    inventory_path: str | Path | None = None,
    backup_status: str = "unreplicated_local_copy",
) -> None:
    """Append one checkpoint-creation record (op=create)."""
    path = Path(inventory_path) if inventory_path else INVENTORY_DIR / CHECKPOINT_INVENTORY
    rec = {
        "op": "create",
        "run_id": meta.run_id,
        "checkpoint_id": meta.checkpoint_id,
        "stage": meta.stage,
        "checkpoint_class": meta.checkpoint_class,
        "milestone_labels": meta.milestone_labels,
        "lambda": meta.lambda_frac,
        "subset_tokens": meta.subset_tokens,
        "step": meta.step,
        "total_tokens": meta.total_tokens,
        "c4_tokens": meta.c4_tokens,
        "musicpile_tokens": meta.musicpile_tokens,
        "chempile_tokens": meta.chempile_tokens,
        "rel_path": rel_path,
        "byte_size": byte_size,
        "checksums": meta.checksums,
        "parent_checkpoint": meta.parent_checkpoint_id,
        "config_hash": meta.config_hash,
        "dataset_manifest_hash": meta.dataset_manifest_hash,
        "environment_hash": meta.environment_hash,
        "git_commit": meta.git_commit,
        "metrics_at_creation": meta.val_metrics,
        "creation_status": "complete",
        "load_validation_status": meta.load_validation_status,
        "backup_status": backup_status,
        "created_at_utc": created_at_utc,
    }
    _append_jsonl(path, rec)


def record_prune(
    run_id: str,
    checkpoint_id: str,
    reason: str,
    superseded_by: str,
    freed_bytes: int,
    at_utc: str,
    inventory_path: str | Path | None = None,
) -> None:
    """Append a superseding prune record for a deleted intermediate resumable."""
    path = Path(inventory_path) if inventory_path else INVENTORY_DIR / CHECKPOINT_INVENTORY
    _append_jsonl(path, {
        "op": "prune",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "reason": reason,
        "superseded_by": superseded_by,
        "freed_bytes": freed_bytes,
        "at_utc": at_utc,
    })


def read_inventory(inventory_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(inventory_path) if inventory_path else INVENTORY_DIR / CHECKPOINT_INVENTORY
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def live_checkpoints(inventory_path: str | Path | None = None) -> set[str]:
    """checkpoint_ids created and not later pruned."""
    created, pruned = set(), set()
    for r in read_inventory(inventory_path):
        if r.get("op") == "create":
            created.add(r["checkpoint_id"])
        elif r.get("op") == "prune":
            pruned.add(r["checkpoint_id"])
    return created - pruned
