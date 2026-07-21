"""Atomic checkpoint creation with two classes: analysis snapshots + full resumables.

Every checkpoint is written to a temporary sibling directory, fsync'd, checksummed,
load-tested, and only then atomically renamed into its immutable final path. Final
paths are never overwritten. See CLAUDE.md "Artifact retention (CRITICAL)".

* **Analysis snapshot** — complete unquantized weights (bf16 safetensors) + model
  config + tokenizer ref + full metadata. Loads independently; no dependence on a
  mutable ``latest`` pointer.
* **Full resumable** — analysis contents PLUS an opaque ``training_state`` blob
  (optimizer / scheduler / scaler / RNG / dataloader cursor), torch-saved. The
  capture/restore of that blob lives in the training module; this layer just writes
  and verifies it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save

CLASS_ANALYSIS = "analysis"
CLASS_RESUMABLE = "resumable"

WEIGHTS_NAME = "model.safetensors"
TRAIN_STATE_NAME = "training_state.pt"
META_NAME = "metadata.json"
CHECKSUMS_NAME = "checksums.sha256"
COMPLETE_MARK = ".complete"


@dataclass
class CheckpointMeta:
    run_id: str
    checkpoint_id: str
    stage: str                    # stage1 | stage2 | stage3
    checkpoint_class: str         # analysis | resumable
    milestone_labels: list[str]
    lambda_frac: float
    subset_tokens: int
    step: int
    total_tokens: int
    c4_tokens: int
    musicpile_tokens: int
    chempile_tokens: int
    lr: Optional[float]
    train_loss: Optional[float]
    val_metrics: dict[str, float]
    parent_checkpoint_id: Optional[str]
    config_hash: str
    dataset_manifest_hash: str
    environment_hash: str
    git_commit: str
    seed: int
    created_at_utc: str
    tokenizer_ref: dict[str, Any]
    checksums: dict[str, str] = field(default_factory=dict)
    load_validation_status: str = "unverified"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _bf16_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Contiguous CPU bf16 tensors; ties are materialised so safetensors is happy."""
    sd = model.state_dict()
    return {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in sd.items()}


def write_checkpoint(
    out_root: str | Path,
    meta: CheckpointMeta,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    training_state: Optional[dict[str, Any]] = None,
) -> Path:
    """Atomically write a checkpoint and return its final immutable directory.

    ``training_state`` present -> resumable class; absent -> analysis snapshot.
    Raises ``FileExistsError`` if the final path already exists (never overwrite).
    """
    out_root = Path(out_root)
    final = out_root / meta.run_id / meta.stage / meta.checkpoint_class / meta.checkpoint_id
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint {final}")
    tmp = final.parent / (meta.checkpoint_id + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover from an aborted write; safe (not a final path)
    tmp.mkdir(parents=True)

    # 1) weights
    st_save(_bf16_state_dict(model), str(tmp / WEIGHTS_NAME))
    # 2) model config + tokenizer ref
    (tmp / "config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True))
    # 3) resumable training state
    if training_state is not None:
        with open(tmp / TRAIN_STATE_NAME, "wb") as f:
            torch.save(training_state, f)
            f.flush()
            os.fsync(f.fileno())

    # 4) checksums over every payload file
    checksums: dict[str, str] = {}
    for p in sorted(tmp.iterdir()):
        if p.is_file():
            checksums[p.name] = _sha256_file(p)
    meta.checksums = checksums

    # 5) load-test BEFORE finalizing
    meta.load_validation_status = "verified" if _load_test(tmp, model_config) else "FAILED"
    if meta.load_validation_status != "verified":
        shutil.rmtree(tmp)
        raise RuntimeError(f"checkpoint load-test failed for {meta.checkpoint_id}")

    # 6) metadata + completion marker, fsync files
    (tmp / META_NAME).write_text(meta.to_json())
    (tmp / CHECKSUMS_NAME).write_text(
        "\n".join(f"{v}  {k}" for k, v in sorted(checksums.items())) + "\n")
    (tmp / COMPLETE_MARK).write_text("ok\n")
    for p in tmp.iterdir():
        if p.is_file():
            with open(p, "rb") as f:
                os.fsync(f.fileno())

    # 7) atomic rename into final immutable path
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)
    _fsync_dir(final.parent)
    return final


def _load_test(ckpt_dir: Path, model_config: dict[str, Any]) -> bool:
    """Reload weights from disk and sanity-check a tensor loads finite."""
    try:
        sd = st_load(str(ckpt_dir / WEIGHTS_NAME))
        if not sd:
            return False
        any_t = next(iter(sd.values()))
        if not torch.isfinite(any_t.float()).all():
            return False
        if (ckpt_dir / TRAIN_STATE_NAME).exists():
            with open(ckpt_dir / TRAIN_STATE_NAME, "rb") as f:
                torch.load(f, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def is_complete(ckpt_dir: str | Path) -> bool:
    """True only for a fully-finalized checkpoint (ignores ``*.tmp`` writes)."""
    d = Path(ckpt_dir)
    return (d / COMPLETE_MARK).exists() and (d / META_NAME).exists() and (d / WEIGHTS_NAME).exists()


def verify_checksums(ckpt_dir: str | Path) -> bool:
    d = Path(ckpt_dir)
    meta = json.loads((d / META_NAME).read_text())
    for name, expect in meta["checksums"].items():
        if not (d / name).exists() or _sha256_file(d / name) != expect:
            return False
    return True


def load_weights(ckpt_dir: str | Path) -> dict[str, torch.Tensor]:
    return st_load(str(Path(ckpt_dir) / WEIGHTS_NAME))


def load_training_state(ckpt_dir: str | Path) -> dict[str, Any]:
    with open(Path(ckpt_dir) / TRAIN_STATE_NAME, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def read_meta(ckpt_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(ckpt_dir) / META_NAME).read_text())
