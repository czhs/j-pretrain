"""Single-GPU experiment orchestrator entrypoint (``python -m j_pretrain.orchestration.run``).

Drives the run DAG one node (``run_id`` x ``stage``) at a time on the 4090, holding
an exclusive :class:`~j_pretrain.orchestration.gpulock.GpuLock`. For each ready node it
builds the REAL packed datasets + plan, constructs the model from the correct parent
checkpoint (stage1<-shared init; stage2<-stage1 final; stage3<-stage2 restored-best),
resume-detects any valid in-stage resumable, runs the :class:`StageDriver`, then runs a
per-run completion audit before marking the node ``complete`` in canonical state.

Design invariants (mission-critical):
    * ONE GPU job at a time (GpuLock refuses a second live holder).
    * No cross-run checkpoint reuse: the lambda=0 init feeds every stage1, but each
      run's stage1 output feeds ONLY that run's stage2/3 (enforced by ``_parent_for``).
    * All stage1 runs start from byte-identical init weights (loaded from ONE init ckpt).
    * A healthy run is never restarted: resume loads the latest valid resumable.
    * Scientific config is frozen per stage; only ExecConfig (microbatch/grad-accum) is
      hardware-tuned and never enters the config hash.

The orchestrator loop is detached (tmux) and survives usage-limit / power gaps: every
node re-derives its state from files, so re-launching after a crash resumes cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from j_pretrain.artifacts import checkpoint as ck
from j_pretrain.artifacts import inventory as inv
from j_pretrain.config.hashing import config_hash
from j_pretrain.config.schemas import ExecConfig, ModelConfig, StageConfig
from j_pretrain.data.shards import PackedDataset
from j_pretrain.models.build import build_model
from j_pretrain.orchestration import dag
from j_pretrain.orchestration.gpulock import GpuLock, GpuLockError
from j_pretrain.orchestration.metrics import MetricLogger, init_wandb
from j_pretrain.orchestration.stage_driver import StageContext, StageDriver
from j_pretrain.training.dataplan import ShuffledSourcePlan, Stage1Plan
from j_pretrain.training.loader import PlanLoader
from j_pretrain.training.loop import Trainer
from j_pretrain.training.optim import build_adamw
from j_pretrain.training.rngstate import capture_rng

REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = REPO_ROOT / "state"
DEFAULT_ARTIFACT_ROOT = "/home/hshi-j-4090/Desktop/j-pretrain-artifacts"

INIT_RUN_ID = "shared-init"
INIT_STAGE = "init"
# loader source key -> on-disk dataset directory name
DATASET_DIR = {"c4": "c4", "mp": "musicpile", "chempile": "chempile"}


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_commit(repo: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _atomic_write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, p)


def _append_jsonl(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
# orchestrator configuration (all paths/knobs; testable with tiny values)
# --------------------------------------------------------------------------- #
@dataclass
class OrchestratorConfig:
    repo_root: Path
    artifact_root: Path            # large payloads (external ok)
    inventory_dir: Path            # small append-only records (committed repo/artifacts)
    datasets_root: Path            # <artifact_root>/datasets
    device: str
    exec_cfg: ExecConfig
    model_cfg_path: Path
    stage_cfg_paths: dict          # {"stage1": path, "stage2": path, "stage3": path}
    run_ids: list                  # run order (lambda=0 first)
    run_specs: dict                # run_id -> {"lambda","subset_tokens","seed"}
    c4_train_budget_tokens: int    # fixed C4 exposure (ADD policy)
    seed: int
    tokenizer_ref: dict
    dataset_manifest_hash: str
    environment_hash: str
    fused_adamw: bool
    wandb_enabled: bool = True
    seq_len: int = 1024
    experiment: str = "music"
    state_dir: Optional[Path] = None  # defaults to <repo_root>/state

    def __post_init__(self) -> None:
        if self.state_dir is None:
            self.state_dir = self.repo_root / "state"

    @property
    def gpu_lock_path(self) -> Path:
        return self.state_dir / "gpu.lock"

    @property
    def experiment_state_path(self) -> Path:
        return self.state_dir / "experiment_state.json"

    @property
    def ckpt_inventory(self) -> Path:
        return self.inventory_dir / inv.CHECKPOINT_INVENTORY


def default_config(artifact_root: Optional[str] = None, wandb_enabled: bool = True,
                   device: Optional[str] = None) -> OrchestratorConfig:
    """Build the production orchestrator config from committed repo files + state."""
    root = Path(artifact_root or os.environ.get("J_PRETRAIN_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    state = _load_json(STATE_DIR / "experiment_state.json") or {}
    scope = _load_json(STATE_DIR / "SCOPE_LOCK.json") or {}
    datasets_cfg = _load_json(REPO_ROOT / "configs" / "data" / "datasets.json") or {}
    run_ids = state.get("run_queue") or [r["run_id"] for r in scope.get("runs", [])]
    run_specs = {rid: {"lambda": v["lambda"], "subset_tokens": v["subset_tokens"],
                       "seed": v["seed"]}
                 for rid, v in state.get("runs", {}).items()}
    tok = (datasets_cfg.get("tokenizer") or {})
    tokenizer_ref = {"id": tok.get("id"), "revision": tok.get("revision"),
                     "dir_sha256": tok.get("dir_sha256")}
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return OrchestratorConfig(
        repo_root=REPO_ROOT, artifact_root=root,
        inventory_dir=REPO_ROOT / "artifacts", datasets_root=root / "datasets",
        device=dev,
        exec_cfg=ExecConfig(microbatch_size=8, torch_compile=False, dtype="bfloat16"),
        model_cfg_path=REPO_ROOT / "configs" / "model" / "smollm2-135m.json",
        stage_cfg_paths={
            "stage1": REPO_ROOT / "configs" / "stage1" / "music.json",
            "stage2": REPO_ROOT / "configs" / "stage2" / "music.json",
            "stage3": REPO_ROOT / "configs" / "stage3" / "chempile.json"},
        run_ids=list(run_ids), run_specs=run_specs,
        c4_train_budget_tokens=int(scope.get("scope", {}).get("stage1_c4_budget_tokens",
                                   (datasets_cfg.get("budgets", {}) or {}).get("stage1_c4_tokens", 8_700_000_000))),
        seed=int(scope.get("scope", {}).get("seed", 1234)),
        tokenizer_ref=tokenizer_ref,
        dataset_manifest_hash=config_hash(state.get("dataset_manifest_hashes", {})),
        environment_hash=config_hash(state.get("env", {})),
        fused_adamw=(dev.startswith("cuda")), wandb_enabled=wandb_enabled,
        seq_len=int(datasets_cfg.get("seq_len", 1024)),
    )


# --------------------------------------------------------------------------- #
# checkpoint lookup (parent resolution + resume detection) over the inventory
# --------------------------------------------------------------------------- #
def _find_ckpt(cfg: OrchestratorConfig, run_id: str, stage: str, label: str,
               cls: str) -> tuple[Optional[Path], Optional[str]]:
    """Latest live checkpoint dir + id matching (run, stage, class, milestone label)."""
    recs = inv.read_inventory(cfg.ckpt_inventory)
    live = inv.live_checkpoints(cfg.ckpt_inventory)
    best = None
    for r in recs:
        if (r.get("op") == "create" and r.get("run_id") == run_id and r.get("stage") == stage
                and r.get("checkpoint_class") == cls and label in r.get("milestone_labels", [])
                and r.get("checkpoint_id") in live):
            if best is None or r.get("step", 0) >= best.get("step", 0):
                best = r
    if best is None:
        return None, None
    d = cfg.artifact_root / best["rel_path"]
    return (d if d.exists() and ck.is_complete(d) else None), best["checkpoint_id"]


def _parent_for(cfg: OrchestratorConfig, run_id: str, stage: str) -> tuple[Optional[Path], Optional[str]]:
    """Weights source for a stage (cross-stage lineage — see module docstring)."""
    if stage == "stage1":
        return _find_ckpt(cfg, INIT_RUN_ID, INIT_STAGE, "init", ck.CLASS_ANALYSIS)
    if stage == "stage2":
        return _find_ckpt(cfg, run_id, "stage1", "final", ck.CLASS_ANALYSIS)
    if stage == "stage3":
        return _find_ckpt(cfg, run_id, "stage2", "restored_best", ck.CLASS_ANALYSIS)
    raise ValueError(stage)


def _latest_resumable(cfg: OrchestratorConfig, run_id: str, stage: str) -> Optional[Path]:
    """Latest valid in-stage resumable (for crash resume); excludes restored_best terminal."""
    recs = inv.read_inventory(cfg.ckpt_inventory)
    live = inv.live_checkpoints(cfg.ckpt_inventory)
    cands = []
    for r in recs:
        if (r.get("op") == "create" and r.get("run_id") == run_id and r.get("stage") == stage
                and r.get("checkpoint_class") == ck.CLASS_RESUMABLE
                and "restored_best" not in r.get("milestone_labels", [])
                and r.get("checkpoint_id") in live):
            d = cfg.artifact_root / r["rel_path"]
            if d.exists() and ck.is_complete(d):
                cands.append((r.get("step", 0), d))
    if not cands:
        return None
    return max(cands, key=lambda x: x[0])[1]


# --------------------------------------------------------------------------- #
# init checkpoint (shared, byte-identical across all stage1 runs)
# --------------------------------------------------------------------------- #
def ensure_init_checkpoint(cfg: OrchestratorConfig) -> str:
    """Create (once) the permanent shared init checkpoint; return its analysis id."""
    d, cid = _find_ckpt(cfg, INIT_RUN_ID, INIT_STAGE, "init", ck.CLASS_ANALYSIS)
    if d is not None:
        return cid
    mc = ModelConfig.from_json(cfg.model_cfg_path)
    model = build_model(mc, seed=cfg.seed, attn_implementation=cfg.exec_cfg.attn_implementation)
    model_cfg_dict = mc.scientific_dict()
    stage1_cfg = StageConfig.from_json(cfg.stage_cfg_paths["stage1"])
    git = _git_commit(cfg.repo_root)

    def _meta(cls: str) -> ck.CheckpointMeta:
        cid = f"init-{cls[:2]}-seed{cfg.seed}-{config_hash((cls, cfg.seed))[:8]}"
        return ck.CheckpointMeta(
            run_id=INIT_RUN_ID, checkpoint_id=cid, stage=INIT_STAGE, checkpoint_class=cls,
            milestone_labels=["init"], lambda_frac=-1.0, subset_tokens=0, step=0,
            total_tokens=0, c4_tokens=0, musicpile_tokens=0, chempile_tokens=0,
            lr=None, train_loss=None, val_metrics={},
            parent_checkpoint_id=None, config_hash=config_hash(model_cfg_dict),
            dataset_manifest_hash=cfg.dataset_manifest_hash, environment_hash=cfg.environment_hash,
            git_commit=git, seed=cfg.seed, created_at_utc=_now(), tokenizer_ref=cfg.tokenizer_ref)

    # analysis snapshot (weights only — this is what every stage1 loads)
    ma = _meta(ck.CLASS_ANALYSIS)
    fa = ck.write_checkpoint(cfg.artifact_root / "checkpoints", ma, model, model_cfg_dict)
    inv.record_checkpoint(ma, rel_path=str(fa.relative_to(cfg.artifact_root)),
                          byte_size=_dir_bytes(fa), created_at_utc=ma.created_at_utc,
                          inventory_path=cfg.ckpt_inventory)
    # full resumable (fresh optimizer + rng at step 0; permanent per schedule)
    opt = build_adamw(model, stage1_cfg, fused=False)
    tstate = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
              "optimizer": opt.state_dict(), "opt_step": 0, "windows_consumed": 0,
              "tokens": {"total": 0, "c4": 0, "mp": 0, "chempile": 0},
              "best_val": None, "no_improve_evals": 0, "total_steps": 0, "rng": capture_rng()}
    mr = _meta(ck.CLASS_RESUMABLE)
    fr = ck.write_checkpoint(cfg.artifact_root / "checkpoints", mr, model, model_cfg_dict,
                             training_state=tstate)
    inv.record_checkpoint(mr, rel_path=str(fr.relative_to(cfg.artifact_root)),
                          byte_size=_dir_bytes(fr), created_at_utc=mr.created_at_utc,
                          inventory_path=cfg.ckpt_inventory)
    return ma.checkpoint_id


# --------------------------------------------------------------------------- #
# per-node dataset + trainer construction
# --------------------------------------------------------------------------- #
def _packed(cfg: OrchestratorConfig, source_key: str, split: str) -> PackedDataset:
    return PackedDataset(cfg.datasets_root / DATASET_DIR[source_key] / split)


def _build_val_sets(cfg: OrchestratorConfig) -> dict:
    return {"c4": _packed(cfg, "c4", "val"), "mp": _packed(cfg, "mp", "val"),
            "chempile": _packed(cfg, "chempile", "val")}


def _build_train(cfg: OrchestratorConfig, stage: str, lambda_frac: float,
                 subset_tokens: int, seed: int, stage_cfg: StageConfig):
    """Return (loader, total_steps, milestone_max_tokens, primary_val)."""
    seq = cfg.seq_len
    if stage == "stage1":
        c4 = _packed(cfg, "c4", "train")
        mp = _packed(cfg, "mp", "train")
        n_c4 = cfg.c4_train_budget_tokens // seq
        if n_c4 > len(c4):
            raise ValueError(f"C4 pool has {len(c4)} windows < required {n_c4}")
        n_mp = round(lambda_frac * (subset_tokens // seq))
        if n_mp > len(mp):
            raise ValueError(f"MusicPile pool has {len(mp)} windows < required {n_mp}")
        plan = Stage1Plan(n_c4, n_mp)
        sources = {"c4": c4, "mp": mp}
        total_windows = n_c4 + n_mp
        total_steps = total_windows // stage_cfg.global_batch_seqs
        milestone_max = total_windows * seq
        primary_val = None
    elif stage == "stage2":
        mp = _packed(cfg, "mp", "train")
        n_sub = subset_tokens // seq
        if n_sub > len(mp):
            raise ValueError(f"MusicPile pool has {len(mp)} windows < subset {n_sub}")
        plan = ShuffledSourcePlan("mp", n_sub, seed=seed, shuffle=True)
        sources = {"mp": mp}
        total_steps = stage_cfg.max_optimizer_steps
        milestone_max = stage_cfg.max_tokens
        primary_val = "mp"
    elif stage == "stage3":
        ch = _packed(cfg, "chempile", "train")
        plan = ShuffledSourcePlan("chempile", len(ch), seed=seed, shuffle=True)
        sources = {"chempile": ch}
        total_steps = stage_cfg.max_optimizer_steps
        milestone_max = stage_cfg.max_tokens
        primary_val = None
    else:
        raise ValueError(stage)
    loader = PlanLoader(plan, sources, seq_len=seq)
    return loader, int(total_steps), int(milestone_max), primary_val


# --------------------------------------------------------------------------- #
# run one DAG node end to end
# --------------------------------------------------------------------------- #
def run_node(cfg: OrchestratorConfig, run_id: str, stage: str,
             init_wandb_fn: Callable = init_wandb) -> dict:
    """Execute (run_id, stage): load parent weights, resume-detect, drive, return summary."""
    spec = cfg.run_specs[run_id]
    lambda_frac, subset_tokens, seed = spec["lambda"], spec["subset_tokens"], spec["seed"]
    stage_cfg = StageConfig.from_json(cfg.stage_cfg_paths[stage])
    mc = ModelConfig.from_json(cfg.model_cfg_path)
    model_cfg_dict = mc.scientific_dict()

    parent_dir, parent_id = _parent_for(cfg, run_id, stage)
    if parent_dir is None:
        raise RuntimeError(f"parent checkpoint missing for {run_id}/{stage}")
    model = build_model(mc, attn_implementation=cfg.exec_cfg.attn_implementation)
    model.load_state_dict(ck.load_weights(parent_dir))  # bf16 payload -> fp32 params (copy casts)

    loader, total_steps, milestone_max, primary_val = _build_train(
        cfg, stage, lambda_frac, subset_tokens, seed, stage_cfg)
    trainer = Trainer(model, stage_cfg, cfg.exec_cfg, loader, total_steps=total_steps,
                      device=cfg.device, fused_adamw=cfg.fused_adamw)

    resumed = False
    resume_dir = _latest_resumable(cfg, run_id, stage)
    if resume_dir is not None:
        trainer.load_training_state(ck.load_training_state(resume_dir))
        resumed = True
    ch = config_hash({"model": model_cfg_dict, "stage": stage_cfg.scientific_dict(),
                      "lambda": lambda_frac, "subset_tokens": subset_tokens, "seed": seed})

    # crash-after-final guard: a completed-but-unrecorded stage would collide on the
    # unique final ckpt id — treat as already trained (node will be marked complete).
    if resumed and trainer.opt_step >= trainer.total_steps:
        return {"already_trained": True, "config_hash": ch, "parent_id": parent_id,
                "opt_step": trainer.opt_step}

    ctx = StageContext(
        run_id=run_id, stage=stage, lambda_frac=lambda_frac, subset_tokens=subset_tokens,
        seed=seed, milestone_max_tokens=milestone_max, config_hash=ch,
        dataset_manifest_hash=cfg.dataset_manifest_hash, environment_hash=cfg.environment_hash,
        git_commit=_git_commit(cfg.repo_root), tokenizer_ref=cfg.tokenizer_ref,
        parent_checkpoint_id=parent_id)

    metrics_path = cfg.artifact_root / "run_metrics" / f"{run_id}__{stage}.jsonl"
    wb = init_wandb_fn(run_id, stage, lambda_frac, subset_tokens,
                       {"config_hash": ch, **stage_cfg.scientific_dict()},
                       enabled=cfg.wandb_enabled)
    logger = MetricLogger(run_id, stage, metrics_path, wandb_run=wb)
    driver = StageDriver(trainer, ctx, stage_cfg, model_cfg_dict, artifact_root=cfg.artifact_root,
                         val_sets=_build_val_sets(cfg), logger=logger, now_fn=_now,
                         primary_val=primary_val, inventory_dir=cfg.inventory_dir)
    result = driver.run(resumed=resumed)
    return {"result": result, "config_hash": ch, "parent_id": parent_id,
            "metrics_path": str(metrics_path), "total_steps": total_steps,
            "opt_step": trainer.opt_step}


# --------------------------------------------------------------------------- #
# per-run completion audit (must pass before a node is marked complete)
# --------------------------------------------------------------------------- #
def audit_node(cfg: OrchestratorConfig, run_id: str, stage: str, config_hash_expected: str) -> dict:
    """Verify config hash / parent / required ckpts load / metrics present. Returns findings."""
    findings: list[str] = []
    recs = [r for r in inv.read_inventory(cfg.ckpt_inventory)
            if r.get("op") == "create" and r.get("run_id") == run_id and r.get("stage") == stage]
    if not recs:
        findings.append("no checkpoint records for node")
    for r in recs:
        if r.get("config_hash") != config_hash_expected:
            findings.append(f"config_hash mismatch on {r['checkpoint_id']}")
        d = cfg.artifact_root / r["rel_path"]
        if not (d.exists() and ck.is_complete(d)):
            findings.append(f"checkpoint missing/incomplete: {r['checkpoint_id']}")
        elif r.get("load_validation_status") != "verified":
            findings.append(f"checkpoint not load-validated: {r['checkpoint_id']}")
    labels = {tuple(r.get("milestone_labels", [])) for r in recs}
    need = {"stage1": [("incoming",), ("final",)], "stage2": [("incoming",), ("final",), ("restored_best",)],
            "stage3": [("incoming",), ("final",)]}[stage]
    for lbl in need:
        if lbl not in labels:
            findings.append(f"missing required milestone {lbl}")
    metrics_path = cfg.artifact_root / "run_metrics" / f"{run_id}__{stage}.jsonl"
    if not metrics_path.exists() or not metrics_path.read_text().strip():
        findings.append("metrics stream empty/missing")
    return {"run_id": run_id, "stage": stage, "ok": not findings, "findings": findings,
            "audited_at_utc": _now()}


# --------------------------------------------------------------------------- #
# canonical-state updates + run manifest / artifact inventories
# --------------------------------------------------------------------------- #
def _status_map(state: dict) -> dict:
    out = {}
    for rid, v in state.get("runs", {}).items():
        for s in dag.STAGES:
            out[f"{rid}::{s}"] = v.get(s, "planned")
    return out


def _set_stage_status(cfg: OrchestratorConfig, run_id: str, stage: str, value: str) -> str:
    state = _load_json(cfg.experiment_state_path) or {}
    state.setdefault("runs", {}).setdefault(run_id, {})[stage] = value
    state["updated_at_utc"] = _now()
    if not state.get("environment_hash"):
        state["environment_hash"] = cfg.environment_hash
    _atomic_write_json(cfg.experiment_state_path, state)
    return value


def _record_completion(cfg: OrchestratorConfig, run_id: str, stage: str, summary: dict,
                       audit: dict) -> None:
    git = _git_commit(cfg.repo_root)
    res = summary.get("result", {})
    _append_jsonl(cfg.repo_root / "runs" / "manifest.jsonl", {
        "run_id": run_id, "stage": stage, "lambda": cfg.run_specs[run_id]["lambda"],
        "subset_tokens": cfg.run_specs[run_id]["subset_tokens"], "seed": cfg.run_specs[run_id]["seed"],
        "config_hash": summary.get("config_hash"), "parent_checkpoint_id": summary.get("parent_id"),
        "opt_step": summary.get("opt_step"), "total_steps": summary.get("total_steps"),
        "final_metrics": res.get("final_metrics"), "tokens": res.get("tokens"),
        "stopped_early": res.get("stopped_early"), "best_val": res.get("best_val"),
        "metrics_path": summary.get("metrics_path"), "git_commit": git,
        "launch_command": "python -m j_pretrain.orchestration.run",
        "audit_ok": audit["ok"], "completed_at_utc": _now(),
    })
    # Inventories are small append-only records and live in the committed repo tree
    # (cfg.inventory_dir), alongside checkpoint_inventory.jsonl — not under artifact_root.
    _append_jsonl(cfg.inventory_dir / "run_artifact_inventory.jsonl", {
        "run_id": run_id, "stage": stage, "metrics_path": summary.get("metrics_path"),
        "checkpoints_dir": str((cfg.artifact_root / "checkpoints" / run_id / stage)),
        "config_hash": summary.get("config_hash"), "git_commit": git, "at_utc": _now()})
    _append_jsonl(cfg.inventory_dir / "backup_status.jsonl", {
        "run_id": run_id, "stage": stage, "backup_status": "unreplicated_local_copy",
        "at_utc": _now()})
    try:
        st = os.statvfs(str(cfg.artifact_root))
        free_gb = st.f_bavail * st.f_frsize / (1 << 30)
    except Exception:
        free_gb = None
    _append_jsonl(cfg.inventory_dir / "storage_usage.jsonl", {
        "run_id": run_id, "stage": stage, "free_gb": free_gb,
        "checkpoints_bytes": _dir_bytes(cfg.artifact_root / "checkpoints")
        if (cfg.artifact_root / "checkpoints").exists() else 0, "at_utc": _now()})
    # per-run audit record (committed)
    _atomic_write_json(cfg.repo_root / "reports" / "per_run_audits" / f"{run_id}__{stage}.json", audit)


# --------------------------------------------------------------------------- #
# top-level orchestration loop
# --------------------------------------------------------------------------- #
def orchestrate(cfg: OrchestratorConfig, max_nodes: Optional[int] = None,
                init_wandb_fn: Callable = init_wandb, acquire_lock: bool = True) -> dict:
    """Drive ready DAG nodes one at a time until none remain (or ``max_nodes`` reached)."""
    lock = GpuLock(cfg.gpu_lock_path)
    if acquire_lock:
        try:
            lock.acquire(run_id="orchestrator", stage="loop", tmux_session=os.environ.get("TMUX"),
                         now_utc=_now())
        except GpuLockError as e:
            return {"status": "lock_held", "detail": str(e)}
    ensure_init_checkpoint(cfg)
    processed, retry_counts = [], {}
    try:
        while max_nodes is None or len(processed) < max_nodes:
            state = _load_json(cfg.experiment_state_path) or {}
            status = _status_map(state)
            if dag.is_complete(cfg.run_ids, status):
                return {"status": "complete", "processed": processed}
            node = dag.next_node(cfg.run_ids, status, init_ready=True, order=cfg.run_ids)
            if node is None:
                return {"status": "no_ready_node", "processed": processed}
            key = node.key()
            _set_stage_status(cfg, node.run_id, node.stage, "running")
            try:
                summary = run_node(cfg, node.run_id, node.stage, init_wandb_fn=init_wandb_fn)
                audit = audit_node(cfg, node.run_id, node.stage,
                                   summary.get("config_hash", ""))
                if not audit["ok"]:
                    _set_stage_status(cfg, node.run_id, node.stage, "failed_retryable")
                    _record_completion(cfg, node.run_id, node.stage, summary, audit)
                    retry_counts[key] = retry_counts.get(key, 0) + 1
                    if retry_counts[key] > 3:
                        _set_stage_status(cfg, node.run_id, node.stage, "failed_blocked")
                        return {"status": "audit_failed", "node": key, "audit": audit,
                                "processed": processed}
                    continue
                _record_completion(cfg, node.run_id, node.stage, summary, audit)
                _set_stage_status(cfg, node.run_id, node.stage, "complete")
                processed.append(key)
            except Exception as e:  # noqa: BLE001 — record + retry per mission failure policy
                retry_counts[key] = retry_counts.get(key, 0) + 1
                status_val = "failed_blocked" if retry_counts[key] > 3 else "failed_retryable"
                _set_stage_status(cfg, node.run_id, node.stage, status_val)
                _append_jsonl(cfg.repo_root / "logs" / "orchestrator_errors.jsonl",
                              {"node": key, "error": repr(e), "retry": retry_counts[key],
                               "at_utc": _now()})
                if status_val == "failed_blocked":
                    return {"status": "node_failed", "node": key, "error": repr(e),
                            "processed": processed}
        return {"status": "max_nodes_reached", "processed": processed}
    finally:
        if acquire_lock:
            lock.release()


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure-3a single-GPU orchestrator")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="process at most N nodes then stop (default: run to completion)")
    ap.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    ap.add_argument("--artifact-root", default=None)
    args = ap.parse_args()
    cfg = default_config(artifact_root=args.artifact_root, wandb_enabled=not args.no_wandb)
    result = orchestrate(cfg, max_nodes=args.max_nodes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in ("complete", "max_nodes_reached", "no_ready_node") else 1


if __name__ == "__main__":
    raise SystemExit(main())
