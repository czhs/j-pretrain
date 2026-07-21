"""Drive ONE stage of ONE run: training loop + milestone checkpoints + eval +
early stopping + resume + inventory recording.

Wires together :class:`~j_pretrain.training.loop.Trainer`, the checkpoint schedules
(:mod:`j_pretrain.training.schedule`), the atomic checkpoint writer + append-only
inventory (:mod:`j_pretrain.artifacts`), and the metric logger. The orchestrator
constructs one driver per DAG node and calls :meth:`run`.

Driver token axis (which cumulative count the milestone schedules key off):
    stage1 -> total tokens ; stage2 -> MusicPile tokens ; stage3 -> ChemPile tokens.

Checkpoint policy:
    * "incoming" (step 0, unless resuming) and "final": write BOTH an analysis snapshot
      and a full resumable (distinct dirs, both milestone-labelled and inventoried).
    * analysis-mark crossings: analysis snapshot only.
    * resumable-mark crossings: full resumable only.
    * stage2 new best-val: analysis + resumable labelled "best".
Resume: if a valid resumable exists for (run, stage) it is loaded and training
continues from its optimizer step — a healthy run is never restarted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import torch

from j_pretrain.artifacts import checkpoint as ck
from j_pretrain.artifacts import inventory as inv
from j_pretrain.config.hashing import short_hash
from j_pretrain.config.schemas import ExecConfig, StageConfig
from j_pretrain.orchestration.metrics import MetricLogger
from j_pretrain.training import schedule as sch
from j_pretrain.training.loop import Trainer

_DRIVER_AXIS = {"stage1": "total", "stage2": "mp", "stage3": "chempile"}


@dataclass
class StageContext:
    run_id: str
    stage: str
    lambda_frac: float
    subset_tokens: int
    seed: int
    milestone_max_tokens: int          # bounds the schedule mark list (per-run for s1)
    config_hash: str
    dataset_manifest_hash: str
    environment_hash: str
    git_commit: str
    tokenizer_ref: dict
    parent_checkpoint_id: Optional[str]  # init (s1) / stage1-final (s2) / stage2-best (s3)


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class StageDriver:
    def __init__(
        self,
        trainer: Trainer,
        ctx: StageContext,
        stage_cfg: StageConfig,
        model_config_dict: dict,
        artifact_root: str | Path,
        val_sets: Mapping[str, object],
        logger: MetricLogger,
        now_fn: Callable[[], str],
        analysis_token_marks: Optional[list[int]] = None,
        resumable_token_marks: Optional[list[int]] = None,
        primary_val: Optional[str] = None,   # eval key used for best/early-stop (stage2)
    ):
        self.tr = trainer
        self.ctx = ctx
        self.cfg = stage_cfg
        self.model_config_dict = model_config_dict
        self.ckpt_root = Path(artifact_root) / "checkpoints"
        self.artifact_root = Path(artifact_root)
        self.val_sets = dict(val_sets)
        self.log = logger
        self.now = now_fn
        self.axis = _DRIVER_AXIS[ctx.stage]
        self.a_marks = (analysis_token_marks if analysis_token_marks is not None
                        else sch.analysis_marks(ctx.stage, ctx.milestone_max_tokens))
        self.r_marks = (resumable_token_marks if resumable_token_marks is not None
                        else sch.resumable_marks(ctx.stage, ctx.milestone_max_tokens))
        self.primary_val = primary_val
        self._prev_driver_tokens = self._driver_tokens()
        self._prev_eval_tokens = 0
        self._incoming_id: Optional[str] = None
        self._stopped_early = False

    # ---- token axis ------------------------------------------------------
    def _driver_tokens(self) -> int:
        return getattr(self.tr.tokens, self.axis)

    # ---- checkpoint helpers ---------------------------------------------
    def _ckpt_id(self, kind: str, cls: str) -> str:
        dt = self._driver_tokens()
        h = short_hash((self.ctx.run_id, self.ctx.stage, cls, kind, self.tr.opt_step, dt))
        tag = "an" if cls == ck.CLASS_ANALYSIS else "rs"
        return f"{self.ctx.stage}-{tag}-{kind}-tok{dt}-step{self.tr.opt_step}-{h}"

    def _meta(self, checkpoint_id: str, cls: str, milestones: list[str],
              val_metrics: dict[str, float]) -> ck.CheckpointMeta:
        t = self.tr.tokens
        return ck.CheckpointMeta(
            run_id=self.ctx.run_id, checkpoint_id=checkpoint_id, stage=self.ctx.stage,
            checkpoint_class=cls, milestone_labels=milestones,
            lambda_frac=self.ctx.lambda_frac, subset_tokens=self.ctx.subset_tokens,
            step=self.tr.opt_step, total_tokens=t.total, c4_tokens=t.c4,
            musicpile_tokens=t.mp, chempile_tokens=t.chempile,
            lr=self.tr.opt.param_groups[0]["lr"], train_loss=None,
            val_metrics=val_metrics, parent_checkpoint_id=self._incoming_id or self.ctx.parent_checkpoint_id,
            config_hash=self.ctx.config_hash, dataset_manifest_hash=self.ctx.dataset_manifest_hash,
            environment_hash=self.ctx.environment_hash, git_commit=self.ctx.git_commit,
            seed=self.ctx.seed, created_at_utc=self.now(), tokenizer_ref=self.ctx.tokenizer_ref,
        )

    def _write(self, cls: str, milestones: list[str], val_metrics: dict[str, float],
               resumable: bool) -> str:
        cid = self._ckpt_id("-".join(milestones), cls)
        meta = self._meta(cid, cls, milestones, val_metrics)
        tstate = self.tr.training_state() if resumable else None
        final = ck.write_checkpoint(self.ckpt_root, meta, self.tr.model,
                                    self.model_config_dict, training_state=tstate)
        inv.record_checkpoint(meta, rel_path=str(final.relative_to(self.artifact_root)),
                              byte_size=_dir_bytes(final), created_at_utc=meta.created_at_utc,
                              inventory_path=self.artifact_root / inv.CHECKPOINT_INVENTORY)
        return cid

    def _write_both(self, milestones: list[str], val_metrics: dict[str, float]) -> str:
        """Analysis + resumable for a milestone that needs both (incoming/final/best)."""
        a = self._write(ck.CLASS_ANALYSIS, milestones, val_metrics, resumable=False)
        self._write(ck.CLASS_RESUMABLE, milestones, val_metrics, resumable=True)
        return a

    # ---- evaluation ------------------------------------------------------
    def _evaluate(self) -> dict[str, float]:
        return {name: self.tr.evaluate(ds) for name, ds in self.val_sets.items()}

    # ---- main loop -------------------------------------------------------
    def run(self, resumed: bool = False, chunk: int = 1) -> dict:
        if not resumed:
            vm = self._evaluate()
            self._incoming_id = self._write_both(["incoming"], vm)
            self.log.log({"event": "incoming", "opt_step": self.tr.opt_step,
                          "driver_tokens": self._driver_tokens(), **_prefix(vm, "val")})

        while self.tr.opt_step < self.tr.total_steps and not self._stopped_early:
            target = min(self.tr.opt_step + chunk, self.tr.total_steps)
            n = target - self.tr.opt_step
            losses: list[float] = []
            self.tr.train_steps(n, on_step=lambda t, l: losses.append(l))
            self._after_chunk(losses[-1] if losses else float("nan"))

        # final checkpoint + eval
        vm = self._evaluate()
        self._write_both(["final"], vm)
        self.log.log({"event": "final", "opt_step": self.tr.opt_step,
                      "driver_tokens": self._driver_tokens(), **_prefix(vm, "val")})

        result = {"final_metrics": vm, "opt_step": self.tr.opt_step,
                  "driver_tokens": self._driver_tokens(),
                  "tokens": self.tr.tokens.as_dict(), "stopped_early": self._stopped_early,
                  "best_val": self.tr.best_val}
        # stage2: restore best -> Stage 3 uses this
        if self.ctx.stage == "stage2" and self.primary_val is not None:
            result["restored_best_written"] = True
            self._write_both(["restored_best"], {self.primary_val: self.tr.best_val or float("nan")})
        self.log.close()
        return result

    def _after_chunk(self, last_loss: float) -> None:
        dt = self._driver_tokens()
        # scheduled resumables
        if sch._crossings(self.r_marks, self._prev_driver_tokens, dt):
            self._write(ck.CLASS_RESUMABLE, ["scheduled"], {}, resumable=True)
        # scheduled analysis snapshots
        crossed_a = sch._crossings(self.a_marks, self._prev_driver_tokens, dt)
        if crossed_a:
            self._write(ck.CLASS_ANALYSIS, [f"tok{crossed_a[-1]}"], {}, resumable=False)
        # eval + best/early-stop
        if dt - self._prev_eval_tokens >= self.cfg.eval_interval_tokens:
            vm = self._evaluate()
            self._prev_eval_tokens = dt
            self.log.log({"event": "eval", "opt_step": self.tr.opt_step,
                          "driver_tokens": dt, "train_loss": last_loss, **_prefix(vm, "val")})
            self._maybe_best_and_earlystop(vm)
        else:
            self.log.log({"event": "step", "opt_step": self.tr.opt_step,
                          "driver_tokens": dt, "train_loss": last_loss,
                          "lr": self.tr.opt.param_groups[0]["lr"]})
        self._prev_driver_tokens = dt

    def _maybe_best_and_earlystop(self, vm: dict[str, float]) -> None:
        if self.primary_val is None or self.primary_val not in vm:
            return
        cur = vm[self.primary_val]
        improved = self.tr.best_val is None or cur < self.tr.best_val - self.cfg.early_stop_min_delta
        if improved:
            self.tr.best_val = cur
            self.tr.no_improve_evals = 0
            self._write_both(["best"], vm)
        else:
            self.tr.no_improve_evals += 1
            if (self.cfg.early_stop_patience is not None
                    and self.tr.no_improve_evals >= self.cfg.early_stop_patience):
                self._stopped_early = True
                self.log.log({"event": "early_stop", "opt_step": self.tr.opt_step,
                              "no_improve_evals": self.tr.no_improve_evals})


def _prefix(d: dict[str, float], p: str) -> dict[str, float]:
    return {f"{p}_{k}": v for k, v in d.items()}
