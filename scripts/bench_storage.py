"""Measure real analysis-snapshot and full-resumable checkpoint sizes for the 135M
model, then project total permanent storage for the locked experiment.

Writes to a temp dir (never the real artifact tree), measures on-disk bytes, cleans
up, and prints a JSON summary consumed by docs/STORAGE_PLAN.md + reports/FEASIBILITY.md.
Run on CPU (sizes are device-independent); does not touch the GPU or the dataset build.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch

from j_pretrain.artifacts.checkpoint import CheckpointMeta, write_checkpoint
from j_pretrain.config.schemas import ExecConfig, StageConfig
from j_pretrain.models.build import build_model, count_parameters, load_model_config
from j_pretrain.training.loop import Trainer
from j_pretrain.training.loader import PlanLoader
from j_pretrain.training.dataplan import ShuffledSourcePlan
import numpy as np


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class _FakePacked:
    def __init__(self, arr):
        self.arr = arr
        self.seq_len = arr.shape[1]
    def __len__(self):
        return self.arr.shape[0]
    def __getitem__(self, i):
        return self.arr[i]


def main() -> None:
    mc = load_model_config()
    model = build_model(mc, seed=1234, dtype=torch.float32)  # fp32 master params
    total, unique = count_parameters(model)

    tmp = Path(tempfile.mkdtemp(prefix="ckpt_bench_"))
    meta_common = dict(
        run_id="bench", lambda_frac=0.0, subset_tokens=300_000_000, step=0,
        total_tokens=0, c4_tokens=0, musicpile_tokens=0, chempile_tokens=0, lr=5e-4,
        train_loss=None, val_metrics={}, parent_checkpoint_id=None, config_hash="bench",
        dataset_manifest_hash="bench", environment_hash="bench", git_commit="bench",
        seed=1234, created_at_utc="2026-07-21T00:00:00Z",
        tokenizer_ref={"id": mc.tokenizer},
    )
    model_cfg_dict = mc.scientific_dict()

    # 1) analysis snapshot (bf16 safetensors)
    a_meta = CheckpointMeta(checkpoint_id="analysis-bench", stage="stage1",
                            checkpoint_class="analysis", milestone_labels=["bench"],
                            **meta_common)
    a_dir = write_checkpoint(tmp, a_meta, model, model_cfg_dict)
    analysis_bytes = _dir_bytes(a_dir)

    # 2) full resumable: populate real AdamW state via one tiny optimizer step
    seq = mc.train_seq_len
    pool = np.random.default_rng(0).integers(0, mc.vocab_size, (8, seq), dtype=np.uint16)
    loader = PlanLoader(ShuffledSourcePlan("c4", 8, seed=0), {"c4": _FakePacked(pool)}, seq)
    scfg = StageConfig.from_json("configs/stage1/music.json")
    scfg = StageConfig(**{**{k: getattr(scfg, k) for k in scfg.__dataclass_fields__},
                          "global_batch_seqs": 1})  # tiny step for state population
    tr = Trainer(model, scfg, ExecConfig(microbatch_size=1, torch_compile=False),
                 loader, total_steps=1, device="cpu")
    tr.train_steps(1)
    r_meta = CheckpointMeta(checkpoint_id="resumable-bench", stage="stage1",
                            checkpoint_class="resumable", milestone_labels=["bench"],
                            **meta_common)
    r_dir = write_checkpoint(tmp, r_meta, tr.model, model_cfg_dict,
                             training_state=tr.training_state())
    resumable_bytes = _dir_bytes(r_dir)

    shutil.rmtree(tmp)

    MB = 1024 ** 2
    GB = 1024 ** 3
    result = {
        "param_count_total": total,
        "param_count_unique": unique,
        "analysis_snapshot_bytes": analysis_bytes,
        "analysis_snapshot_MB": round(analysis_bytes / MB, 1),
        "resumable_ckpt_bytes": resumable_bytes,
        "resumable_ckpt_MB": round(resumable_bytes / MB, 1),
    }

    # ---- projection over the locked checkpoint schedules -----------------
    from j_pretrain.training.schedule import analysis_marks, resumable_marks

    lam_tokens = {0: 8_700_000_000, 0.25: 8_775_000_000, 0.5: 8_850_000_000,
                  0.75: 8_925_000_000, 1.0: 9_000_000_000}
    n_analysis = 0
    # Stage 1: per-lambda analysis marks (+ init shared once, + ~2 LR-boundary, + final)
    for lam, toks in lam_tokens.items():
        n_analysis += len(analysis_marks("stage1", toks)) + 2 + 1  # marks + LR-bounds + final
    n_analysis += 1  # shared init (byte-identical, stored once)
    # Stage 2: marks up to 2B + best-val(~4) + incoming + restored-best, x5
    n_analysis += 5 * (len(analysis_marks("stage2", 2_000_000_000)) + 4 + 1 + 1)
    # Stage 3: marks up to 200M + incoming + final, x5
    n_analysis += 5 * (len(analysis_marks("stage3", 200_000_000)) + 1 + 1)

    # Permanent resumables (retention policy): init(shared) + per-run
    # stage1-final, stage2-{incoming,best,final,restored}, stage3-{incoming,final} = 7/run
    n_resumable_perm = 1 + 5 * 7
    # rolling transient: keep 2 most recent per active run (1 active at a time)
    n_resumable_rolling = 2

    analysis_total = n_analysis * analysis_bytes
    resumable_total = (n_resumable_perm + n_resumable_rolling) * resumable_bytes
    tokenized_bytes = 17 * GB  # measured separately (see STORAGE_PLAN)
    misc_logs = 5 * GB
    subtotal = analysis_total + resumable_total + tokenized_bytes + misc_logs
    with_headroom = subtotal * 1.15

    result["projection"] = {
        "n_analysis_snapshots": n_analysis,
        "n_resumable_permanent": n_resumable_perm,
        "n_resumable_rolling": n_resumable_rolling,
        "analysis_total_GB": round(analysis_total / GB, 1),
        "resumable_total_GB": round(resumable_total / GB, 1),
        "tokenized_GB_est": 17,
        "logs_misc_GB_est": 5,
        "subtotal_GB": round(subtotal / GB, 1),
        "with_15pct_headroom_GB": round(with_headroom / GB, 1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
