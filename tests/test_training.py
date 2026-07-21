"""Tests for the training stack: LR/checkpoint schedules, resumable data plans,
loader, optimizer groups, RNG roundtrip, and a deterministic resume integration test.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from j_pretrain.config.schemas import ExecConfig, StageConfig
from j_pretrain.training import schedule as sch
from j_pretrain.training.dataplan import ShuffledSourcePlan, Stage1Plan
from j_pretrain.training.loader import PlanLoader
from j_pretrain.training.loop import Trainer
from j_pretrain.training.optim import build_adamw
from j_pretrain.training.rngstate import capture_rng, restore_rng


# --------------------------------------------------------------------------- #
# LR + checkpoint schedules
# --------------------------------------------------------------------------- #
def test_cosine_warmup_peak_min():
    peak, mn, warm, total = 5e-4, 5e-5, 10, 100
    assert sch.cosine_lr(0, peak, mn, warm, total) == pytest.approx(peak / warm)
    assert sch.cosine_lr(warm - 1, peak, mn, warm, total) == pytest.approx(peak)
    # monotone non-increasing after warmup, ends at min
    lrs = [sch.cosine_lr(s, peak, mn, warm, total) for s in range(warm, total)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))
    assert sch.cosine_lr(total, peak, mn, warm, total) == pytest.approx(mn)
    mid = sch.cosine_lr((warm + total) // 2, peak, mn, warm, total)
    assert mn < mid < peak


def test_analysis_and_resumable_marks_stage1():
    a = sch.analysis_marks("stage1", 1_100_000_000)
    for m in (1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000):
        assert m in a
    assert 350_000_000 in a and 1_100_000_000 in a  # 100M then +250M steps
    r = sch.resumable_marks("stage1", 1_100_000_000)
    assert r[0] == 500_000_000 and r[-1] == 1_000_000_000


def test_crossings_detect_marks():
    assert sch.crossed_analysis("stage3", 200_000_000, 0, 1_500_000) == [1_000_000]
    assert sch.crossed_resumable("stage3", 200_000_000, 40_000_000, 60_000_000) == [50_000_000]
    assert sch.crossed_analysis("stage3", 200_000_000, 11_000_000, 19_000_000) == []


# --------------------------------------------------------------------------- #
# Data plans: Stage1 interleave (O(log n)) vs reference, C4 order invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_c4,n_mp", [(100, 0), (100, 1), (100, 7), (97, 13), (1000, 250)])
def test_stage1_plan_matches_reference(n_c4, n_mp):
    assert Stage1Plan(n_c4, n_mp).verify_against_reference()


def test_c4_order_identical_across_lambda():
    """The ordered stream of C4 local indices must be 0,1,2,... for every n_mp."""
    for n_mp in (0, 5, 50, 200):
        p = Stage1Plan(1000, n_mp)
        c4_seq = [p.at(g)[1] for g in range(len(p)) if p.at(g)[0] == "c4"]
        assert c4_seq == list(range(1000))


def test_mp_is_nested_prefix():
    small = [Stage1Plan(1000, 50).at(g) for g in range(1050) if Stage1Plan(1000, 50).at(g)[0] == "mp"]
    assert [m[1] for m in small] == list(range(50))  # MusicPile windows 0..49 in order


def test_shuffled_source_plan_determinism_and_coverage():
    a = ShuffledSourcePlan("mp", 64, seed=1)
    b = ShuffledSourcePlan("mp", 64, seed=1)
    assert [a.at(g) for g in range(200)] == [b.at(g) for g in range(200)]
    # each epoch is a permutation (full coverage, no repeats within an epoch)
    epoch0 = [a.at(g)[1] for g in range(64)]
    assert sorted(epoch0) == list(range(64))
    epoch1 = [a.at(g)[1] for g in range(64, 128)]
    assert sorted(epoch1) == list(range(64)) and epoch0 != epoch1  # reshuffled
    assert all(a.at(g)[0] == "mp" for g in range(200))
    # different seed => different order
    assert [a.at(g)[1] for g in range(64)] != [ShuffledSourcePlan("mp", 64, seed=2).at(g)[1] for g in range(64)]


def test_no_shuffle_is_packed_order():
    p = ShuffledSourcePlan("chempile", 10, seed=0, shuffle=False)
    assert [p.at(g) for g in range(25)] == [("chempile", g % 10) for g in range(25)]


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
class FakePacked:
    """Minimal PackedDataset stand-in: rows of a fixed-width int array."""
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.seq_len = arr.shape[1]

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, i):
        return self.arr[i]


def test_loader_shapes_and_counts():
    seq = 4
    c4 = FakePacked(np.arange(20 * seq, dtype=np.uint16).reshape(20, seq))
    mp = FakePacked((np.arange(10 * seq, dtype=np.uint16).reshape(10, seq) + 1000))
    plan = Stage1Plan(20, 4)
    ld = PlanLoader(plan, {"c4": c4, "mp": mp}, seq_len=seq)
    ids, counts = ld.windows(0, 6)
    assert ids.shape == (6, seq) and ids.dtype == torch.int64
    assert counts["c4"] + counts["mp"] == 6
    # first C4 row equals source row 0
    src, local = plan.at(0)
    ref = c4[local] if src == "c4" else mp[local]
    assert torch.equal(ids[0], torch.from_numpy(ref.astype(np.int64)))


def test_loader_rejects_seq_len_mismatch():
    with pytest.raises(ValueError):
        PlanLoader(Stage1Plan(4, 0),
                   {"c4": FakePacked(np.zeros((4, 8), np.uint16))}, seq_len=4)


# --------------------------------------------------------------------------- #
# Optimizer param groups + RNG
# --------------------------------------------------------------------------- #
def test_adamw_decay_groups_exclude_norm_embed_bias():
    cfg = _tiny_stage_cfg()
    m = _tiny_lm()
    opt = build_adamw(m, cfg, fused=False)
    decay_params = {id(p) for p in opt.param_groups[0]["params"]}
    for name, p in m.named_parameters():
        if p.ndim < 2 or "norm" in name.lower() or "embed" in name.lower():
            assert id(p) not in decay_params, f"{name} should be weight-decay-exempt"
    assert opt.param_groups[0]["weight_decay"] == cfg.weight_decay
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_rng_roundtrip():
    torch.manual_seed(123)
    _ = torch.rand(5)
    state = capture_rng()
    a = torch.rand(5)
    restore_rng(state)
    b = torch.rand(5)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Deterministic resume integration test (MANDATORY)
# --------------------------------------------------------------------------- #
def _tiny_lm(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        rms_norm_eps=1e-5, tie_word_embeddings=True, attention_bias=False,
        attention_dropout=0.0, rope_theta=10000.0,
    )
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg)


def _tiny_stage_cfg(seq_len: int = 16) -> StageConfig:
    return StageConfig(
        stage="stage1", peak_lr=1e-3, min_lr=1e-4, warmup_steps=2,
        global_batch_seqs=2, max_tokens=10_000, seq_len=seq_len, weight_decay=0.1,
    )


def _make_loader(seq_len: int) -> PlanLoader:
    rng = np.random.default_rng(7)
    pool = rng.integers(0, 64, size=(80, seq_len), dtype=np.uint16)
    ds = FakePacked(pool)
    plan = ShuffledSourcePlan("c4", pool_windows=len(ds), seed=3, shuffle=True)
    return PlanLoader(plan, {"c4": ds}, seq_len=seq_len)


def _snapshot(blob: dict) -> dict:
    """Round-trip through a buffer to fully detach from live tensors (simulate disk)."""
    buf = io.BytesIO()
    torch.save(blob, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def test_deterministic_resume_matches_continuous():
    torch.use_deterministic_algorithms(True, warn_only=True)
    seq_len = 16
    N, K = 8, 3
    exec_cfg = ExecConfig(microbatch_size=1, torch_compile=False)
    cfg = _tiny_stage_cfg(seq_len)

    # --- Run A: continuous N steps ---
    tr_a = Trainer(_tiny_lm(seed=0), cfg, exec_cfg, _make_loader(seq_len),
                   total_steps=N, device="cpu")
    losses_a: list[float] = []
    tr_a.train_steps(N, on_step=lambda t, l: losses_a.append(l))
    params_a = {k: v.clone() for k, v in tr_a.model.state_dict().items()}

    # --- Run B: K steps, checkpoint, restart into a fresh Trainer, finish ---
    tr_b = Trainer(_tiny_lm(seed=0), cfg, exec_cfg, _make_loader(seq_len),
                   total_steps=N, device="cpu")
    losses_b: list[float] = []
    tr_b.train_steps(K, on_step=lambda t, l: losses_b.append(l))
    blob = _snapshot(tr_b.training_state())

    tr_c = Trainer(_tiny_lm(seed=999), cfg, exec_cfg, _make_loader(seq_len),
                   total_steps=N, device="cpu")  # different init, then load
    tr_c.load_training_state(blob)
    assert tr_c.windows_consumed == K * cfg.grad_accum(exec_cfg.microbatch_size)
    tr_c.train_steps(N - K, on_step=lambda t, l: losses_b.append(l))

    # losses over all N steps agree
    assert len(losses_b) == N
    for la, lb in zip(losses_a, losses_b):
        assert la == pytest.approx(lb, abs=1e-5)
    # final parameters agree
    params_c = tr_c.model.state_dict()
    for k in params_a:
        assert torch.allclose(params_a[k], params_c[k], atol=1e-5), k
    # token counters consistent
    assert tr_c.tokens.total == N * cfg.global_batch_tokens


def test_evaluate_returns_finite_ce():
    seq_len = 16
    cfg = _tiny_stage_cfg(seq_len)
    tr = Trainer(_tiny_lm(seed=0), cfg, ExecConfig(microbatch_size=2, torch_compile=False),
                 _make_loader(seq_len), total_steps=4, device="cpu")
    val = FakePacked(np.random.default_rng(0).integers(0, 64, (6, seq_len), dtype=np.uint16))
    ce = tr.evaluate(val)
    assert np.isfinite(ce) and ce > 0
