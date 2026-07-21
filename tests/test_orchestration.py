"""Tests for the exclusive GPU lock and the run dependency DAG."""
from __future__ import annotations

import os

import pytest

from j_pretrain.orchestration import dag
from j_pretrain.orchestration.dag import Node
from j_pretrain.orchestration.gpulock import GpuLock, GpuLockError


# --------------------------------------------------------------------------- #
# GPU lock
# --------------------------------------------------------------------------- #
def test_lock_acquire_and_reentrant(tmp_path):
    lk = GpuLock(tmp_path / "gpu.lock")
    info = lk.acquire("run-a", "stage1", "tmux-a", "2026-07-21T00:00:00Z")
    assert info.pid == os.getpid()
    assert lk.is_held_by_live_process()
    # same process re-acquire is re-entrant, not an error
    again = lk.acquire("run-a", "stage1", "tmux-a", "2026-07-21T00:01:00Z")
    assert again.pid == info.pid


def test_lock_refuses_second_live_holder(tmp_path):
    lk = GpuLock(tmp_path / "gpu.lock")
    # simulate a live foreign holder: use our own pid (definitely alive) but pretend
    # the acquirer is a different pid.
    lk.acquire("run-a", "stage1", None, "t", pid=os.getpid())
    with pytest.raises(GpuLockError):
        lk.acquire("run-b", "stage1", None, "t", pid=os.getpid() + 10_000_000)


def test_lock_reclaims_stale(tmp_path):
    lk = GpuLock(tmp_path / "gpu.lock")
    dead_pid = 2_000_000_000  # not a real live pid
    lk.acquire("run-a", "stage1", None, "t", pid=dead_pid)
    assert not lk.is_held_by_live_process()  # owner dead => stale
    info = lk.acquire("run-b", "stage2", None, "t2")  # reclaims
    assert info.run_id == "run-b" and info.pid == os.getpid()


def test_lock_release_only_owner(tmp_path):
    lk = GpuLock(tmp_path / "gpu.lock")
    lk.acquire("run-a", "stage1", None, "t", pid=1)  # pid 1 = always live, not us
    assert lk.release(pid=os.getpid()) is False  # not ours, owner alive -> refuse
    assert lk.path.exists()


def test_corrupt_lock_treated_as_absent(tmp_path):
    p = tmp_path / "gpu.lock"
    p.write_text("{not json")
    lk = GpuLock(p)
    assert lk.read() is None and lk.is_held_by_live_process() is False
    assert lk.acquire("run-a", "stage1", None, "t").run_id == "run-a"


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #
RUNS = ["music-300m_lambda-0.0", "music-300m_lambda-0.25", "music-300m_lambda-0.5"]


def test_only_stage1_ready_initially():
    status: dict[str, str] = {}
    ready = dag.ready_nodes(RUNS, status, init_ready=True)
    assert {n.stage for n in ready} == {"stage1"}
    assert len(ready) == len(RUNS)


def test_stage1_blocked_until_init_ready():
    assert dag.ready_nodes(RUNS, {}, init_ready=False) == []


def test_stage2_waits_for_stage1_same_run():
    r = RUNS[0]
    status = {Node(r, "stage1").key(): "complete"}
    ready = dag.ready_nodes(RUNS, status)
    assert Node(r, "stage2") in ready
    # other runs' stage2 not ready (their stage1 not complete)
    assert Node(RUNS[1], "stage2") not in ready


def test_no_cross_run_dependency():
    # completing run0 entirely must not unlock run1's stage2
    r0 = RUNS[0]
    status = {Node(r0, s).key(): "complete" for s in dag.STAGES}
    ready = dag.ready_nodes(RUNS, status)
    assert all(n.run_id != r0 for n in ready)  # run0 fully done
    assert Node(RUNS[1], "stage1") in ready
    assert Node(RUNS[1], "stage2") not in ready


def test_next_node_prefers_in_progress_run_then_order():
    r0 = RUNS[0]
    # run0 stage1 complete, stage2 planned -> should pick run0 stage2 before run1 stage1
    status = {Node(r0, "stage1").key(): "complete"}
    nxt = dag.next_node(RUNS, status, order=RUNS)
    assert nxt == Node(r0, "stage2")


def test_next_node_lambda0_first_when_fresh():
    nxt = dag.next_node(RUNS, {}, order=RUNS)
    assert nxt == Node(RUNS[0], "stage1")  # lambda=0 first for the Stage-2 pilot


def test_is_complete():
    status = {Node(r, s).key(): "complete" for r in RUNS for s in dag.STAGES}
    assert dag.is_complete(RUNS, status)
    status.pop(Node(RUNS[0], "stage3").key())
    assert not dag.is_complete(RUNS, status)
