"""Run dependency DAG.

Node = ``(run_id, stage)``. Edges (per run, no cross-run reuse — the Fig-3a
guarantee that the lambda=0 Stage-1 checkpoint feeds ONLY the lambda=0 condition):

    init (shared) -> (run, stage1) -> (run, stage2) -> (run, stage3)

The DAG answers "what is ready to run next" given a status map, so the orchestrator
can drive one GPU job at a time and skip already-complete nodes on resume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

STAGES = ("stage1", "stage2", "stage3")
TERMINAL_OK = {"complete"}
IN_PROGRESS = {"ready", "running", "checkpointing", "evaluating", "complete_unverified"}


@dataclass(frozen=True)
class Node:
    run_id: str
    stage: str

    def key(self) -> str:
        return f"{self.run_id}::{self.stage}"


def all_nodes(run_ids: Iterable[str]) -> list[Node]:
    return [Node(r, s) for r in run_ids for s in STAGES]


def dependencies(node: Node) -> list[Node]:
    """Prior stage of the same run (Stage-1 depends only on the shared init)."""
    i = STAGES.index(node.stage)
    if i == 0:
        return []  # depends on shared init, handled separately (init_ready flag)
    return [Node(node.run_id, STAGES[i - 1])]


def is_ready(node: Node, status: dict[str, str], init_ready: bool) -> bool:
    """A node is ready iff it is not started and every dependency is complete."""
    st = status.get(node.key(), "planned")
    if st != "planned":
        return False
    if node.stage == "stage1" and not init_ready:
        return False
    return all(status.get(d.key(), "planned") in TERMINAL_OK for d in dependencies(node))


def ready_nodes(run_ids: Iterable[str], status: dict[str, str],
                init_ready: bool = True) -> list[Node]:
    """All currently-ready nodes, in a stable (run, stage) order."""
    return [n for n in all_nodes(run_ids) if is_ready(n, status, init_ready)]


def next_node(run_ids: Iterable[str], status: dict[str, str], init_ready: bool = True,
              order: list[str] | None = None) -> Node | None:
    """Pick the single next node to launch.

    ``order`` optionally prioritizes run_ids (e.g. lambda=0 first so the Stage-2
    pilot can finalize the fixed config). Within a run, earlier stages first;
    a partially-advanced run is finished before an untouched one is started.
    """
    ready = ready_nodes(run_ids, status, init_ready)
    if not ready:
        return None
    order = order or list(run_ids)

    def rank(n: Node) -> tuple[int, int, int]:
        # prefer: (a) runs already in progress, (b) explicit run order, (c) earlier stage
        run_started = any(status.get(Node(n.run_id, s).key(), "planned") in TERMINAL_OK
                          for s in STAGES)
        run_rank = order.index(n.run_id) if n.run_id in order else len(order)
        return (0 if run_started else 1, run_rank, STAGES.index(n.stage))

    return sorted(ready, key=rank)[0]


def is_complete(run_ids: Iterable[str], status: dict[str, str]) -> bool:
    return all(status.get(n.key(), "planned") in TERMINAL_OK for n in all_nodes(run_ids))
