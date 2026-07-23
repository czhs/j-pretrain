"""Behavioural tests for the completion verifier (scripts/verify_completion.py).

The verifier IS the completion gate, so it needs its own guards: it must fail on
unmet mandatory criteria, it must resolve the append-only inventories from the
committed repo tree (not the artifact root, which holds only weights/datasets),
and it must keep covering every mandatory category (final-audit criterion 28
forbids weakening or stubbing it).

``main()`` is never invoked here: one of its own checks shells out to pytest, so
calling it from inside pytest would recurse. The individual ``check_*`` functions
and helpers are exercised instead.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_completion", REPO / "scripts" / "verify_completion.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module via sys.modules; register first
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vc():
    return _load_verifier()


def test_inventories_resolve_to_committed_repo_tree(vc, monkeypatch):
    """Inventories live in repo/artifacts; the artifact root holds only big data.

    Regression guard: reading them from J_PRETRAIN_ARTIFACT_ROOT made every
    inventory check fail, which would have made completion unreachable.
    """
    monkeypatch.setenv("J_PRETRAIN_ARTIFACT_ROOT", "/nonexistent/artifact/root")
    assert vc._inventory_root() == REPO / "artifacts"
    assert vc._inventory_root() != vc._artifact_root()
    # the real checkpoint inventory is found there, and it is non-empty
    inv = vc._load_jsonl(vc._inventory_root() / "checkpoint_inventory.jsonl")
    assert [x for x in inv if x.get("op") == "create"]


def test_inventory_checks_read_repo_tree_not_artifact_root(vc, monkeypatch):
    monkeypatch.setenv("J_PRETRAIN_ARTIFACT_ROOT", "/nonexistent/artifact/root")
    r = vc.Report()
    vc.check_checkpoints(r)
    named = {c.name: c for c in r.checks}
    assert named["inventory_exists"].ok


def test_report_fails_when_a_mandatory_check_is_unmet(vc):
    r = vc.Report()
    r.add("ok_check", "cat", True)
    assert r.failed == []
    r.add("bad_check", "cat", False, "unmet")
    assert [c.name for c in r.failed] == ["bad_check"]
    assert r.to_json()["passed"] is False
    assert r.to_json()["n_failed_mandatory"] == 1


def test_optional_checks_do_not_gate_completion(vc):
    r = vc.Report()
    r.add("advisory", "cat", False, mandatory=False)
    assert r.failed == []
    assert r.to_json()["passed"] is True


def test_verifier_covers_every_mandatory_category(vc):
    """Criterion 28: the gate must not be stubbed down to a few easy checks."""
    r = vc.Report()
    for fn in (vc.check_scope, vc.check_config, vc.check_dataset, vc.check_checkpoints,
               vc.check_results, vc.check_docs, vc.check_artifacts_preservation):
        fn(r)
    cats = {c.category for c in r.checks}
    assert {"scope", "config", "dataset", "checkpoints",
            "results", "docs", "preservation"} <= cats
    # every check registered by these categories is mandatory (no silent downgrade)
    assert all(c.mandatory for c in r.checks)
    assert len(r.checks) >= 30


def test_scope_check_requires_all_five_lambdas(vc):
    assert vc.LAMBDAS == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert vc.STAGES == ("stage1", "stage2", "stage3")


def test_incomplete_experiment_currently_fails(vc):
    """While runs are in flight the gate must report INCOMPLETE, never pass."""
    r = vc.Report()
    vc.check_results(r)
    vc.check_docs(r)
    assert r.failed, "verifier passed results/docs before any results exist"


def test_report_json_is_serialisable(vc):
    r = vc.Report()
    r.add("x", "cat", True, "detail")
    json.dumps(r.to_json())
