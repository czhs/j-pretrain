#!/usr/bin/env python
"""Deterministic completion verifier for the Figure-3a reproduction.

Exits 0 ONLY when every mandatory criterion in the mission document is satisfied;
nonzero otherwise. Writes ``reports/completion_verification.json`` (machine-readable)
and prints a concise human summary. This script IS the completion gate — it must
faithfully implement the mission's "Completion verifier" checklist and must never be
weakened or stubbed (the final audit checks this).

It reads only the canonical local record (state files, append-only inventories,
results tables, docs) — never wandb. Run: ``python scripts/verify_completion.py``.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "state"
DOCS = REPO / "docs"
REPORTS = REPO / "reports"
RESULTS = REPO / "results"
RUNS = REPO / "runs"

EXPECTED_REMOTE = "git@github.com:czhs/j-pretrain.git"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
STAGES = ("stage1", "stage2", "stage3")


@dataclass
class Check:
    name: str
    category: str
    ok: bool
    detail: str = ""
    mandatory: bool = True


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name, category, ok, detail="", mandatory=True):
        self.checks.append(Check(name, category, bool(ok), str(detail), mandatory))

    @property
    def failed(self):
        return [c for c in self.checks if c.mandatory and not c.ok]

    def to_json(self):
        return {
            "passed": len(self.failed) == 0,
            "n_checks": len(self.checks),
            "n_failed_mandatory": len(self.failed),
            "checks": [vars(c) for c in self.checks],
        }


def _load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _artifact_root() -> Path:
    env = os.environ.get("J_PRETRAIN_ARTIFACT_ROOT")
    if env:
        return Path(env)
    st = _load_json(STATE / "experiment_state.json") or {}
    return Path(st.get("artifact_root", str(REPO / "artifacts")))


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
def check_scope(r: Report):
    lock = _load_json(STATE / "SCOPE_LOCK.json")
    r.add("scope_lock_exists", "scope", lock is not None and lock.get("locked") is True,
          "state/SCOPE_LOCK.json present and locked")
    st = _load_json(STATE / "experiment_state.json") or {}
    runs = st.get("runs", {})
    expected = {f"music-300m_lambda-{('%g' % l) + ('.0' if '.' not in ('%g' % l) else '')}"
                for l in LAMBDAS}
    r.add("five_runs_present", "scope", set(runs.keys()) == expected,
          f"runs={sorted(runs.keys())}")
    complete = [rid for rid, v in runs.items()
                if all(v.get(s) == "complete" for s in STAGES)]
    r.add("all_locked_runs_complete", "scope", len(complete) == 5,
          f"{len(complete)}/5 runs have all 3 stages complete")
    manifest = _load_jsonl(RUNS / "manifest.jsonl")
    r.add("run_manifest_covers_locked", "scope",
          len({m.get("run_id") for m in manifest} & expected) == 5 if manifest else False,
          f"manifest run_ids cover 5 locked runs (have {len({m.get('run_id') for m in manifest})})")


def check_config(r: Report):
    st = _load_json(STATE / "experiment_state.json") or {}
    r.add("frozen_paper_spec", "config", (DOCS / "PAPER_SPEC.md").exists())
    for name, p in [("model", "configs/model/smollm2-135m.json"),
                    ("stage1", "configs/stage1/music.json"),
                    ("stage2", "configs/stage2/music.json"),
                    ("stage3", "configs/stage3/chempile.json"),
                    ("data", "configs/data/datasets.json")]:
        r.add(f"frozen_config_{name}", "config", (REPO / p).exists(), p)
    r.add("spec_hash_recorded", "config", bool(st.get("spec_hash")))
    r.add("scope_hash_recorded", "config", bool(st.get("scope_hash")))
    inv = _load_jsonl(_artifact_root() / "checkpoint_inventory.jsonl")
    creates = [x for x in inv if x.get("op") == "create"]
    have_hashes = all(c.get("config_hash") and c.get("git_commit") for c in creates)
    r.add("ckpts_record_config_and_git", "config", have_hashes if creates else False,
          f"{len(creates)} checkpoint records" if creates else "no checkpoints yet")


def check_dataset(r: Report):
    st = _load_json(STATE / "experiment_state.json") or {}
    dh = st.get("dataset_manifest_hashes", {})
    r.add("dataset_revisions_recorded", "dataset",
          all(dh.get(k) for k in ("c4_revision", "musicpile_revision", "chempile_revision")),
          f"{ {k: dh.get(k) for k in ('c4_revision','musicpile_revision','chempile_revision')} }")
    root = _artifact_root() / "datasets"
    manifests = list(root.rglob("manifest.json")) if root.exists() else []
    r.add("tokenized_manifests_exist", "dataset", len(manifests) >= 6,
          f"{len(manifests)} (source,split) manifests under {root}")
    # 300M subset exact token count
    mp_train = root / "musicpile" / "train" / "manifest.json"
    m = _load_json(mp_train)
    if m:
        n_win = m.get("n_seqs", 0)
        r.add("musicpile_pool_covers_300M", "dataset", n_win >= 292968,
              f"MusicPile train packed windows={n_win} (need >=292968 for 300M subset)")
    else:
        r.add("musicpile_pool_covers_300M", "dataset", False, "no MusicPile train manifest yet")
    probe = _artifact_root() / "probes" / "probe_manifest.json"
    r.add("frozen_probe_manifest", "dataset", probe.exists(), str(probe))


def check_checkpoints(r: Report):
    root = _artifact_root()
    inv = _load_jsonl(root / "checkpoint_inventory.jsonl")
    creates = [x for x in inv if x.get("op") == "create"]
    r.add("inventory_exists", "checkpoints", (root / "checkpoint_inventory.jsonl").exists())
    # every recorded checkpoint has checksums + is present + load-validated
    all_present = bool(creates)
    for c in creates:
        d = root / c.get("rel_path", "")
        if not (d.exists() and c.get("checksums") and c.get("load_validation_status") == "verified"):
            all_present = False
            break
    r.add("all_ckpts_present_checksummed_validated", "checkpoints", all_present,
          f"{len(creates)} checkpoint records" if creates else "no checkpoints yet")
    # permanent milestone presence is validated per-run once runs complete (audit);
    # here require at least the shared init snapshot once any run started.
    labels = {tuple(c.get("milestone_labels", [])) for c in creates}
    r.add("init_snapshot_when_started", "checkpoints",
          (("init",) in labels or ("incoming",) in labels) if creates else False,
          "init/incoming snapshot recorded" if creates else "no checkpoints yet")


def check_results(r: Report):
    csv_path = RESULTS / "results.csv"
    ok = csv_path.exists()
    rows = []
    if ok:
        try:
            rows = list(csv.DictReader(csv_path.open()))
        except Exception:
            ok = False
    r.add("results_csv_exists", "results", ok, str(csv_path))
    if rows:
        lambdas = [float(x.get("lambda", "nan")) for x in rows if x.get("lambda") not in (None, "")]
        r.add("results_one_row_per_condition", "results",
              sorted(lambdas) == LAMBDAS, f"lambdas in csv={sorted(lambdas)}")
        needed = {"L_im", "L_ret", "L_ft", "L_pre"}
        r.add("results_have_core_metrics", "results",
              all(needed.issubset(x.keys()) for x in rows),
              f"columns={list(rows[0].keys()) if rows else []}")
    else:
        r.add("results_one_row_per_condition", "results", False, "no result rows yet")
        r.add("results_have_core_metrics", "results", False, "no result rows yet")
    for fig in ("figure3a_replication.png", "figure3a_replication.pdf"):
        r.add(f"figure_{fig.split('.')[-1]}", "results",
              (REPO / "figures" / fig).exists(), f"figures/{fig}")


def check_tests_audits(r: Report):
    # unit + integration tests (includes deterministic resume test)
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           cwd=str(REPO), capture_output=True, text=True, timeout=1200)
        r.add("tests_pass", "tests_audits", p.returncode == 0,
              (p.stdout.strip().splitlines() or [""])[-1])
    except Exception as e:
        r.add("tests_pass", "tests_audits", False, f"pytest failed to run: {e}")
    r.add("data_audit_exists", "tests_audits", (DOCS / "DATA_AUDIT.md").exists())
    r.add("pretrain_readiness_audit_exists", "tests_audits",
          (REPORTS / "PRETRAIN_READINESS_AUDIT.md").exists())
    r.add("final_audit_exists", "tests_audits", (REPORTS / "AUDIT.md").exists())
    audit = REPORTS / "AUDIT.md"
    if audit.exists():
        txt = audit.read_text().lower()
        # crude gate: audit must assert no unresolved critical/major findings
        r.add("final_audit_no_unresolved_critical", "tests_audits",
              ("no unresolved critical" in txt or "0 critical" in txt or
               "no critical or major" in txt),
              "AUDIT.md must state no unresolved critical/major findings")
    else:
        r.add("final_audit_no_unresolved_critical", "tests_audits", False, "no AUDIT.md yet")


def check_docs(r: Report):
    required = [DOCS / f for f in ("PAPER_SPEC.md", "REFERENCES.md", "COMPUTE_PLAN.md",
                                   "STORAGE_PLAN.md", "DATA_AUDIT.md", "FAILURE_PLAYBOOK.md",
                                   "FAILURES.md", "HANDOFF.md")]
    required += [REPORTS / f for f in ("FINAL_REPORT.md", "REPRODUCIBILITY.md",
                                       "COMPUTE_ACCOUNTING.md", "STORAGE_ACCOUNTING.md",
                                       "AUDIT.md")]
    for p in required:
        r.add(f"doc_{p.parent.name}_{p.name}", "docs", p.exists(), str(p.relative_to(REPO)))


def check_artifacts_preservation(r: Report):
    root = _artifact_root()
    r.add("checkpoint_inventory_validates", "preservation",
          (root / "checkpoint_inventory.jsonl").exists())
    r.add("run_artifact_inventory", "preservation",
          (root / "run_artifact_inventory.jsonl").exists())
    r.add("storage_usage_history", "preservation",
          (root / "storage_usage.jsonl").exists())
    r.add("backup_status_recorded", "preservation",
          (root / "backup_status.jsonl").exists())


def check_git(r: Report):
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    r.add("branch_main", "git", branch == "main", f"branch={branch}")
    remote = _git("remote", "get-url", "origin")
    r.add("remote_correct", "git", remote == EXPECTED_REMOTE, f"origin={remote}")
    porcelain = _git("status", "--porcelain")
    dirty = [l for l in porcelain.splitlines()
             if l and not l.endswith((".log",)) and "wandb/" not in l]
    r.add("working_tree_clean", "git", len(dirty) == 0,
          f"{len(dirty)} tracked-file changes (excl logs/wandb)")
    local = _git("rev-parse", "HEAD")
    remote_head = _git("rev-parse", "origin/main")
    r.add("head_pushed_to_origin", "git", bool(local) and local == remote_head,
          f"local={local[:8]} origin/main={remote_head[:8]}")


def main() -> int:
    r = Report()
    check_scope(r)
    check_config(r)
    check_dataset(r)
    check_checkpoints(r)
    check_results(r)
    check_tests_audits(r)
    check_docs(r)
    check_artifacts_preservation(r)
    check_git(r)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "completion_verification.json"
    out.write_text(json.dumps(r.to_json(), indent=2, sort_keys=True))

    passed = len(r.failed) == 0
    by_cat: dict[str, list[Check]] = {}
    for c in r.checks:
        by_cat.setdefault(c.category, []).append(c)
    print("=" * 68)
    print("COMPLETION VERIFICATION —", "PASS" if passed else "INCOMPLETE")
    print("=" * 68)
    for cat, checks in by_cat.items():
        n_ok = sum(1 for c in checks if c.ok)
        print(f"\n[{cat}] {n_ok}/{len(checks)} ok")
        for c in checks:
            if not c.ok:
                mark = "✗" if c.mandatory else "○"
                print(f"  {mark} {c.name}: {c.detail}")
    print(f"\n{len(r.failed)} mandatory checks unmet. Full report: {out.relative_to(REPO)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
