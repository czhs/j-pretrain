"""Analysis pipeline: inventory -> results.csv/json + derived stats + Figure 3a."""
from __future__ import annotations

import csv
import json

from j_pretrain.analysis.figure import make_figure
from j_pretrain.analysis.results import build_results, classify, derived_stats, extract_all

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RUN_SPECS = {f"music-300m_lambda-{l}": {"lambda": l} for l in LAMBDAS}


def _rec(run_id, stage, cls, labels, metrics, step, **tok):
    return {"op": "create", "run_id": run_id, "stage": stage, "checkpoint_class": cls,
            "milestone_labels": labels, "metrics_at_creation": metrics, "step": step,
            "total_tokens": tok.get("total", 0), "c4_tokens": tok.get("c4", 0),
            "musicpile_tokens": tok.get("mp", 0), "chempile_tokens": tok.get("chempile", 0),
            "created_at_utc": "2026-07-21T00:00:00Z"}


def _write_inventory(path, l_im_flat=True, l_ret_decreasing=True):
    lines = []
    for i, l in enumerate(LAMBDAS):
        rid = f"music-300m_lambda-{l}"
        L_im = 3.00 + (0.001 * i if l_im_flat else 0.5 * i)      # ~flat vs steep
        L_ret = (3.50 - 0.20 * i) if l_ret_decreasing else (3.50 + 0.20 * i)
        lines.append(_rec(rid, "stage1", "analysis", ["final"], {}, 100,
                          total=8_700_000_000 + int(l * 300_000_000),
                          c4=8_700_000_000, mp=int(l * 300_000_000)))
        lines.append(_rec(rid, "stage2", "analysis", ["restored_best"], {"mp": L_im}, 40, mp=1_500_000_000))
        lines.append(_rec(rid, "stage2", "analysis", ["final"], {"mp": L_im + 0.01}, 60, mp=2_000_000_000))
        lines.append(_rec(rid, "stage3", "analysis", ["final"],
                          {"mp": L_ret, "chempile": 2.0 - 0.05 * i, "c4": 4.0 + 0.02 * i}, 400,
                          chempile=200_000_000))
    path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in lines) + "\n")


def test_extract_and_classify_replicated(tmp_path):
    inv = tmp_path / "checkpoint_inventory.jsonl"
    _write_inventory(inv, l_im_flat=True, l_ret_decreasing=True)
    rows = extract_all(inv, RUN_SPECS)
    assert [r["lambda"] for r in rows] == LAMBDAS
    assert all(r["complete"] for r in rows)
    # L_ret decreases with lambda; forgetting = L_ret - L_im
    assert rows[0]["L_ret"] > rows[-1]["L_ret"]
    for r in rows:
        assert abs(r["forgetting"] - (r["L_ret"] - r["L_im"])) < 1e-9
    # retention improvement vs lambda 0 is positive and increasing
    assert rows[-1]["L_ret_abs_improvement_vs_l0"] > 0
    d = derived_stats(rows)
    assert d["spearman_lambda_L_ret"]["rho"] <= -0.9  # strong negative
    assert d["best_lambda_by_L_ret"] == 1.0
    assert d["L_ret_monotonicity_violations"] == 0
    cls = classify(rows, d)
    assert cls["label"] == "Replicated", cls


def test_did_not_replicate_when_retention_worsens(tmp_path):
    inv = tmp_path / "checkpoint_inventory.jsonl"
    _write_inventory(inv, l_im_flat=True, l_ret_decreasing=False)  # L_ret increases w/ lambda
    rows = extract_all(inv, RUN_SPECS)
    d = derived_stats(rows)
    assert classify(rows, d)["label"] == "Did not replicate"


def test_incomplete_is_inconclusive(tmp_path):
    inv = tmp_path / "checkpoint_inventory.jsonl"
    # only 2 conditions have any records
    lines = [_rec("music-300m_lambda-0.0", "stage3", "analysis", ["final"],
                  {"mp": 3.5, "chempile": 2.0, "c4": 4.0}, 400),
             _rec("music-300m_lambda-0.0", "stage2", "analysis", ["restored_best"], {"mp": 3.0}, 40)]
    inv.write_text("\n".join(json.dumps(x, sort_keys=True) for x in lines) + "\n")
    rows = extract_all(inv, RUN_SPECS)
    d = derived_stats(rows)
    assert classify(rows, d)["label"] == "Inconclusive"
    assert d["n_complete_conditions"] == 1


def test_build_results_and_figure(tmp_path):
    inv = tmp_path / "checkpoint_inventory.jsonl"
    _write_inventory(inv)
    out = tmp_path / "results"
    payload = build_results(inv, RUN_SPECS, out)
    # results.csv: one row per condition, core metrics present
    rows = list(csv.DictReader((out / "results.csv").open()))
    assert sorted(float(r["lambda"]) for r in rows) == LAMBDAS
    for r in rows:
        for k in ("L_im", "L_ret", "L_ft", "L_pre"):
            assert r[k] not in (None, ""), (k, r)
    assert (out / "results.json").exists()
    assert payload["classification"]["label"] == "Replicated"
    # figure regenerates from the csv
    figs = make_figure(out / "results.csv", tmp_path / "figures")
    assert all(p.exists() and p.stat().st_size > 0 for p in figs)
    assert {p.suffix for p in figs} == {".png", ".pdf"}
