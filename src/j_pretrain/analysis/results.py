"""Aggregate Figure-3a results from the canonical checkpoint inventory.

The inventory (``artifacts/checkpoint_inventory.jsonl``) is the single source of
truth — never wandb. For each lambda condition we read the validation losses stored
on the milestone checkpoints:

* ``L_im``  = MusicPile val loss at Stage-2 ``restored_best`` (theta_post, best-val).
* ``L_ret`` = MusicPile val loss at Stage-3 ``final`` (after ChemPile fine-tuning).
* ``L_ft``  = ChemPile val loss at Stage-3 ``final``.
* ``L_pre`` = C4 val loss at Stage-3 ``final``.

Derived quantities (forgetting, retention improvement, Spearman correlations, linear
trend, monotonicity, best lambda, token accounting) and an outcome classification are
computed with criteria fixed HERE, before any result is seen. Everything regenerates
deterministically from the inventory + run manifest so the figure needs no manual entry.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

CORE_METRICS = ["L_im", "L_ret", "L_ft", "L_pre"]
# eval-key -> which loss it maps to at the Stage-3 final checkpoint
_S3_MAP = {"mp": "L_ret", "chempile": "L_ft", "c4": "L_pre"}


# --------------------------------------------------------------------------- #
# inventory reading
# --------------------------------------------------------------------------- #
def read_creates(inventory_path: str | Path) -> list[dict]:
    """All op=create checkpoint records from an append-only inventory."""
    p = Path(inventory_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("op") == "create":
            out.append(r)
    return out


def _pick(recs: list[dict], run_id: str, stage: str, label: str,
          cls: str = "analysis") -> Optional[dict]:
    """Latest create record matching (run, stage, class, milestone label)."""
    cands = [r for r in recs if r.get("run_id") == run_id and r.get("stage") == stage
             and r.get("checkpoint_class") == cls and label in r.get("milestone_labels", [])]
    if not cands:
        return None
    return max(cands, key=lambda r: (r.get("step", 0), r.get("created_at_utc", "")))


def _metric(rec: Optional[dict], key: str) -> Optional[float]:
    if rec is None:
        return None
    v = (rec.get("metrics_at_creation") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


# --------------------------------------------------------------------------- #
# per-condition extraction
# --------------------------------------------------------------------------- #
def extract_condition(recs: list[dict], run_id: str, lambda_frac: float) -> dict:
    """Build one result row for a lambda condition (values None if not yet produced)."""
    rb = _pick(recs, run_id, "stage2", "restored_best")
    s2_final = _pick(recs, run_id, "stage2", "final")
    s3 = _pick(recs, run_id, "stage3", "final")
    s1 = _pick(recs, run_id, "stage1", "final")

    L_im = _metric(rb, "mp")
    L_ret = _metric(s3, "mp")
    L_ft = _metric(s3, "chempile")
    L_pre = _metric(s3, "c4")
    forgetting = (L_ret - L_im) if (L_ret is not None and L_im is not None) else None

    return {
        "run_id": run_id, "lambda": lambda_frac,
        "L_im": L_im, "L_ret": L_ret, "L_ft": L_ft, "L_pre": L_pre,
        "forgetting": forgetting,
        "stage1_total_tokens": (s1 or {}).get("total_tokens"),
        "stage1_c4_tokens": (s1 or {}).get("c4_tokens"),
        "stage1_mp_tokens": (s1 or {}).get("musicpile_tokens"),
        "stage2_mp_tokens_to_stop": (s2_final or {}).get("musicpile_tokens"),
        "stage2_mp_tokens_to_best": (rb or {}).get("musicpile_tokens"),
        "stage3_chempile_tokens": (s3 or {}).get("chempile_tokens"),
        "complete": all(v is not None for v in (L_im, L_ret, L_ft, L_pre)),
    }


def extract_all(inventory_path: str | Path, run_specs: dict[str, dict]) -> list[dict]:
    """One row per run, sorted by lambda. ``run_specs``: run_id -> {'lambda': ...}."""
    recs = read_creates(inventory_path)
    rows = [extract_condition(recs, rid, float(spec["lambda"]))
            for rid, spec in run_specs.items()]
    rows.sort(key=lambda r: r["lambda"])
    _add_retention_improvement(rows)
    return rows


def _add_retention_improvement(rows: list[dict]) -> None:
    """Absolute & relative L_ret improvement vs lambda=0 (positive = better retention)."""
    base = next((r for r in rows if r["lambda"] == 0.0 and r["L_ret"] is not None), None)
    b = base["L_ret"] if base else None
    for r in rows:
        if b is not None and r["L_ret"] is not None:
            r["L_ret_abs_improvement_vs_l0"] = b - r["L_ret"]
            r["L_ret_rel_improvement_vs_l0"] = (b - r["L_ret"]) / b if b != 0 else None
        else:
            r["L_ret_abs_improvement_vs_l0"] = None
            r["L_ret_rel_improvement_vs_l0"] = None


# --------------------------------------------------------------------------- #
# derived statistics
# --------------------------------------------------------------------------- #
def _spearman(x: list, y: list) -> Optional[dict]:
    xy = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(xy) < 3:
        return None
    from scipy.stats import spearmanr
    rho, p = spearmanr([a for a, _ in xy], [b for _, b in xy])
    return {"rho": float(rho), "p": float(p), "n": len(xy)}


def _linfit(x: list, y: list) -> Optional[dict]:
    xy = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(xy) < 2:
        return None
    slope, intercept = np.polyfit([a for a, _ in xy], [b for _, b in xy], 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "assumptions": "OLS on one seed per lambda; no confidence interval (design has no replicate seeds)"}


def _monotonicity_violations(lams: list, vals: list) -> Optional[int]:
    """# adjacent lambda-sorted pairs where L_ret does NOT decrease (expected: decreasing)."""
    pairs = sorted([(l, v) for l, v in zip(lams, vals) if v is not None])
    if len(pairs) < 2:
        return None
    return sum(1 for (_, a), (_, b) in zip(pairs, pairs[1:]) if b > a)


def derived_stats(rows: list[dict]) -> dict:
    lam = [r["lambda"] for r in rows]
    L_ret = [r["L_ret"] for r in rows]
    L_im = [r["L_im"] for r in rows]
    im_vals = [v for v in L_im if v is not None]
    ret_vals = [(r["lambda"], r["L_ret"]) for r in rows if r["L_ret"] is not None]
    best = min(ret_vals, key=lambda t: t[1]) if ret_vals else (None, None)
    return {
        "spearman_lambda_L_ret": _spearman(lam, L_ret),
        "spearman_lambda_L_im": _spearman(lam, L_im),
        "linear_trend_L_ret_vs_lambda": _linfit(lam, L_ret),
        "L_ret_monotonicity_violations": _monotonicity_violations(lam, L_ret),
        "best_lambda_by_L_ret": best[0],
        "best_L_ret": best[1],
        "L_im_range": (max(im_vals) - min(im_vals)) if im_vals else None,
        "L_im_mean": (sum(im_vals) / len(im_vals)) if im_vals else None,
        "n_complete_conditions": sum(1 for r in rows if r["complete"]),
    }


# --------------------------------------------------------------------------- #
# outcome classification — criteria fixed BEFORE results are seen
# --------------------------------------------------------------------------- #
def classify(rows: list[dict], derived: dict) -> dict:
    """Classify the replication outcome. Paper claim: as lambda increases L_ret
    IMPROVES (decreases) while L_im stays roughly flat. Lower loss == better."""
    if derived["n_complete_conditions"] < len(rows) or derived["spearman_lambda_L_ret"] is None:
        return {"label": "Inconclusive",
                "rationale": "Not all conditions produced finite L_im/L_ret/L_ft/L_pre.",
                "criteria": "requires all locked conditions complete"}
    rho = derived["spearman_lambda_L_ret"]["rho"]
    im_mean = derived["L_im_mean"] or 0.0
    im_flat = (derived["L_im_range"] is not None
               and (im_mean == 0 or derived["L_im_range"] / abs(im_mean) < 0.05))
    crit = ("neg rho = retention improves with lambda; flat = L_im range <5% of mean; "
            "one seed, no CIs")
    if rho <= -0.8 and im_flat:
        label = "Replicated"
    elif rho <= -0.4:
        label = "Directionally replicated but weaker"
    elif rho < 0.2:
        label = "Mixed or pipeline-dependent"
    elif rho < 0.4:
        label = "Inconclusive"
    else:
        label = "Did not replicate"
    return {"label": label,
            "rationale": f"spearman(lambda,L_ret) rho={rho:.3f}; L_im "
                         f"{'flat' if im_flat else 'not flat'} (range={derived['L_im_range']}).",
            "criteria": crit}


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
_CSV_COLUMNS = ["run_id", "lambda", "L_im", "L_ret", "L_ft", "L_pre", "forgetting",
                "L_ret_abs_improvement_vs_l0", "L_ret_rel_improvement_vs_l0",
                "stage1_total_tokens", "stage1_c4_tokens", "stage1_mp_tokens",
                "stage2_mp_tokens_to_stop", "stage2_mp_tokens_to_best",
                "stage3_chempile_tokens", "complete"]


def write_results(rows: list[dict], derived: dict, classification: dict,
                  out_dir: str | Path, extra_meta: Optional[dict] = None) -> dict:
    """Write results.csv + results.json; return the JSON payload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _CSV_COLUMNS})
    payload = {"rows": rows, "derived": derived, "classification": classification,
               "meta": extra_meta or {}}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_results(inventory_path: str | Path, run_specs: dict[str, dict],
                  out_dir: str | Path, extra_meta: Optional[dict] = None) -> dict:
    """End-to-end: inventory -> rows + derived + classification -> results.{csv,json}."""
    rows = extract_all(inventory_path, run_specs)
    derived = derived_stats(rows)
    classification = classify(rows, derived)
    return write_results(rows, derived, classification, out_dir, extra_meta)
