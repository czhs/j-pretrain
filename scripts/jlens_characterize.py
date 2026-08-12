#!/usr/bin/env python
"""Characterize the J-lens of one checkpoint: establish this model's own workspace scale.

Run BEFORE the M1-M5 measurements. The workspace paper's phenomenology (k ~ 25 concurrent
concepts, onset near layer 38, ~10% of activation variance) comes from a much larger model
and must not be assumed for a 30-layer, d=576 SmolLM2-135M. This script re-derives the
equivalents empirically so the later measurements pick k and the layer set from evidence
rather than from a borrowed constant.

Reports, per layer:
  * ||J_l||_F and the singular-value spectrum of J_l
  * effective rank (participation ratio of the squared singular values)
  * fraction of on-distribution activation variance the top-k J-space atoms reconstruct,
    per domain (c4 / musicpile / chempile)
  * active-k saturation: reconstruction gain vs sparsity budget

Usage:
  PYTHONPATH=src python scripts/jlens_characterize.py \
      --checkpoint <dir> --artifact-root <root> [--layers 4,9,14,19,24] [--out out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from j_pretrain.analysis.jlens import (
    compute_jlens,
    jlens_dictionary,
    pursuit_residual_path,
)
from j_pretrain.artifacts import checkpoint as ck
from j_pretrain.data.shards import PackedDataset
from j_pretrain.models.build import build_model, load_model_config

DOMAINS = ("c4", "musicpile", "chempile")


def probe_batches(root: Path, corpus: str, n_windows: int, seq_len: int,
                  batch_size: int, device: str):
    """Yield [B, T] int64 batches from a corpus's fixed probe set."""
    ds = PackedDataset(root / "probes" / corpus)
    n = min(n_windows, len(ds))
    for s in range(0, n, batch_size):
        rows = [torch.from_numpy(ds[i][:seq_len].astype("int64"))
                for i in range(s, min(s + batch_size, n))]
        yield torch.stack(rows).to(device)


def collect_activations(model, batches, layers, max_positions: int = 4096):
    """Sample residual-stream vectors per layer: {layer: [N, d]}."""
    from j_pretrain.analysis.jlens import _captured_residuals
    out = {l: [] for l in layers}
    with torch.no_grad():
        for ids in batches:
            with _captured_residuals(model, layers) as cap:
                model.model(input_ids=ids)
                for l in layers:
                    h = cap[l].reshape(-1, cap[l].shape[-1]).float()
                    out[l].append(h[torch.randperm(h.shape[0])[:512]].cpu())
    return {l: torch.cat(v)[:max_positions] for l, v in out.items()}


def effective_rank(sv: torch.Tensor) -> float:
    """Participation ratio of squared singular values: (sum s^2)^2 / sum s^4."""
    p = sv.double().pow(2)
    return float(p.sum().pow(2) / p.pow(2).sum().clamp_min(1e-30))


def reconstruction_curve(acts: torch.Tensor, D: torch.Tensor, k_grid: list,
                         n_samples: int = 32) -> dict:
    """Explained variance at each k in ``k_grid``, from ONE pursuit pass per sample.

    Aggregated as 1 - sum_i||x_i - recon_i||^2 / sum_i||x_i||^2. Monotone in k because
    :func:`pursuit_residual_path` returns a non-increasing best-residual path.
    """
    idxs = torch.randperm(acts.shape[0])[:n_samples]
    k_max = max(k_grid)
    num = torch.zeros(k_max, device=acts.device)
    den = 0.0
    for i in idxs:
        x = acts[i]
        num += pursuit_residual_path(x, D, k_max=k_max).pow(2)
        den += float(x.pow(2).sum())
    return {str(k): 1.0 - float(num[k - 1]) / max(den, 1e-30) for k in k_grid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--layers", default=None, help="comma list; default 5 evenly spaced")
    ap.add_argument("--n-lens-windows", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed-chunk", type=int, default=32)
    ap.add_argument("--k-grid", default="1,5,10,25,50")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.artifact_root)

    mc = load_model_config()
    model = build_model(mc).to(device)
    model.load_state_dict(ck.load_weights(Path(args.checkpoint)))
    model.eval()

    n_layers = model.config.num_hidden_layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = [int(round(f * (n_layers - 2))) for f in (0.15, 0.35, 0.5, 0.7, 0.9)]
    layers = sorted(set(layers))
    print(f"[jlens] device={device} layers={layers} of {n_layers}")

    # --- the lens itself: computed on the pretraining-like distribution (C4) ---
    J = compute_jlens(
        model,
        probe_batches(root, "c4", args.n_lens_windows, args.seq_len, args.batch_size, device),
        layers=layers, seed_chunk=args.seed_chunk)
    print("[jlens] J_l computed")

    W_U = model.lm_head.weight.detach()
    k_grid = [int(x) for x in args.k_grid.split(",")]
    report: dict = {"checkpoint": str(args.checkpoint), "layers": layers,
                    "n_layers": n_layers, "d_model": model.config.hidden_size,
                    "seq_len": args.seq_len, "n_lens_windows": args.n_lens_windows,
                    "per_layer": {}}

    acts = {d: collect_activations(
                model,
                probe_batches(root, d, 32, args.seq_len, args.batch_size, device),
                layers)
            for d in DOMAINS}

    for l in layers:
        sv = torch.linalg.svdvals(J[l].float())
        D = jlens_dictionary(J[l], W_U).to(device)
        entry = {
            "frobenius_norm": float(J[l].float().norm()),
            "effective_rank": effective_rank(sv),
            "singular_values_top10": [float(x) for x in sv[:10]],
            "singular_value_decay_50pct_at": int((sv.cumsum(0) / sv.sum() < 0.5).sum()),
            "reconstruction": {},
        }
        for dom in DOMAINS:
            a = acts[dom][l].to(device)
            entry["reconstruction"][dom] = reconstruction_curve(a, D, k_grid)
        report["per_layer"][str(l)] = entry
        print(f"  layer {l:>2}: |J|_F={entry['frobenius_norm']:.3f} "
              f"eff_rank={entry['effective_rank']:.1f} "
              f"recon@k={k_grid[-1]} " +
              " ".join(f"{d}={entry['reconstruction'][d][str(k_grid[-1])]:.3f}"
                       for d in DOMAINS))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"[jlens] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
