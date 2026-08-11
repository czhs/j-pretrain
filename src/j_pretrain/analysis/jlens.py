"""J-lens: the averaged first-order causal map from a layer to the model's output.

Following the workspace/J-lens construction (transformer-circuits.pub/2026/workspace):

    J_l = E_{t, t' >= t, prompt} [ d h_final,t' / d h_l,t ]          (d_model x d_model)
    lens(h_l) = softmax( W_U . norm( J_l h_l ) )

``h_l`` is the residual stream after decoder layer ``l`` and ``h_final`` is the residual
stream after the last decoder layer (pre-final-norm, since the lens applies ``norm``
itself). The rows of ``J_l`` are indexed by the *output* dimension, so ``J_l @ h`` maps a
layer-l activation into final-layer coordinates.

The **J-lens dictionary** at layer l is ``D_l = W_U @ J_l`` (n_vocab x d_model): one vector
per vocabulary item, the layer-l direction that most promotes that token at the output. It
is overcomplete (49152 >> 576), and the *J-space* is the set of points expressible as a
sparse nonnegative combination of its columns.

Why this is the right instrument for this project
-------------------------------------------------
The early-exposure paper's Section 5 explains its own phenomenon geometrically: mixing
creates *specialized features* carrying the post-training capability, which survive
fine-tuning precisely because the fine-tuning distribution has **zero covariance** along
them, so downstream gradients cannot reach them (Thm 5.3). That is a claim about which
output-relevant directions a capability occupies. The J-lens measures exactly that: a
causally-grounded, output-referred basis for a layer's representation.

Cost
----
The expensive part is the Jacobian. Two structural facts make it cheap:

1. Seeding the backward pass with basis vector ``e_j`` at *every* output position yields,
   by causality, ``sum_{t' >= t} d h_final,t'[j] / d h_l,t`` for **every** source position
   t simultaneously -- one backward per output dimension, not one per (t, t') pair.
2. A single backward populates gradients for **all** layers at once, so the per-layer cost
   is shared.

Total: ``d_model`` backward passes per prompt batch for the full set of layers. With
``is_grads_batched`` these are vmapped in chunks.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch


# --------------------------------------------------------------------------- #
# residual-stream capture
# --------------------------------------------------------------------------- #
@contextmanager
def _captured_residuals(model, layer_idxs: Iterable[int]):
    """Forward-hook the given decoder layers and yield a dict {idx: hidden_state}."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(i: int):
        def hook(_mod, _args, out):
            captured[i] = out[0] if isinstance(out, tuple) else out
        return hook

    for i in sorted(set(layer_idxs)):
        handles.append(model.model.layers[i].register_forward_hook(_mk(i)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _future_weights(T: int, device, mean_over_future: bool) -> torch.Tensor:
    """w[t] normalizing the causal sum over t' >= t into a mean (T - t terms)."""
    if not mean_over_future:
        return torch.ones(T, device=device, dtype=torch.float32)
    return 1.0 / torch.arange(T, 0, -1, device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# J-lens
# --------------------------------------------------------------------------- #
def jlens_batch(
    model,
    input_ids: torch.Tensor,
    layers: Optional[Iterable[int]] = None,
    mean_over_future: bool = True,
    seed_chunk: int = 16,
    out_dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """Un-normalized J_l accumulator for ONE batch: {layer: [d, d]} summed over (b, t).

    Returns sums, not means, so callers can accumulate across batches and divide by the
    total (b, t) count once. Use :func:`compute_jlens` for the batched driver.
    """
    n_layers = len(model.model.layers)
    final_idx = n_layers - 1
    layers = sorted(set(layers)) if layers is not None else list(range(final_idx))
    if any(l < 0 or l > final_idx for l in layers):
        raise ValueError(f"layer index out of range for {n_layers}-layer model: {layers}")
    d = model.config.hidden_size

    with _captured_residuals(model, list(layers) + [final_idx]) as cap:
        model.model(input_ids=input_ids)
        h_final = cap[final_idx]
        B, T, _ = h_final.shape
        srcs = [cap[i] for i in layers]
        w = _future_weights(T, h_final.device, mean_over_future)

        acc = {i: torch.zeros(d, d, dtype=out_dtype, device=h_final.device) for i in layers}
        for start in range(0, d, seed_chunk):
            js = torch.arange(start, min(start + seed_chunk, d), device=h_final.device)
            seed = torch.zeros(len(js), B, T, d, device=h_final.device, dtype=h_final.dtype)
            seed[torch.arange(len(js)), :, :, js] = 1.0
            grads = torch.autograd.grad(
                outputs=h_final, inputs=srcs, grad_outputs=seed,
                retain_graph=True, is_grads_batched=True)
            for i, g in zip(layers, grads):
                # g: [len(js), B, T, d] -> weight by 1/(T-t), sum over batch+position
                acc[i][start:start + len(js)] += (
                    g.to(out_dtype) * w[None, None, :, None]).sum(dim=(1, 2))
    return acc


def compute_jlens(
    model,
    batches: Iterable[torch.Tensor],
    layers: Optional[Iterable[int]] = None,
    mean_over_future: bool = True,
    seed_chunk: int = 16,
    out_dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """J_l averaged over a corpus. ``batches`` yields int64 ``input_ids`` [B, T]."""
    was_training = model.training
    model.eval()
    total: dict[int, torch.Tensor] = {}
    n = 0
    try:
        for input_ids in batches:
            acc = jlens_batch(model, input_ids, layers=layers,
                              mean_over_future=mean_over_future,
                              seed_chunk=seed_chunk, out_dtype=out_dtype)
            for i, v in acc.items():
                total[i] = v if i not in total else total[i] + v
            n += input_ids.shape[0] * input_ids.shape[1]
    finally:
        if was_training:
            model.train()
    if n == 0:
        raise ValueError("no batches supplied")
    return {i: v / n for i, v in total.items()}


def jlens_batch_reference(
    model,
    input_ids: torch.Tensor,
    layer: int,
    mean_over_future: bool = True,
) -> torch.Tensor:
    """Brute-force single-layer J_l accumulator: one plain backward per output dim.

    Deliberately naive and slow. Exists so the batched/vmapped :func:`jlens_batch` is
    *verified* rather than assumed — see ``tests/test_jlens.py``.
    """
    n_layers = len(model.model.layers)
    final_idx = n_layers - 1
    d = model.config.hidden_size
    with _captured_residuals(model, [layer, final_idx]) as cap:
        model.model(input_ids=input_ids)
        h_final, h_src = cap[final_idx], cap[layer]
        B, T, _ = h_final.shape
        w = _future_weights(T, h_final.device, mean_over_future)
        out = torch.zeros(d, d, dtype=torch.float32, device=h_final.device)
        for j in range(d):
            seed = torch.zeros_like(h_final)
            seed[:, :, j] = 1.0
            (g,) = torch.autograd.grad(h_final, [h_src], grad_outputs=seed,
                                       retain_graph=True)
            out[j] = (g.float() * w[None, :, None]).sum(dim=(0, 1))
    return out


# --------------------------------------------------------------------------- #
# dictionary + J-space
# --------------------------------------------------------------------------- #
def jlens_dictionary(J: torch.Tensor, unembed: torch.Tensor,
                     normalize: bool = True) -> torch.Tensor:
    """``D = W_U @ J`` -> [n_vocab, d_model]; row v is layer-l's "promote token v" direction.

    Rows are L2-normalized by default so downstream sparse coding sees a proper dictionary
    and coefficients are comparable across tokens.
    """
    D = unembed.to(J.dtype) @ J
    if normalize:
        D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return D


def gradient_pursuit(x: torch.Tensor, D: torch.Tensor, k: int = 25,
                     n_steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse **nonnegative** approximation of ``x`` by <= k rows of ``D``.

    Matching-pursuit style: repeatedly take the dictionary row with the largest positive
    correlation with the residual, then refit the active set with a few projected-gradient
    steps (nonnegativity by clamping). Returns ``(indices [k], coeffs [k])``.

    Nonnegativity matters: the J-space is defined as a sparse *nonnegative* combination,
    so "this concept is active" is meaningful and cancellation between opposite-signed
    dictionary atoms is disallowed.
    """
    x = x.float()
    D = D.float()
    residual = x.clone()
    idx: list[int] = []
    coef = torch.zeros(0, device=x.device)
    for _ in range(k):
        corr = D @ residual
        if idx:
            corr[torch.tensor(idx, device=x.device)] = -float("inf")
        best = int(torch.argmax(corr))
        if corr[best] <= 0:
            break
        idx.append(best)
        A = D[torch.tensor(idx, device=x.device)]          # [m, d]
        coef = torch.cat([coef, torch.zeros(1, device=x.device)])
        for _ in range(max(1, n_steps) * 8):               # projected gradient, nonneg
            grad = A @ (A.t() @ coef - x)
            step = 1.0 / (A @ A.t()).diagonal().clamp_min(1e-8).max()
            coef = (coef - step * grad).clamp_min(0.0)
        residual = x - A.t() @ coef
    if not idx:
        return (torch.zeros(0, dtype=torch.long, device=x.device),
                torch.zeros(0, device=x.device))
    return torch.tensor(idx, dtype=torch.long, device=x.device), coef


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians, ascending) between the column spaces of A and B.

    The subspace-overlap primitive behind measurement M1: how much of the MusicPile
    J-space is shared with the C4 / ChemPile J-space.
    """
    Qa, _ = torch.linalg.qr(A.float())
    Qb, _ = torch.linalg.qr(B.float())
    s = torch.linalg.svdvals(Qa.t() @ Qb).clamp(-1.0, 1.0)
    return torch.arccos(s)


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean cos^2 of the principal angles: 1.0 = identical subspaces, 0.0 = orthogonal."""
    return float(torch.cos(principal_angles(A, B)).pow(2).mean())
