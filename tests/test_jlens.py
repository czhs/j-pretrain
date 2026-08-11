"""Tests for the J-lens / J-space instrument.

The load-bearing test is :func:`test_batched_jlens_matches_bruteforce`: the fast path uses
``is_grads_batched`` (vmap) plus a causality argument (one backward seeded at every output
position yields the whole t' >= t sum for every source position at once). That is easy to
get subtly wrong, so it is checked against a naive one-backward-per-output-dim reference.
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from j_pretrain.analysis.jlens import (
    compute_jlens,
    gradient_pursuit,
    jlens_batch,
    jlens_batch_reference,
    jlens_dictionary,
    principal_angles,
    subspace_overlap,
)

D_MODEL = 16
N_LAYERS = 3
VOCAB = 32


def _tiny_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=32,
        num_hidden_layers=N_LAYERS, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    cfg._attn_implementation = "eager"
    m = LlamaForCausalLM(cfg).float()
    m.eval()
    return m


def _ids(b: int = 2, t: int = 6, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (b, t), generator=g)


# --------------------------------------------------------------------------- #
# correctness of the fast path
# --------------------------------------------------------------------------- #
def test_batched_jlens_matches_bruteforce():
    m, ids = _tiny_model(), _ids()
    for layer in range(N_LAYERS - 1):
        fast = jlens_batch(m, ids, layers=[layer], seed_chunk=5)[layer]
        ref = jlens_batch_reference(m, ids, layer=layer)
        assert torch.allclose(fast, ref, atol=1e-4, rtol=1e-3), (
            f"layer {layer}: max abs diff {(fast - ref).abs().max():.3e}")


def test_seed_chunk_size_does_not_change_result():
    m, ids = _tiny_model(), _ids()
    a = jlens_batch(m, ids, layers=[0], seed_chunk=1)[0]
    b = jlens_batch(m, ids, layers=[0], seed_chunk=D_MODEL)[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_all_layers_in_one_pass_matches_per_layer_calls():
    """One backward populates every layer's gradient; that shortcut must be exact."""
    m, ids = _tiny_model(), _ids()
    together = jlens_batch(m, ids, layers=[0, 1], seed_chunk=4)
    for layer in (0, 1):
        alone = jlens_batch(m, ids, layers=[layer], seed_chunk=4)[layer]
        assert torch.allclose(together[layer], alone, atol=1e-5)


def test_compute_jlens_averages_over_batches():
    m = _tiny_model()
    b1, b2 = _ids(seed=1), _ids(seed=2)
    got = compute_jlens(m, [b1, b2], layers=[0], seed_chunk=8)[0]
    n = b1.numel() + b2.numel()
    want = (jlens_batch(m, b1, layers=[0])[0] + jlens_batch(m, b2, layers=[0])[0]) / n
    assert torch.allclose(got, want, atol=1e-5)


def test_model_left_in_eval_mode_and_grads_not_polluted():
    m, ids = _tiny_model(), _ids()
    m.eval()
    compute_jlens(m, [ids], layers=[0], seed_chunk=8)
    assert not m.training
    assert all(p.grad is None for p in m.parameters()), "J-lens must not accumulate .grad"


def test_future_weighting_changes_result():
    """mean_over_future divides each source position by its own count of future tokens."""
    m, ids = _tiny_model(), _ids()
    mean = jlens_batch(m, ids, layers=[0], mean_over_future=True)[0]
    summed = jlens_batch(m, ids, layers=[0], mean_over_future=False)[0]
    assert not torch.allclose(mean, summed)


# --------------------------------------------------------------------------- #
# dictionary + J-space primitives
# --------------------------------------------------------------------------- #
def test_dictionary_shape_and_normalization():
    m, ids = _tiny_model(), _ids()
    J = jlens_batch(m, ids, layers=[0])[0]
    D = jlens_dictionary(J, m.lm_head.weight.detach())
    assert D.shape == (VOCAB, D_MODEL)
    assert torch.allclose(D.norm(dim=1), torch.ones(VOCAB), atol=1e-5)


def test_gradient_pursuit_recovers_sparse_nonnegative_combination():
    torch.manual_seed(0)
    D = torch.randn(40, D_MODEL)
    D = D / D.norm(dim=1, keepdim=True)
    true_idx = torch.tensor([3, 17, 29])
    true_coef = torch.tensor([1.5, 0.8, 2.2])
    x = (D[true_idx] * true_coef[:, None]).sum(0)
    idx, coef = gradient_pursuit(x, D, k=5)
    assert set(true_idx.tolist()).issubset(set(idx.tolist()))
    assert (coef >= 0).all(), "J-space coefficients must be nonnegative"
    recon = (D[idx] * coef[:, None]).sum(0)
    assert torch.allclose(recon, x, atol=1e-2)


def test_gradient_pursuit_respects_sparsity_budget():
    torch.manual_seed(1)
    D = torch.randn(50, D_MODEL)
    D = D / D.norm(dim=1, keepdim=True)
    idx, coef = gradient_pursuit(torch.randn(D_MODEL), D, k=7)
    assert len(idx) <= 7 and len(coef) == len(idx)


# --------------------------------------------------------------------------- #
# subspace comparison (M1 primitive)
# --------------------------------------------------------------------------- #
def test_subspace_overlap_identical_and_orthogonal():
    torch.manual_seed(0)
    A = torch.randn(D_MODEL, 4)
    assert subspace_overlap(A, A) > 0.999
    # build an explicitly orthogonal complement pair
    Q, _ = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL))
    assert subspace_overlap(Q[:, :4], Q[:, 4:8]) < 1e-4


def test_principal_angles_are_sorted_and_bounded():
    torch.manual_seed(2)
    ang = principal_angles(torch.randn(D_MODEL, 5), torch.randn(D_MODEL, 5))
    assert torch.all(ang >= -1e-6) and torch.all(ang <= torch.pi / 2 + 1e-6)
    assert torch.all(ang[1:] >= ang[:-1] - 1e-6)
