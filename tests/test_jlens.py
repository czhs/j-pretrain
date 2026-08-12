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
    nnls_projected_gradient,
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


def _coherent_dictionary(n: int, d: int, spread: float = 0.05,
                         seed: int = 0) -> torch.Tensor:
    """Atoms clustered tightly around a few directions -> large lambda_max(A A^T).

    Mimics the real J-lens dictionary, which packs n_vocab (49152) vectors into d_model
    (576) dimensions and is therefore extremely coherent.
    """
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(3, d, generator=g)
    base = anchors[torch.randint(0, 3, (n,), generator=g)]
    D = base + spread * torch.randn(n, d, generator=g)
    return D / D.norm(dim=1, keepdim=True)


def _explained(x: torch.Tensor, D: torch.Tensor, idx, coef) -> float:
    recon = (D[idx] * coef[:, None]).sum(0) if len(idx) else torch.zeros_like(x)
    return 1.0 - float((x - recon).pow(2).sum() / x.pow(2).sum())


def _coherent_active_set(m: int = 6, d: int = D_MODEL) -> tuple[torch.Tensor, torch.Tensor]:
    """m nearly-identical unit atoms (lambda_max ~ m) plus a target in their span."""
    g = torch.Generator().manual_seed(0)
    base = torch.randn(d, generator=g)
    base = base / base.norm()
    A = base.repeat(m, 1) + 1e-3 * torch.randn(m, d, generator=g)
    A = A / A.norm(dim=1, keepdim=True)
    return A, base * 2.0


def test_nnls_handles_coherent_active_set():
    """Regression for NOTEBOOK I-11: step must be 1/lambda_max(A A^T), not 1/max(diag)."""
    A, x = _coherent_active_set()
    assert float(torch.linalg.eigvalsh(A @ A.t()).max()) > 2.0, (
        "test is only meaningful if lambda_max > 2, where a step of 1.0 diverges")
    coef = nnls_projected_gradient(A, x, n_steps=200)
    assert torch.isfinite(coef).all(), "coefficients diverged"
    assert (coef >= 0).all()
    resid = float((A.t() @ coef - x).norm())
    assert resid <= float(x.norm()), "fit worse than c=0"
    assert resid < 0.1 * float(x.norm()), "should fit a target inside the atoms' span"


def test_nnls_never_worse_than_zero_across_random_coherent_cases():
    """The invariant the analysis actually depends on, enforced unconditionally.

    NOTE on provenance: the real bug (NOTEBOOK I-11) was found on a real checkpoint, not by
    a unit test, and could NOT be reproduced synthetically — with nonnegativity clamping the
    bad step size oscillates (0 -> 2 -> 0) and lands exactly at ||x|| rather than beyond it.
    The real-data failure needed the wrong step *and* coefficients warm-started across
    pursuit iterations. Rather than pretend to a regression test that does not regress, the
    solver now *guarantees* the invariant (falling back to c = 0), and this test pins that
    guarantee over many random coherent problems.
    """
    for trial in range(25):
        g = torch.Generator().manual_seed(trial)
        m = int(torch.randint(2, 12, (1,), generator=g))
        base = torch.randn(D_MODEL, generator=g)
        A = base.repeat(m, 1) + float(torch.rand(1, generator=g)) * torch.randn(
            m, D_MODEL, generator=g)
        A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)
        x = torch.randn(D_MODEL, generator=g) * float(torch.rand(1, generator=g) * 5)
        coef = nnls_projected_gradient(A, x, n_steps=100)
        assert torch.isfinite(coef).all() and (coef >= 0).all()
        assert float((A.t() @ coef - x).norm()) <= float(x.norm()) + 1e-5, (
            f"trial {trial}: fit worse than c=0")


def test_gradient_pursuit_explained_fraction_is_monotone_in_k():
    """More atoms may not help much, but must never make the fit worse."""
    torch.manual_seed(3)
    D = _coherent_dictionary(60, D_MODEL, seed=7)
    x = torch.randn(D_MODEL)
    prev = -1.0
    for k in (1, 3, 5, 10, 20):
        idx, coef = gradient_pursuit(x, D, k=k)
        got = _explained(x, D, idx, coef)
        assert got >= prev - 1e-3, f"explained fraction dropped at k={k}: {got} < {prev}"
        prev = got


def test_monotone_in_k_on_ill_conditioned_overcomplete_dictionary():
    """Regression for NOTEBOOK I-12, at a scale that resembles the real dictionary.

    The real J-lens dictionary is 49152 x 576 and extremely coherent. At large k the
    active set becomes ill-conditioned, lambda_max grows, the 1/lambda_max step shrinks,
    and a fixed iteration budget silently under-converges — so explained variance can
    *drop* as atoms are added, which is impossible for a converged solve. The small
    16-dim coherent case above did not expose this; this one does.
    """
    torch.manual_seed(11)
    d, n = 64, 4000                       # heavily overcomplete, like n_vocab >> d_model
    D = torch.randn(n, d)
    D = D / D.norm(dim=1, keepdim=True)
    for trial in range(3):
        x = torch.randn(d, generator=torch.Generator().manual_seed(trial))
        prev = -1.0
        for k in (1, 5, 10, 25, 50):
            idx, coef = gradient_pursuit(x, D, k=k)
            got = _explained(x, D, idx, coef)
            assert got >= prev - 1e-6, (
                f"trial {trial}: explained variance dropped at k={k}: {got:.4f} < {prev:.4f}")
            prev = got


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
