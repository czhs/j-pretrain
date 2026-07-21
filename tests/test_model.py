"""Tests for the frozen model config and the 135M model builder."""
import torch

from j_pretrain.config.schemas import ModelConfig
from j_pretrain.models.build import (
    DEFAULT_MODEL_CONFIG,
    build_model,
    count_parameters,
    load_model_config,
)


def test_model_config_loads():
    mc = load_model_config()
    assert mc.name == "smollm2-135m"
    assert mc.vocab_size == 49152
    assert mc.num_hidden_layers == 30
    assert mc.hidden_size == 576
    assert mc.num_attention_heads == 9
    assert mc.num_key_value_heads == 3  # GQA
    assert mc.intermediate_size == 1536
    assert mc.rope_theta == 100000.0
    assert mc.rms_norm_eps == 1e-5
    assert mc.tie_word_embeddings is True
    assert mc.attention_bias is False
    assert mc.train_seq_len == 1024


def test_param_count_matches_expected():
    mc = load_model_config()
    model = build_model(mc, seed=0)
    total, unique = count_parameters(model)
    # ~135M target; assert within 1M and exact-match the frozen expectation.
    assert 133_000_000 <= total <= 136_000_000
    if mc.expected_param_count is not None:
        assert total == mc.expected_param_count, (total, mc.expected_param_count)


def test_embeddings_are_tied():
    model = build_model(seed=0)
    emb = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    assert emb.data_ptr() == lm_head.data_ptr()


def test_deterministic_init_same_seed():
    m1 = build_model(seed=123)
    m2 = build_model(seed=123)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1


def test_different_seed_different_init():
    m1 = build_model(seed=1)
    m2 = build_model(seed=2)
    diffs = [not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters())]
    assert any(diffs)


def test_forward_shape_and_loss():
    mc = load_model_config()
    model = build_model(mc, seed=0)
    ids = torch.randint(0, mc.vocab_size, (2, 32))
    out = model(input_ids=ids, labels=ids)
    assert out.logits.shape == (2, 32, mc.vocab_size)
    assert torch.isfinite(out.loss)
