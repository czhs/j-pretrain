"""Typed configuration schemas.

Scientific parameters (architecture, optimizer, schedule, budgets, lambda) are
kept strictly separate from execution parameters (microbatch, grad-accum, workers,
compile) so that the scientific config hash is invariant to hardware autotuning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Frozen SmolLM2-135M architecture (scientific). See configs/model/smollm2-135m.json."""

    name: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_act: str
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    attention_dropout: float
    initializer_range: float
    bos_token_id: int
    eos_token_id: int
    train_seq_len: int
    tokenizer: str
    torch_dtype: str = "bfloat16"
    head_dim: Optional[int] = None
    rope_scaling: Optional[dict] = None
    pad_token_id: Optional[int] = None
    expected_param_count: Optional[int] = None
    provenance: str = "INFERRED_FROM_SMOLLM2"

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(**kwargs)

    def to_hf_kwargs(self) -> dict[str, Any]:
        """Kwargs for transformers.LlamaConfig (architecture only, no dropout tuning)."""
        return dict(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=self.tie_word_embeddings,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def scientific_dict(self) -> dict[str, Any]:
        """The subset that defines the architecture for hashing (excludes comments/expected)."""
        d = self.to_hf_kwargs()
        d.update(name=self.name, train_seq_len=self.train_seq_len,
                 tokenizer=self.tokenizer, torch_dtype=self.torch_dtype)
        return d


@dataclass(frozen=True)
class ExecConfig:
    """Execution / hardware parameters. NOT part of the scientific hash."""

    microbatch_size: int = 8
    grad_accum_steps: int = 1
    attn_implementation: str = "sdpa"
    torch_compile: bool = True
    compile_mode: str = "default"
    dataloader_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    gradient_checkpointing: bool = False
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class RunSpec:
    """Scientific identity of one lambda pipeline (Stage1->2->3)."""

    experiment: str  # e.g. "music"
    subset_tokens: int  # e.g. 300_000_000
    lambda_frac: float  # in {0, 0.25, 0.5, 0.75, 1.0}
    seed: int
