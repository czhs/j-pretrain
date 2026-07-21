#!/usr/bin/env python
"""Standalone throughput/VRAM benchmark for the 135M SmolLM2-style model.

Fail-fast feasibility probe: measures training tokens/sec and peak VRAM on the
RTX 4090 using the ACTUAL architecture (random init — we must NOT use pretrained
weights) and the training sequence length (1024). Results feed the feasibility
gate in reports/FEASIBILITY.md. No real data, no checkpoints; pure compute probe.
"""
import argparse, json, time, sys
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# SmolLM2-135M config (verified against HuggingFaceTB/SmolLM2-135M/config.json)
CFG = dict(
    vocab_size=49152, hidden_size=576, intermediate_size=1536,
    num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
    hidden_act="silu", max_position_embeddings=8192, rope_theta=100000.0,
    rms_norm_eps=1e-5, tie_word_embeddings=True, attention_bias=False,
    mlp_bias=False, initializer_range=0.041666666666666664,
    bos_token_id=0, eos_token_id=0, attention_dropout=0.0,
)


def build_model():
    cfg = LlamaConfig(**CFG, attn_implementation="sdpa")
    model = LlamaForCausalLM(cfg)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    # tied embeddings: lm_head shares embedding weight; count unique storages
    seen = {}
    uniq = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen[p.data_ptr()] = True
            uniq += p.numel()
    return total, uniq


def bench(microbatch, seq_len, steps, warmup, compile_mode):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model().to(dev)
    model.gradient_checkpointing_disable()
    model.train()
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    vocab = CFG["vocab_size"]
    torch.cuda.reset_peak_memory_stats(dev)
    torch.manual_seed(0)

    def one_step():
        ids = torch.randint(0, vocab, (microbatch, seq_len), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
            loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize(dev)
    t0 = time.time()
    last = None
    for _ in range(steps):
        last = one_step()
    torch.cuda.synchronize(dev)
    dt = time.time() - t0
    toks = microbatch * seq_len * steps
    tok_s = toks / dt
    peak_alloc = torch.cuda.max_memory_allocated(dev) / 1e9
    peak_resv = torch.cuda.max_memory_reserved(dev) / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return dict(microbatch=microbatch, seq_len=seq_len, steps=steps,
                tok_s=round(tok_s, 1), s_per_step=round(dt / steps, 4),
                peak_alloc_gb=round(peak_alloc, 2), peak_reserved_gb=round(peak_resv, 2),
                last_loss=round(last, 3), compile=compile_mode or "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--microbatches", type=int, nargs="+", default=[16, 24, 32, 48, 64])
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--compile", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    m = build_model()
    total, uniq = count_params(m)
    del m
    info = dict(gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                total_params=total, unique_params_tied=uniq,
                config=CFG)
    print(json.dumps({"model_info": info}, indent=2))
    print(f"PARAM_COUNT total={total/1e6:.2f}M unique(tied)={uniq/1e6:.2f}M", file=sys.stderr)

    results = []
    for mb in args.microbatches:
        try:
            r = bench(mb, args.seq_len, args.steps, args.warmup, args.compile or None)
            results.append(r)
            print("OK ", json.dumps(r))
        except RuntimeError as e:
            msg = str(e)[:200]
            e.__traceback__ = None  # drop frame refs holding model/opt alive
            e = None
            results.append(dict(microbatch=mb, error=msg))
            print(f"OOM/ERR microbatch={mb}: {msg}")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    out = dict(model_info=info, results=results)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
