"""
Part 1 — memory budget for streamed DPO, computed BEFORE training.

Usage:
    python memory_budget.py                       # Qwen2.5-1.5B, config values below
    python memory_budget.py --model Qwen/Qwen2.5-7B-Instruct --bytes-per-weight 0.5625

Everything is arithmetic on the model config + our measured token lengths; the
numbers are compared against torch.cuda.max_memory_allocated / reserved and
nvidia-smi after the run (see the notebook, section "Compare").

Assumptions (each one is a place where the measured peak can differ):
- Layer streaming keeps `stream_buffers` decoder layers in VRAM, never the
  whole stack. Embedding + final norm + (tied) LM head stay resident.
- DPO concatenates chosen and rejected: 2 x batch_size rows per forward.
- TRL pads each batch to its longest row, so `seq` is the longest pair in a
  batch, not max_length. We report both the typical and the worst batch.
- The reference pass reuses the same streamed base with the adapter swapped to
  the frozen `ref` copy, under no_grad; its logits are freed before the policy
  forward, so it adds max(), not sum(), to the peak.
- fp16 on T4 (no bf16). LoRA weights are upcast to fp32 for the optimizer
  (soup's align_trainable_dtype_for_fp16), Adam keeps 2 fp32 moments.
- Activations: every streamed layer is wrapped in checkpoint(), so only each
  layer's input hidden state is saved; one layer is recomputed at a time in
  backward.
- Logits: lower bound = fp16 logits + fp16 grad (4 B/elem). Upper bound = soup's
  own measured loss-path figure (12 B/elem in backward) + the 2 B source tensor.
"""

import argparse
import json

GB = 1e9

# Qwen2.5-1.5B-Instruct config.json (used if transformers is not available)
QWEN25_15B = dict(hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
                  num_attention_heads=12, num_key_value_heads=2, vocab_size=151936,
                  tie_word_embeddings=True)


def load_cfg(model_id: str) -> dict:
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model_id)
        return dict(hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
                    num_hidden_layers=c.num_hidden_layers,
                    num_attention_heads=c.num_attention_heads,
                    num_key_value_heads=c.num_key_value_heads, vocab_size=c.vocab_size,
                    tie_word_embeddings=getattr(c, "tie_word_embeddings", False))
    except Exception:
        return QWEN25_15B


def budget(cfg: dict, *, batch: int, seq: int, r: int, buffers: int,
           bytes_per_weight: float, cuda_context_gb: float) -> dict:
    h, i, L, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    head_dim = h // cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"] * head_dim
    rows = 2 * batch                                    # chosen + rejected concatenated

    # --- frozen weights ------------------------------------------------------
    attn = h * h + 2 * h * kv + h * h                   # q, k, v, o
    mlp = 3 * h * i                                     # gate, up, down
    layer_params = attn + mlp
    layer_bytes = layer_params * bytes_per_weight
    embed_bytes = V * h * 2                             # fp16 embedding, resident
    head_bytes = 0 if cfg["tie_word_embeddings"] else V * h * 2

    # --- LoRA on q,k,v,o,gate,up,down ------------------------------------------
    lora_layer = r * ((h + h) + (h + kv) * 2 + (h + h) + (h + i) * 3)
    lora_params = lora_layer * L
    lora_train = lora_params * (4 + 4 + 8)              # fp32 weight + grad + Adam m,v
    lora_ref = lora_params * 4                          # frozen `ref` adapter copy (TRL 0.29)

    # --- activations (checkpointed per layer) ---------------------------------
    saved_inputs = rows * seq * h * 2 * L               # one fp16 hidden state per layer
    recompute = rows * seq * (4 * h + 3 * i) * 2        # one layer's internals in backward

    # --- logits ----------------------------------------------------------------
    elems = rows * seq * V
    logits_low = elems * 4                              # fp16 logits + fp16 grad
    logits_high = elems * (2 + 12)                      # source + soup's loss-path figure

    base = {
        "streamed_layer_buffers": buffers * layer_bytes,
        "embedding_(+tied_head)": embed_bytes + head_bytes,
        "lora_params+grads+adam": lora_train,
        "lora_ref_copy": lora_ref,
        "activations_saved": saved_inputs,
        "activations_recompute": recompute,
    }
    subtotal = sum(base.values())
    return {
        "shape": {"batch": batch, "rows": rows, "seq": seq, "vocab": V, "layers": L},
        "params": {"per_layer": layer_params, "lora_total": lora_params,
                   "base_total_approx": layer_params * L + V * h},
        "items_GB": {k: round(v / GB, 3) for k, v in base.items()},
        "logits_GB": {"low_4B": round(logits_low / GB, 3), "high_14B": round(logits_high / GB, 3)},
        "predicted_allocated_GB": {"low": round((subtotal + logits_low) / GB, 2),
                                   "high": round((subtotal + logits_high) / GB, 2)},
        "predicted_nvidia_smi_GB": {"low": round((subtotal + logits_low) / GB + cuda_context_gb, 2),
                                    "high": round((subtotal + logits_high) / GB + cuda_context_gb, 2)},
        "resident_equivalent_weights_GB": round((layer_bytes * L + embed_bytes + head_bytes) / GB, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-typical", type=int, default=200, help="typical padded batch length")
    ap.add_argument("--seq-worst", type=int, default=261, help="longest prompt+answer in the data")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--buffers", type=int, default=2)
    ap.add_argument("--bytes-per-weight", type=float, default=2.0, help="2.0 fp16, ~0.5625 NF4")
    ap.add_argument("--cuda-context-gb", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_cfg(a.model)
    res = {s: budget(cfg, batch=a.batch, seq=seq, r=a.r, buffers=a.buffers,
                     bytes_per_weight=a.bytes_per_weight, cuda_context_gb=a.cuda_context_gb)
           for s, seq in (("typical", a.seq_typical), ("worst", a.seq_worst))}
    res["model"] = a.model
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
