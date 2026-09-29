"""
Part 2 — did the DPO run actually change the model in the intended direction?

A falling DPO loss is not evidence: the loss is computed against a reference
that the same code builds, on the training rows, and can fall with a broken
reference, a skipped optimizer, or by learning surface cues. This script works
only from ARTIFACTS ON DISK (base model id + saved adapter + held-out data),
in a fresh process, and asks five questions:

  1. LOADED   every tensor in adapter_model.safetensors is present in the model,
              byte-identical, with no missing / unexpected LoRA keys
  2. MOVED    lora_B is no longer zero (PEFT initialises it to 0); size of the
              effective update  dW = B @ A * scaling  relative to W, per module
  3. ACTIVE   adapter on vs off gives different logits on the same input
  4. LEARNED  on the 100 HELD-OUT pairs, the DPO implicit margin
                d = [logp(c) - logp(r)]_policy - [logp(c) - logp(r)]_reference
              is positive (bootstrap CI, exact sign test), sliced by failure
              mode / category / prompt template seen-vs-unseen in training
  5. CONTROL  the same adapter with each lora_B replaced by a random matrix of
              the SAME norm gives d ~ 0; if it does as well, "learning" is just
              "weights changed"

It also measures a length shortcut (accuracy on pairs where the rejected
answer is LONGER) and saves greedy generations with the adapter on/off.

Usage (T4):
  python verify_training.py --adapter adapters/<run> --out logs/verify_<run>.json
Self-test (CPU, small model, fake adapters):
  python verify_training.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

LIMITATIONS = [
    "Does not show the answers are GOOD Russian support replies: margins measure preference on "
    "our synthetic pairs, not free-form quality. Generations are saved for manual reading.",
    "Cannot tell a real preference from a surface cue shared by train and eval (template "
    "wording, keywords, length). The length and unseen-template slices reduce, not remove, this.",
    "Held-out pairs come from the same generator as training pairs, so a positive margin is an "
    "in-distribution result, not evidence on real tickets.",
    "Runs the model resident (not streamed). Streamed-vs-resident equality is covered by the "
    "parity test (forward only), not by this script.",
    "Says nothing about general capability loss; that is soup ship's forgetting leg.",
]


# ----------------------------------------------------------------------------- model loading

def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base(base: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device).eval()
    return tok, model


def attach_adapter(model, adapter_dir: str) -> tuple:
    """Load the adapter and verify, independently of PEFT's own warnings, that
    every saved tensor landed in the model unchanged."""
    from peft import PeftModel
    from safetensors.torch import load_file

    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False).eval()
    saved = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    params = dict(peft_model.named_parameters())
    matched, mismatched, missing = 0, [], []
    for key, tensor in saved.items():
        # saved: base_model.model.X.lora_A.weight  -> live: base_model.model.X.lora_A.default.weight
        live_key = key.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default.")
        live = params.get(live_key)
        if live is None:
            missing.append(key)
        elif torch.equal(live.detach().cpu().to(tensor.dtype), tensor):
            matched += 1
        else:
            mismatched.append(key)
    live_lora = [n for n in params if "lora_" in n]
    loaded = {
        "saved_tensors": len(saved),
        "matched_byte_identical": matched,
        "missing_in_model": missing[:5], "n_missing": len(missing),
        "value_mismatch": mismatched[:5], "n_mismatch": len(mismatched),
        "live_lora_params": len(live_lora),
        "ok": matched == len(saved) == len(live_lora) and not missing and not mismatched,
    }
    return peft_model, saved, loaded


def weight_movement(peft_model, saved: dict) -> dict:
    """Is lora_B non-zero, and how big is dW = B A * scaling relative to W?"""
    cfg = peft_model.peft_config["default"]
    scaling = cfg.lora_alpha / cfg.r
    b_keys = [k for k in saved if "lora_B" in k]
    nonzero_b = sum(bool(saved[k].abs().max() > 0) for k in b_keys)
    per_module = defaultdict(list)
    params = dict(peft_model.named_parameters())
    for kb in b_keys:
        ka = kb.replace("lora_B", "lora_A")
        B, A = saved[kb].float(), saved[ka].float()
        dW = (B @ A) * scaling
        base_key = kb.replace(".lora_B.weight", ".base_layer.weight")
        W = params.get(base_key)
        rel = dW.norm().item() / W.float().norm().item() if W is not None else float("nan")
        module = kb.split(".")[-3]           # q_proj / k_proj / ...
        per_module[module].append(rel)
    return {
        "lora_B_tensors": len(b_keys), "lora_B_nonzero": nonzero_b,
        "all_lora_B_zero": nonzero_b == 0,
        "rel_update_norm_by_module": {m: {"mean": sum(v) / len(v), "max": max(v)} for m, v in per_module.items()},
        "rel_update_norm_overall_mean": sum(sum(v) for v in per_module.values()) / max(1, sum(len(v) for v in per_module.values())),
    }


# ----------------------------------------------------------------------------- scoring

def encode_pair(tok, prompt_msgs, answer_msgs, device):
    p_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    f_text = tok.apply_chat_template(prompt_msgs + answer_msgs, tokenize=False)
    p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(f_text, add_special_tokens=False)["input_ids"]
    prefix_ok = f_ids[: len(p_ids)] == p_ids
    return torch.tensor([f_ids], device=device), len(p_ids), prefix_ok


@torch.no_grad()
def response_logp(model, ids, n_prompt) -> tuple:
    logits = model(input_ids=ids).logits[:, :-1].float()
    targets = ids[:, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    resp = lp[:, n_prompt - 1:]
    return resp.sum().item(), resp.shape[1]


def score_pairs(peft_model, tok, rows, device, *, with_adapter: bool) -> list:
    out = []
    for r in rows:
        rec = {}
        for side in ("chosen", "rejected"):
            ids, n_p, ok = encode_pair(tok, r["prompt"], r[side], device)
            if with_adapter:
                lp, n = response_logp(peft_model, ids, n_p)
            else:
                with peft_model.disable_adapter():
                    lp, n = response_logp(peft_model, ids, n_p)
            rec[side] = lp
            rec[f"{side}_tokens"] = n
            rec["prefix_ok"] = rec.get("prefix_ok", True) and ok
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- statistics

def bootstrap_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def sign_test_p(k, n):
    """One-sided exact binomial P(X >= k | n, 0.5)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def summarize(deltas, label_wins=None):
    n = len(deltas)
    wins = sum(d > 0 for d in deltas)
    lo, hi = bootstrap_ci(deltas) if n > 1 else (float("nan"), float("nan"))
    return {"n": n, "mean_delta": sum(deltas) / n, "ci95": [lo, hi],
            "frac_delta_positive": wins / n, "sign_test_p": sign_test_p(wins, n)}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for rank, i in enumerate(order):
        r[i] = float(rank)
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)); vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


# ----------------------------------------------------------------------------- generations

@torch.no_grad()
def generate_side_by_side(peft_model, tok, rows, device, n, max_new_tokens=160):
    out = []
    for r in rows[:n]:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        on = peft_model.generate(**ids, **kw)[0, ids["input_ids"].shape[1]:]
        with peft_model.disable_adapter():
            off = peft_model.generate(**ids, **kw)[0, ids["input_ids"].shape[1]:]
        out.append({"ticket": r["prompt"][-1]["content"],
                    "adapter_on": tok.decode(on, skip_special_tokens=True),
                    "adapter_off": tok.decode(off, skip_special_tokens=True)})
    return out


# ----------------------------------------------------------------------------- main check

def verify(base, adapter, eval_path, eval_meta, train_meta, beta, limit, n_generate, device, out):
    t0 = time.time()
    device = pick_device(device)
    rows = [json.loads(l) for l in open(eval_path, encoding="utf-8")][: limit or None]
    meta = [json.loads(l) for l in open(eval_meta, encoding="utf-8")][: len(rows)] if eval_meta else [{}] * len(rows)
    seen = set()
    if train_meta and Path(train_meta).exists():
        seen = {(m["category"], m["prompt_template"]) for m in map(json.loads, open(train_meta, encoding="utf-8"))}

    tok, model = load_base(base, device)
    peft_model, saved, loaded = attach_adapter(model, adapter)
    moved = weight_movement(peft_model, saved)

    # 3. ACTIVE: logits with adapter on vs off on a real eval input
    ids, _, _ = encode_pair(tok, rows[0]["prompt"], rows[0]["chosen"], device)
    with torch.no_grad():
        on = peft_model(input_ids=ids).logits.float()
        with peft_model.disable_adapter():
            off = peft_model(input_ids=ids).logits.float()
    kl = torch.nn.functional.kl_div(torch.log_softmax(on, -1), torch.log_softmax(off, -1),
                                    log_target=True, reduction="batchmean").item() / ids.shape[1]
    active = {"max_abs_logit_diff": (on - off).abs().max().item(), "mean_token_KL_off_vs_on": kl,
              "identical": bool(torch.equal(on, off))}

    # 4. LEARNED on held-out pairs
    pol = score_pairs(peft_model, tok, rows, device, with_adapter=True)
    ref = score_pairs(peft_model, tok, rows, device, with_adapter=False)
    deltas, deltas_tok, ref_pref, per_pair = [], [], [], []
    for i, (p, q) in enumerate(zip(pol, ref)):
        d = (p["chosen"] - p["rejected"]) - (q["chosen"] - q["rejected"])
        deltas.append(d)
        # per-token version: summed log-probs penalise the longer answer for any
        # perturbation, so the summed margin is length-confounded
        nc, nr = p["chosen_tokens"], p["rejected_tokens"]
        deltas_tok.append((p["chosen"] / nc - p["rejected"] / nr) - (q["chosen"] / nc - q["rejected"] / nr))
        ref_pref.append(q["chosen"] - q["rejected"])
        m = meta[i] if i < len(meta) else {}
        per_pair.append({
            "delta": d, "beta_margin": beta * d,
            "len_diff_tokens": p["chosen_tokens"] - p["rejected_tokens"],
            "failure_mode": m.get("failure_mode"), "category": m.get("category"),
            "template_seen_in_train": (m.get("category"), m.get("prompt_template")) in seen if seen else None,
            "prefix_ok": p["prefix_ok"],
        })

    # 5. CONTROL: same norms, random directions for lora_B
    params = dict(peft_model.named_parameters())
    b_names = [n for n in params if "lora_B" in n and ".default." in n]
    originals = {n: params[n].detach().clone() for n in b_names}
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n in b_names:
            rnd = torch.randn(params[n].shape, generator=gen).to(params[n].device, torch.float32)
            norm = originals[n].float().norm()
            params[n].copy_((rnd * (norm / rnd.norm().clamp_min(1e-12))).to(params[n].dtype))
    ctrl = score_pairs(peft_model, tok, rows, device, with_adapter=True)
    with torch.no_grad():
        for n in b_names:
            params[n].copy_(originals[n])
    ctrl_deltas = [(c["chosen"] - c["rejected"]) - (q["chosen"] - q["rejected"]) for c, q in zip(ctrl, ref)]

    # slices
    def slice_by(key):
        groups = defaultdict(list)
        for pp in per_pair:
            groups[str(pp[key])].append(pp["delta"])
        return {k: {"n": len(v), "mean_delta": sum(v) / len(v), "frac_positive": sum(x > 0 for x in v) / len(v)}
                for k, v in sorted(groups.items())}

    rej_longer = [pp["delta"] for pp in per_pair if pp["len_diff_tokens"] < 0]
    learned = summarize(deltas)
    learned_tok = summarize(deltas_tok)
    control = summarize(ctrl_deltas)
    ctrl_tok = [(c["chosen"] / p["chosen_tokens"] - c["rejected"] / p["rejected_tokens"])
                - (q["chosen"] / p["chosen_tokens"] - q["rejected"] / p["rejected_tokens"])
                for c, q, p in zip(ctrl, ref, pol)]
    control_tok = summarize(ctrl_tok)
    reference_prior = {"frac_reference_already_prefers_chosen": sum(x > 0 for x in ref_pref) / len(ref_pref)}
    length = {
        "n_pairs_rejected_longer": len(rej_longer),
        "frac_delta_positive_when_rejected_longer": (sum(x > 0 for x in rej_longer) / len(rej_longer)) if rej_longer else None,
        "spearman_delta_vs_len_diff": spearman(deltas, [pp["len_diff_tokens"] for pp in per_pair]),
    }

    # verdict
    reasons = []
    if not loaded["ok"]:
        verdict = "NOT_LOADED"; reasons.append("saved adapter tensors did not all land in the model")
    elif moved["all_lora_B_zero"] or active["identical"]:
        verdict = "NO_OP"; reasons.append("lora_B all zero or adapter does not change logits")
    elif learned["ci95"][0] <= 0 or learned["sign_test_p"] > 0.01:
        verdict = "CHANGED_NOT_LEARNED"; reasons.append("held-out margin not reliably > 0")
    elif learned["mean_delta"] <= control["ci95"][1]:
        verdict = "CHANGED_NOT_LEARNED"; reasons.append("trained margin not above random-direction control")
    else:
        verdict = "LEARNED"
    warnings = []
    if verdict in ("NO_OP", "NOT_LOADED"):
        pass
    elif learned_tok["ci95"][0] <= 0 < learned["ci95"][0]:
        warnings.append("summed margin > 0 but per-token margin is not -> gain may come from answer length")
    if verdict not in ("NO_OP", "NOT_LOADED") and length["frac_delta_positive_when_rejected_longer"] is not None and length["frac_delta_positive_when_rejected_longer"] < 0.5:
        warnings.append("margin mostly negative when the rejected answer is longer -> length shortcut likely")
    if not all(pp["prefix_ok"] for pp in per_pair):
        warnings.append("chat template prompt is not a prefix of prompt+answer for some rows; response masks may be off")

    report = {
        "verdict": verdict, "reasons": reasons, "warnings": warnings,
        "inputs": {"base": base, "adapter": adapter, "eval": eval_path, "n_pairs": len(rows), "beta": beta,
                   "device": device, "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "1_loaded": loaded, "2_moved": moved, "3_active": active,
        "4_learned_heldout": learned, "4_learned_heldout_per_token": learned_tok,
        "4_reference_prior": reference_prior,
        "4_by_failure_mode": slice_by("failure_mode"), "4_by_category": slice_by("category"),
        "4_by_template_seen_in_train": slice_by("template_seen_in_train"),
        "4_length_shortcut": length,
        "5_random_direction_control": control, "5_random_direction_control_per_token": control_tok,
        "limitations_what_this_does_not_detect": LIMITATIONS,
        "runtime_s": round(time.time() - t0, 1),
    }
    if n_generate:
        report["generations"] = generate_side_by_side(peft_model, tok, rows, device, n_generate)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(out).with_suffix(".pairs.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in per_pair), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "reasons", "warnings", "3_active",
                                             "4_learned_heldout", "4_learned_heldout_per_token",
                                             "5_random_direction_control", "5_random_direction_control_per_token",
                                             "4_length_shortcut")}, ensure_ascii=False, indent=2))
    return report


# ----------------------------------------------------------------------------- self-test

def make_test_adapter(base: str, out_dir: str, kind: str) -> None:
    """kind='zero'  : PEFT default init (lora_B = 0)  -> must be reported NO_OP
       kind='random': random lora_B                   -> must NOT be reported LEARNED"""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type=TaskType.CAUSAL_LM,
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "gate_proj", "up_proj", "down_proj"]))
    if kind == "random":
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    m.save_pretrained(out_dir)


def self_test(base: str, limit: int) -> None:
    results = {}
    for kind, allowed in (("zero", {"NO_OP"}), ("random", {"CHANGED_NOT_LEARNED"})):
        d = f"selftest_adapter_{kind}"
        make_test_adapter(base, d, kind)
        rep = verify(base, d, "data/dpo_eval.jsonl", "data/eval_meta.jsonl", "data/train_meta.jsonl",
                     0.1, limit, 0, "auto", f"logs/selftest_{kind}.json")
        results[kind] = (rep["verdict"], rep["verdict"] in allowed)
    print("\nSELF-TEST:", results)
    assert all(ok for _, ok in results.values()), "verification check does not discriminate"
    print("self-test passed: a no-op adapter is caught, and a random adapter is not mistaken for learning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--eval", default="data/dpo_eval.jsonl")
    ap.add_argument("--eval-meta", default="data/eval_meta.jsonl")
    ap.add_argument("--train-meta", default="data/train_meta.jsonl")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-generate", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="logs/verify.json")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-base", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()
    if a.self_test:
        self_test(a.self_test_base, a.limit or 12)
    else:
        verify(a.base, a.adapter, a.eval, a.eval_meta, a.train_meta, a.beta, a.limit, a.n_generate, a.device, a.out)
