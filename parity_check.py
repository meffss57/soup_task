"""Section 3: streamed vs resident parity. Run in the notebook with: %run -i parity_check.py
(-i so it can use log() and RUN_TAG from section 0)."""
# ---- Section 3 (self-contained): streamed vs resident parity ------------------------------
import datetime, gc, json, os, shutil, tempfile
from pathlib import Path
import torch

# work from the project folder on Drive even if the session was restarted
PROJ = "/content/drive/MyDrive/soup_project"
if not os.path.exists(PROJ):
    from google.colab import drive
    drive.mount("/content/drive")
os.chdir(PROJ)
os.makedirs("logs", exist_ok=True)
print("working in", os.getcwd())

if "RUN_TAG" not in globals():
    RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if "log" not in globals():
    def log(msg, file="logs/run.log"):
        line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line)
        with open(file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# 1) free anything left over from an earlier (failed) run of this cell
for _name in ("runtime",):
    if _name in globals():
        try:
            globals()[_name].close()
        except Exception as e:
            print("previous runtime close:", e)
for _name in ("streamed", "resident", "a", "b", "runtime"):
    globals().pop(_name, None)
gc.collect(); torch.cuda.empty_cache()
print(f"GPU allocated before start: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from soup_cli.utils.layer_shard import shard_checkpoint
from soup_cli.utils.layer_stream import resolve_stream_dtype
from soup_cli.utils.layer_stream_runtime import build_streamed_model
from soup_cli.utils.spectrum_scan import resolve_model_weights

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = resolve_stream_dtype("cuda")                      # 'float16' on a T4
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                  task_type=TaskType.CAUSAL_LM)

# 2) inputs: two real Russian tickets from our data (prompt + chosen, prompt + rejected)
tok = AutoTokenizer.from_pretrained(MODEL)
row0 = json.loads(open("data/dpo_eval.jsonl", encoding="utf-8").readline())
texts = [tok.apply_chat_template(row0["prompt"] + row0[side], tokenize=False) for side in ("chosen", "rejected")]
inputs = [tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda") for t in texts]

# 3) streamed model with a NON-ZERO adapter (lora_B starts at 0, which would compare only the base)
shard_dir = Path(tempfile.mkdtemp()) / "shards"
weights = resolve_model_weights(MODEL)
index = shard_checkpoint(weights, str(shard_dir), dtype=DTYPE, arch="qwen2")
streamed, runtime = build_streamed_model(model_id=weights, shard_dir=str(shard_dir), index=index,
                                         lora_config=LORA, device="cuda", dtype=DTYPE,
                                         buffers=2, pin=True, seed=0)
try:
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for n, p in streamed.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=gen).to(p.device, p.dtype) * 0.01)

    # 4) ordinary resident model, same adapter weights copied in
    resident = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, DTYPE), device_map={"": "cuda"}), LORA)
    src = {k.replace(".inner.", "."): v for k, v in streamed.state_dict().items() if "lora_" in k}
    dst = {k.replace(".inner.", "."): v for k, v in resident.state_dict().items() if "lora_" in k}
    assert src and set(src) == set(dst), f"adapter keys differ: {len(src)} vs {len(dst)}"
    with torch.no_grad():
        for k, v in src.items():
            dst[k].copy_(v.to(dst[k].dtype))

    # 5) compare logits, exact equality
    results = []
    with torch.no_grad():
        for side, ids in zip(("chosen", "rejected"), inputs):
            a = streamed(input_ids=ids).logits
            b = resident(input_ids=ids).logits
            results.append({"input": side, "seq_len": ids.shape[1],
                            "max_abs_diff": (a.float() - b.float()).abs().max().item(),
                            "bit_exact": bool(torch.equal(a, b))})
            del a, b
    parity = {"dtype": DTYPE, "adapter_tensors": len(src), "results": results,
              "all_bit_exact": all(r["bit_exact"] for r in results),
              "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    log(f"parity (forward, real Russian ticket): {parity}", f"logs/parity_{RUN_TAG}.txt")
finally:
    # 6) always release the pinned host store + GPU memory, even if something above failed
    runtime.close()
    for _name in ("streamed", "resident"):
        globals().pop(_name, None)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    shutil.rmtree(shard_dir.parent, ignore_errors=True)
    print(f"cleaned up, GPU allocated now: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
