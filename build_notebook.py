"""Builds soup_dpo_t4.ipynb (sections 0-5: setup, checks, parity, budget, DPO run)."""

import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# Soup take-home — DPO with layer streaming on a Colab T4

**Before running:** upload the `soup_project` folder to Google Drive (`MyDrive/soup_project`),
then *Runtime → Change runtime type → T4 GPU*.

Every cell writes raw output to `logs/` on Drive with timestamps, so nothing is lost if Colab
disconnects. Sections:

| # | What | GPU time |
|---|---|---|
| 0 | Drive, logging, install | ~4 min |
| 1 | Environment record, bf16 check, raw `nvidia-smi` | 1 min |
| 2 | Data checks (`soup data lint`, token lengths, `soup doctor`) | 2 min |
| 3 | Parity: streamed model == resident model, bit for bit | ~5 min |
| 4 | Memory budget (predicted, before training) | 0 |
| 5 | DPO run, streamed, with per-step timestamped log + `nvidia-smi` logger | ~40-70 min |
| 6 | Part 2 verification from the saved adapter (+ optional CPU self-test) | ~5 min |
""")

md("## 0. Drive, logging, install")
code(r'''
from google.colab import drive
drive.mount("/content/drive")

import os, sys, json, time, datetime, subprocess
PROJ = "/content/drive/MyDrive/soup_project"
os.chdir(PROJ)
os.makedirs("logs", exist_ok=True)
RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def stamp():
    return datetime.datetime.now().isoformat(timespec="seconds")

def log(msg, file="logs/run.log"):
    line = f"[{stamp()}] {msg}"
    print(line)
    with open(file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def sh(cmd, file):
    """Run a shell command, print it, and save raw output with a timestamp header."""
    log(f"$ {cmd}  -> {file}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    text = out.stdout + out.stderr
    with open(file, "a", encoding="utf-8") as f:
        f.write(f"===== {stamp()} $ {cmd} (exit {out.returncode})\n{text}\n")
    print(text[-4000:])
    return out.returncode

log(f"session start, RUN_TAG={RUN_TAG}, cwd={os.getcwd()}")
print(sorted(os.listdir(".")))
''')
code(r'''
# Colab preinstalls an old torchao that makes peft raise ImportError; soup's own notebook removes it.
%pip uninstall -q -y torchao
_extra = "--ignore-requires-python" if sys.version_info >= (3, 13) else ""
%pip install -q "soup-cli[train]==0.75.1" {_extra}
''')

md("## 1. Environment record, bf16 check, raw nvidia-smi")
code(r'''
import torch, transformers, peft, trl, bitsandbytes, soup_cli
env = {
    "time": stamp(), "python": sys.version.split()[0], "soup_cli": soup_cli.__version__,
    "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
    "peft": peft.__version__, "trl": trl.__version__, "bitsandbytes": bitsandbytes.__version__,
    "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
    "vram_total_GB": torch.cuda.get_device_properties(0).total_memory / 1e9,
    "bf16_incl_emulation": torch.cuda.is_bf16_supported(),
    "bf16_in_hardware": torch.cuda.is_bf16_supported(including_emulation=False),
}
from soup_cli.utils.gpu import bf16_fp16_flags
env["soup_precision"] = dict(zip(("bf16", "fp16"), bf16_fp16_flags("cuda")))
import psutil; env["host_ram_GB"] = psutil.virtual_memory().total / 1e9
json.dump(env, open(f"logs/env_{RUN_TAG}.json", "w"), indent=2, default=str)
print(json.dumps(env, indent=2, default=str))
assert env["soup_precision"]["fp16"] and not env["soup_precision"]["bf16"], "T4 must train in fp16"
sh("nvidia-smi", f"logs/nvidia-smi_{RUN_TAG}.txt")
''')

md("## 2. Data checks (soup's own + ours)")
code(r'''
# soup's preference linter: length bias, near-duplicates, identical pairs, prompt leak
sh("soup data lint data/dpo_train.jsonl --format dpo -o logs/lint_train.json", f"logs/data_checks_{RUN_TAG}.txt")
sh("soup data lint data/dpo_eval.jsonl --format dpo -o logs/lint_eval.json", f"logs/data_checks_{RUN_TAG}.txt")
# soup data doctor refuses preference data ("use soup data lint") -> template / EOS / truncation
# are NOT checked by soup for DPO. We check them ourselves:
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
rows = [json.loads(l) for f in ("data/dpo_train.jsonl", "data/dpo_eval.jsonl") for l in open(f, encoding="utf-8")]
lens, eos_ok = [], 0
for r in rows:
    for side in ("chosen", "rejected"):
        t = tok.apply_chat_template(r["prompt"] + r[side], tokenize=False)
        lens.append(len(tok(t)["input_ids"]))
        eos_ok += t.rstrip().endswith(tok.eos_token)
report = {"max_len": max(lens), "over_512": sum(l > 512 for l in lens),
          "mean_len": sum(lens) / len(lens), "answers_ending_with_eos": f"{eos_ok}/{len(lens)}"}
log(f"token check: {report}", f"logs/data_checks_{RUN_TAG}.txt")
sh("soup doctor", f"logs/soup_doctor_{RUN_TAG}.txt")
''')

md("""
## 3. Parity: is the streamed model the same model?

A streaming bug is silent: the loss still falls. So before trusting any streamed run we check
that a streamed Qwen2.5-1.5B and an ordinary resident one, carrying the **same non-zero adapter**,
give bit-identical logits (method from soup's own `proof-4gb.ipynb`, applied to our model).
""")
code(r'''
import tempfile, gc
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from soup_cli.utils.layer_shard import shard_checkpoint
from soup_cli.utils.layer_stream import resolve_stream_dtype
from soup_cli.utils.layer_stream_runtime import build_streamed_model
from soup_cli.utils.spectrum_scan import resolve_model_weights

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = resolve_stream_dtype("cuda")
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                  task_type=TaskType.CAUSAL_LM)
workdir = Path(tempfile.mkdtemp())
weights = resolve_model_weights(MODEL)
index = shard_checkpoint(weights, str(workdir / "shards"), dtype=DTYPE, arch="qwen2")
streamed, runtime = build_streamed_model(model_id=weights, shard_dir=str(workdir / "shards"), index=index,
                                         lora_config=LORA, device="cuda", dtype=DTYPE, buffers=2, pin=True, seed=0)
gen = torch.Generator().manual_seed(7)
with torch.no_grad():   # lora_B starts at 0 -> make the adapter load-bearing, else we only compare the base
    for n, p in streamed.named_parameters():
        if "lora_B" in n:
            p.copy_(torch.randn(p.shape, generator=gen).to(p.device, p.dtype) * 0.01)
resident = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, DTYPE),
                                                               device_map={"": "cuda"}), LORA)
src = {k.replace(".inner.", "."): v for k, v in streamed.state_dict().items() if "lora_" in k}
dst = {k.replace(".inner.", "."): v for k, v in resident.state_dict().items() if "lora_" in k}
assert src and set(src) == set(dst)
with torch.no_grad():
    for k, v in src.items():
        dst[k].copy_(v.to(dst[k].dtype))
# transformers 5: apply_chat_template(return_tensors=...) returns a dict, so render text, then tokenize
text = tok.apply_chat_template(rows[0]["prompt"] + rows[0]["chosen"], tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
with torch.no_grad():
    a = streamed(input_ids=ids).logits
    b = resident(input_ids=ids).logits
parity = {"max_abs_diff": (a.float() - b.float()).abs().max().item(), "bit_exact": bool(torch.equal(a, b)),
          "seq_len": ids.shape[1]}
log(f"parity (forward, real Russian ticket): {parity}", f"logs/parity_{RUN_TAG}.txt")
runtime.close(); del streamed, resident, a, b; gc.collect(); torch.cuda.empty_cache()
''')

md("## 4. Memory budget — predicted before training")
code(r'''
sh("python memory_budget.py --out logs/memory_budget_predicted.json", f"logs/memory_budget_{RUN_TAG}.txt")
''')

md("""
## 5. DPO run (streamed)

Runs **in this process** (not `!soup train`) so `torch.cuda.max_memory_allocated()` measures the
training itself — same code path the CLI uses. A callback writes one timestamped JSON line per step
with the DPO metrics, memory, the fp16 GradScaler scale and whether the adapter actually changed
(a skipped optimizer step leaves it unchanged). `nvidia-smi` logs to CSV every 2 s in the background.
""")
code(r'''
# background nvidia-smi logger (raw)
smi = subprocess.Popen(
    f"nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw "
    f"--format=csv -l 2 > logs/nvidia-smi_train_{RUN_TAG}.csv", shell=True)
log(f"nvidia-smi logger pid {smi.pid}")
''')
code(r'''
from transformers import TrainerCallback
from soup_cli.config.loader import load_config
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.dpo import DPOTrainerWrapper

class StepLogger(TrainerCallback):
    """One JSON line per log event: DPO metrics + memory + scaler + did-the-adapter-move."""
    def __init__(self, path, model, trainer):
        self.path, self.model, self.trainer, self.prev = path, model, trainer, None
        self.t0 = time.time()
    def _lora_fingerprint(self):
        with torch.no_grad():
            return sum(p.float().abs().sum().item() for n, p in self.model.named_parameters()
                       if "lora_B" in n and ".default." in n)
    def _ref_fingerprint(self):
        with torch.no_grad():
            return sum(p.float().abs().sum().item() for n, p in self.model.named_parameters()
                       if "lora_B" in n and ".ref." in n)
    def on_log(self, args, state, control, logs=None, **kw):
        fp = self._lora_fingerprint()
        scaler = getattr(self.trainer.accelerator, "scaler", None)
        rec = {"time": stamp(), "elapsed_s": round(time.time() - self.t0, 1), "step": state.global_step,
               **(logs or {}),
               "lora_B_abs_sum": fp, "lora_changed_since_last_log": None if self.prev is None else fp != self.prev,
               "ref_lora_B_abs_sum": self._ref_fingerprint(),   # frozen DPO reference: must stay constant
               "grad_scaler_scale": scaler.get_scale() if scaler is not None else None,
               "cuda_alloc_GB": torch.cuda.memory_allocated() / 1e9,
               "cuda_max_alloc_GB": torch.cuda.max_memory_allocated() / 1e9,
               "cuda_max_reserved_GB": torch.cuda.max_memory_reserved() / 1e9}
        self.prev = fp
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

cfg = load_config("soup.yaml")
dataset = load_dataset(cfg.data)
log(f"dataset: train={len(dataset['train'])} val={len(dataset.get('val') or [])}")

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
wrapper = DPOTrainerWrapper(cfg)
wrapper.setup(dataset)            # downloads, shards, builds the streamed model + DPO trainer
mem_after_setup = {"alloc_GB": torch.cuda.memory_allocated() / 1e9, "max_GB": torch.cuda.max_memory_allocated() / 1e9}
log(f"after setup: {mem_after_setup}")
steplog = f"logs/train_steps_{RUN_TAG}.jsonl"
wrapper.trainer.add_callback(StepLogger(steplog, wrapper.trainer.model, wrapper.trainer))

torch.cuda.reset_peak_memory_stats()
log("training start")
result = wrapper.train()
log(f"training end: {result}")
peaks = {"max_allocated_GB": torch.cuda.max_memory_allocated() / 1e9,
         "max_reserved_GB": torch.cuda.max_memory_reserved() / 1e9, "after_setup": mem_after_setup,
         "result": result}
json.dump(peaks, open(f"logs/memory_measured_{RUN_TAG}.json", "w"), indent=2, default=str)
print(json.dumps(peaks, indent=2, default=str))
''')
code(r'''
smi.terminate()
sh("nvidia-smi", f"logs/nvidia-smi_{RUN_TAG}.txt")
import shutil
shutil.copytree(result["output_dir"], f"adapters/{RUN_TAG}", dirs_exist_ok=True)
log(f"adapter copied to adapters/{RUN_TAG}")
''')

md("""
## 6. Part 2 — verification from artifacts on disk

Fresh model + the adapter **as saved on Drive**, the 100 held-out pairs, and five checks:
LOADED / MOVED / ACTIVE / LEARNED (held-out margin vs reference, sliced) / CONTROL (random
direction, same norm). Writes `logs/verify_<run>.json`, per-pair rows and side-by-side generations.
""")
code(r'''
import gc
try:
    del wrapper
except NameError:
    pass
gc.collect(); torch.cuda.empty_cache()
ADAPTER = f"adapters/{RUN_TAG}"      # or set by hand, e.g. "adapters/20261001_120000"
sh(f"python verify_training.py --adapter {ADAPTER} --out logs/verify_{RUN_TAG}.json",
   f"logs/verify_{RUN_TAG}.txt")
''')

md("""
### Optional (CPU is fine): self-test of the verification script

Builds two fake adapters on Qwen2.5-0.5B — an all-zero one (untrained) and a random one — and
asserts the script calls them `NO_OP` and `CHANGED_NOT_LEARNED`. Shows the check discriminates.
""")
code(r'''
sh("python verify_training.py --self-test --limit 12", f"logs/verify_selftest_{RUN_TAG}.txt")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("soup_dpo_t4.ipynb", "w"), indent=1, ensure_ascii=False)
print("wrote soup_dpo_t4.ipynb with", len(cells), "cells")
