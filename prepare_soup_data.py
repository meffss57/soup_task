"""
Convert the generated dataset into the format soup's DPO trainer reads.

soup `format: dpo` expects {"prompt", "chosen", "rejected"}. We use the
conversational form (lists of messages) so TRL applies the model's chat
template; with plain strings the instruct model would see raw text with no
template at all.

Outputs (data/):
  dpo_train.jsonl      400 rows, fed to `soup train`
  dpo_eval.jsonl       100 held-out rows, used only by our own verification
  train_meta.jsonl, eval_meta.jsonl   id / category / failure_mode / template per row,
                       for per-slice analysis in verify_training.py
"""

import json
from pathlib import Path

SRC = Path("soup_dpo_data")
OUT = Path("data")

SYSTEM = (
    "Ты — оператор службы поддержки интернет-магазина. Отвечай клиенту по-русски, "
    "вежливо и по делу: разберись в конкретной ситуации и предложи следующий шаг."
)


def to_soup(row: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["prompt"]},
        ],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for split, name in (("train", "dpo_train.jsonl"), ("eval", "dpo_eval.jsonl")):
        rows = [json.loads(line) for line in (SRC / f"{split}.jsonl").open(encoding="utf-8")]
        with (OUT / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_soup(r), ensure_ascii=False) + "\n")
        if split in ("train", "eval"):
            with (OUT / f"{split}_meta.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({
                        "id": r["id"],
                        "category": r["category"],
                        "failure_mode": r["metadata"]["failure_mode"],
                        "prompt_template": r["metadata"]["prompt_template"],
                        "noise": r["metadata"]["noise"],
                    }, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows -> {OUT / name}")


if __name__ == "__main__":
    main()
