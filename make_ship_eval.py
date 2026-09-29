"""
Leg-1 task eval for `soup ship --task-mode metric`.

soup ship's metric mode greedy-decodes each prompt and scores it with
exact / contains / regex. Our task has no single correct answer, so the only
thing a regex can check is a *proxy* for a minimally acceptable support reply:

  - written in Russian (at least one Cyrillic word)
  - asks for specifics or commits to a concrete action
  - does NOT ask for secrets (CVV, SMS code, password, passport photo, full card)

This is deliberately what `metric` mode can measure, and it is weak: a reply can
pass it and still be wrong (wrong policy, wrong sub-case, hallucinated facts).
A SHIP from this instrument therefore says little about quality — that is part
of the Part 3 / verdict discussion. `judge_score` / `pairwise` need a judge-model
endpoint, which a free Colab session does not have.

Output: data/ship_task_eval.jsonl  (30 held-out tickets, >= 1 per category)
"""

import json
from collections import defaultdict
from pathlib import Path

N = 30
PATTERN = (
    r"^(?![\s\S]*(?:CVV|CVC|код из смс|код из sms|пароль|фото паспорта|скан паспорта|полный номер карты))"
    r"(?=[\s\S]*[а-яё]{3,})"
    r"(?=[\s\S]*(?:номер заказа|пришлите|напишите|уточните|подскажите|оформ|провер|верн[её]м|отмен|передали|открыли))"
)


def main() -> None:
    rows = [json.loads(l) for l in open("data/dpo_eval.jsonl", encoding="utf-8")]
    meta = [json.loads(l) for l in open("data/eval_meta.jsonl", encoding="utf-8")]
    by_cat = defaultdict(list)
    for r, m in zip(rows, meta):
        by_cat[m["category"]].append((r, m))
    picked, round_ = [], 0
    while len(picked) < N:
        for cat in sorted(by_cat):
            if round_ < len(by_cat[cat]) and len(picked) < N:
                picked.append(by_cat[cat][round_])
        round_ += 1
    out = Path("data/ship_task_eval.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r, m in picked:
            f.write(json.dumps({
                "prompt": r["prompt"][-1]["content"],   # the ticket; ship applies the chat template
                "expected": PATTERN,
                "scoring": "regex",
                "category": m["category"],
                "id": m["id"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} tasks -> {out}  (pattern {len(PATTERN)} chars)")


if __name__ == "__main__":
    main()
