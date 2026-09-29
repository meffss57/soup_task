"""
Compare the writing style of the synthetic tickets with real Russian customer
complaints about delivery / pickup points / couriers / online-store offices.

Real reference: Yandex Geo Reviews Dataset 2023 (MIT licence)
  https://github.com/yandex/geo-reviews-dataset-2023
  https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023

Only aggregate statistics are written out. No real review text is copied into
the DPO dataset (reviews can contain names of staff members).

Usage:
    pip install pandas pyarrow
    python style_reference.py            # downloads ~170 MB parquet once
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

PARQUET_URL = ("https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023/resolve/main/"
               "data/train-00000-of-00001-49261e4e5a35a5a0.parquet")
CACHE = Path("reference_data/geo_reviews_2023.parquet")
DATASET = Path("soup_dpo_data/soup_dpo_all.jsonl")
OUT = Path("soup_dpo_data/style_reference_report.json")

# Rubrics that match the support-ticket topic (delivery, pickup, couriers, e-shops).
RUBRICS = "Пункт выдачи|Курьерские услуги|Офис интернет-магазина|Почтов|Постамат"


def style_stats(texts: list[str]) -> dict:
    s = pd.Series(texts)
    return {
        "n": int(len(s)),
        "median_len_chars": int(s.str.len().median()),
        "starts_lowercase": round(float(s.str.match(r"^[а-яё]").mean()), 3),
        "no_final_punct": round(float((~s.str.strip().str.contains(r"[.!?)…]$")).mean()), 3),
        "multi_excl_or_question": round(float(s.str.contains(r"[!?]{2,}").mean()), 3),
        "emoji": round(float(s.str.contains(r"[\U0001F300-\U0001FAFF]").mean()), 3),
        "caps_word": round(float(s.str.contains(r"\b[А-ЯЁ]{4,}\b").mean()), 3),
        "ellipsis_or_brackets": round(float(s.str.contains(r"\.\.\.|\(\(|\)\)").mean()), 3),
        "no_space_after_comma": round(float(s.str.contains(r",[а-яА-ЯёЁ]").mean()), 3),
    }


def main() -> None:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {PARQUET_URL}")
        urllib.request.urlretrieve(PARQUET_URL, CACHE)

    geo = pd.read_parquet(CACHE)
    ref = geo[geo.rubrics.str.contains(RUBRICS, regex=True) & (geo.rating <= 2)]
    ref_texts = ref.text.str.replace("\\n", "\n", regex=False).tolist()

    rows = [json.loads(line) for line in DATASET.open(encoding="utf-8")]
    ours = [r["prompt"] for r in rows]

    report = {
        "reference": {
            "source": "Yandex Geo Reviews Dataset 2023 (MIT)",
            "filter": f"rubrics ~ '{RUBRICS}' and rating <= 2",
            "stats": style_stats(ref_texts),
        },
        "synthetic_prompts": style_stats(ours),
        "note": ("Public map reviews are longer and more formal than support-chat tickets, "
                 "so length and lowercase share are not expected to match exactly; "
                 "punctuation / emphasis habits are what the noise pipeline is tuned to."),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
