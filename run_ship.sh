#!/usr/bin/env bash
# soup ship on the trained adapter (run from Colab: !bash /content/run_ship.sh)
set -u
cd /content/drive/MyDrive/soup_project || exit 1
SHA=96ae9c0502f69a355d5f0fe24a07302d3b67f799
wget -q -O data/ship_task_eval.jsonl \
  "https://raw.githubusercontent.com/meffss57/soup_task/${SHA}/data/ship_task_eval.jsonl"
echo "[$(date -u +%FT%TZ)] task eval rows: $(wc -l < data/ship_task_eval.jsonl)" | tee -a logs/ship_run.txt
soup ship \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter adapters/20260929_165111 \
  --task-eval data/ship_task_eval.jsonl \
  --task-mode metric \
  --device cuda \
  -o logs/ship_verdict.json \
  --emit-evidence logs/ship_evidence.json 2>&1 | tee -a logs/ship_run.txt
echo "[$(date -u +%FT%TZ)] soup ship exit code: ${PIPESTATUS[0]}  (0=SHIP, 2=DON'T SHIP)" | tee -a logs/ship_run.txt
