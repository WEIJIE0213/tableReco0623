#!/usr/bin/env bash
# M1 结构分支训练（Qwen2.5-VL-3B LoRA + 跨度/空标志头）。
# 用法：
#   smoke: MODE=smoke GPU=1 bash scripts/train_m1.sh
#   full : MODE=full  GPU=1 bash scripts/train_m1.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CB="$(/usr/local/anaconda3/bin/conda info --base 2>/dev/null || conda info --base)"
# shellcheck disable=SC1091
source "$CB/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-tablereco}"
export PYTHONNOUSERSITE=1
export USE_HF="${USE_HF:-0}"
export MAX_PIXELS="${MAX_PIXELS:-401408}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU="${GPU:-1}"
MODE="${MODE:-smoke}"
LSPAN="${LSPAN:-0.5}"
LEMPTY="${LEMPTY:-0.5}"

COMMON=(--train data/jsonl/train.jsonl --val data/jsonl/val.jsonl
        --lambda-span "$LSPAN" --lambda-empty "$LEMPTY"
        --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
        --bsz 1 --grad-accum 8 --lr 1e-4 --grad-checkpoint)

if [ "$MODE" = "smoke" ]; then
  OUT="checkpoints/m1/smoke-$(date +%m%d-%H%M)"
  EXTRA=(--max-steps 20 --log-steps 2 --limit 200)
else
  OUT="checkpoints/m1/v1-$(date +%m%d-%H%M)"
  EXTRA=(--epochs 2 --log-steps 10 --save-steps 200)
fi

echo "[m1] MODE=$MODE GPU=$GPU OUT=$OUT MAX_PIXELS=$MAX_PIXELS"
CUDA_VISIBLE_DEVICES="$GPU" python src/m1_train.py --out "$OUT" "${COMMON[@]}" "${EXTRA[@]}"
echo "[m1] done -> $OUT"
