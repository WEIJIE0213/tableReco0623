#!/usr/bin/env bash
# 内容一 Stage1：Qwen2.5-VL-3B + LoRA 表格识别(HTML)微调。
# 在项目根目录运行：
#   bash scripts/train_stage1.sh smoke   # 先冒烟验证(20 步,小样)
#   bash scripts/train_stage1.sh full    # 正式训练
# 注意：运行前需保证 data/jsonl/*.jsonl 已生成(见 src/build_dataset.py)。
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-smoke}"
ENV_NAME="${ENV_NAME:-tablereco}"

# 激活 conda 环境
CB="$(/usr/local/anaconda3/bin/conda info --base 2>/dev/null || conda info --base)"
# shellcheck disable=SC1091
source "$CB/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"

# 隔离用户级 ~/.local，避免和 conda 环境版本打架
export PYTHONNOUSERSITE=1
# 强制从 ModelScope 下载模型（服务器 github/huggingface 的 443 可能不通，ModelScope 可用）
export USE_HF="${USE_HF:-0}"

# 4×4090：DDP；如需指定卡用 CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 限制 Qwen2.5-VL 单图最大像素，防止 3508x2480 高分辨率把显存撑爆(内容二会专门优化这块)
export MAX_PIXELS="${MAX_PIXELS:-1605632}"   # ≈1.5M px，可按显存上调/下调

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUT="checkpoints/stage1_lora"

COMMON=(
  --model "$MODEL"
  --tuner_type lora
  --dataset data/jsonl/train.jsonl
  --val_dataset data/jsonl/val.jsonl
  --torch_dtype bfloat16
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --freeze_vit true
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 8
  --learning_rate 1e-4
  --max_length 4096
  --dataset_num_proc 4
  --logging_steps 5
  --output_dir "$OUT"
)

if [ "$MODE" = "smoke" ]; then
  echo "[train] SMOKE：20 步,验证全链路"
  swift sft "${COMMON[@]}" \
    --max_steps 20 \
    --save_steps 20 --eval_steps 20 \
    --output_dir "${OUT}_smoke"
else
  echo "[train] FULL"
  swift sft "${COMMON[@]}" \
    --num_train_epochs 2 \
    --save_steps 500 --eval_steps 500 \
    --save_total_limit 3 \
    --warmup_ratio 0.03
fi
echo "[train] done -> $OUT"
