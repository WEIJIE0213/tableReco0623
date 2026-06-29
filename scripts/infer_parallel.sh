#!/usr/bin/env bash
# 多卡并行推理（数据分片）：每张卡跑一个独立进程处理一部分，最后合并。
# 用法：
#   GPUS=1,2 bash scripts/infer_parallel.sh checkpoints/stage1_lora/v0-XXXX/checkpoint-200
# 不传 GPUS 时默认用空闲的 1,2 两张卡。
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?用法: GPUS=1,2 bash scripts/infer_parallel.sh <adapter_dir>}"
GPUS="${GPUS:-1,2}"
MAXPX="${MAX_PIXELS:-401408}"
LIMIT="${LIMIT:-0}"           # >0 时每片只跑前 N 条（调试）

# 激活环境 + 隔离 ~/.local
CB="$(/usr/local/anaconda3/bin/conda info --base 2>/dev/null || conda info --base)"
# shellcheck disable=SC1091
source "$CB/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-tablereco}"
export PYTHONNOUSERSITE=1

IFS=',' read -ra ARR <<< "$GPUS"
N=${#ARR[@]}
mkdir -p data/eval
echo "[infer] 共 $N 片，GPU: $GPUS"

pids=()
for idx in "${!ARR[@]}"; do
  g="${ARR[$idx]}"
  CUDA_VISIBLE_DEVICES="$g" MAX_PIXELS="$MAXPX" python src/infer_test.py \
    --adapter "$ADAPTER" --num-shards "$N" --shard-id "$idx" --limit "$LIMIT" \
    --out "data/eval/pred_test.part${idx}.jsonl" \
    > "data/eval/infer_gpu${g}.log" 2>&1 &
  pids+=("$!")
  echo "[infer] 分片 $idx -> GPU $g (pid ${pids[$idx]}), 日志 data/eval/infer_gpu${g}.log"
done

echo "[infer] 等待全部完成…(可另开窗口 tail -f data/eval/infer_gpu*.log 看进度)"
wait "${pids[@]}"

cat data/eval/pred_test.part*.jsonl > data/eval/pred_test.jsonl
echo "[infer] 合并完成 -> data/eval/pred_test.jsonl ($(wc -l < data/eval/pred_test.jsonl) 行)"
echo "[infer] 下一步: python src/eval_teds.py --pred data/eval/pred_test.jsonl"
