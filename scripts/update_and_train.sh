#!/usr/bin/env bash
# 服务器一键：拉取最新代码后开始训练。
# 用法：bash scripts/update_and_train.sh [配置名]
set -e

cd "$(dirname "$0")/.."

echo "[update] git pull..."
git pull origin main

# ===== 按你的环境改这两行 =====
ENV_NAME="tablereco"          # conda 环境名
CONFIG="${1:-stage1_lora}"    # 默认跑的配置（对应 configs/ 下的文件）
# =============================

# 激活环境（conda 或 venv 二选一）
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$ENV_NAME" || true

echo "[train] launching config: $CONFIG"
# 占位训练命令，骨架阶段先空跑确认链路；真正训练脚本后续补到 src/train.py
if [ -f "configs/${CONFIG}.yaml" ]; then
  echo "[train] (placeholder) would run: swift sft --config configs/${CONFIG}.yaml"
else
  echo "[train] config configs/${CONFIG}.yaml 还不存在——同步链路已通，等待 Claude 补训练配置。"
fi
echo "[done]"
