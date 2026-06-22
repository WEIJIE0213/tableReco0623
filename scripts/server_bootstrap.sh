#!/usr/bin/env bash
# 服务器一次性环境初始化。先手动 clone 仓库，再在仓库根目录运行本脚本。
#   git clone git@github.com:WEIJIE0213/tableReco0623.git
#   cd tableReco0623 && bash scripts/server_bootstrap.sh
set -e

ENV_NAME="tablereco"
PY_VER="3.10"

echo "[bootstrap] creating conda env: $ENV_NAME (python $PY_VER)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python="$PY_VER" || true
conda activate "$ENV_NAME"

echo "[bootstrap] installing deps..."
pip install -U pip
# 训练框架与基础依赖（版本后续在 requirements.txt 固定）
pip install "ms-swift" "transformers" "accelerate" "deepspeed" "vllm" "qwen-vl-utils" "pillow" "datasets"

echo "[bootstrap] GPU check:"
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"

echo "[bootstrap] done. 下一步：bash scripts/update_and_train.sh"
