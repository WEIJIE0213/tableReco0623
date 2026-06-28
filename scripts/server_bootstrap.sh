#!/usr/bin/env bash
# One-time server bootstrap for the training machine.
# Works when the server cannot reach GitHub and receives code by scp/tar.
set -euo pipefail

ENV_NAME="${ENV_NAME:-tablereco}"
PY_VER="${PY_VER:-3.10}"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  for p in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/usr/local/anaconda3/bin/conda" \
    "/usr/local/miniconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [ -x "$p" ]; then
      echo "$p"
      return 0
    fi
  done

  return 1
}

CONDA_EXE="$(find_conda || true)"
if [ -z "$CONDA_EXE" ]; then
  echo "[bootstrap] conda was not found. Install conda or tell Codex to switch this repo to venv."
  exit 1
fi

CONDA_BASE="$("$CONDA_EXE" info --base)"
echo "[bootstrap] conda: $CONDA_EXE"
echo "[bootstrap] conda base: $CONDA_BASE"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[bootstrap] conda env already exists: $ENV_NAME"
else
  echo "[bootstrap] creating conda env: $ENV_NAME (python $PY_VER)"
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi

conda activate "$ENV_NAME"

echo "[bootstrap] python: $(python --version 2>&1)"
echo "[bootstrap] pip: $(python -m pip --version 2>&1)"

echo "[bootstrap] upgrading pip"
python -m pip install -U pip

# 服务器：Driver 570 / CUDA 12.8 -> 用 torch cu124 轮子（PyPI 默认 Linux 轮子即 CUDA 版）。
# 内容一只需 SFT/LoRA，不装 vllm/deepspeed（留到内容三 GRPO 再单独装），避免版本冲突。
echo "[bootstrap] installing torch (pinned, CUDA build)"
python -m pip install "torch==2.5.1" "torchvision==0.20.1"

echo "[bootstrap] installing ms-swift + 训练依赖"
python -m pip install \
  "ms-swift" \
  "accelerate" \
  "peft" \
  "qwen-vl-utils" \
  "pillow" \
  "datasets" \
  "tensorboard"

echo "[bootstrap] 版本与 GPU 自检"
python - <<'PY'
import importlib
def ver(m):
    try: return importlib.import_module(m).__version__
    except Exception as e: return f"<missing: {e.__class__.__name__}>"
import torch
print("torch:", torch.__version__, "| torch.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
print("transformers:", ver("transformers"))
print("swift:", ver("swift"))
print("peft:", ver("peft"), "| accelerate:", ver("accelerate"), "| datasets:", ver("datasets"))
PY

echo "[bootstrap] done"
echo "[bootstrap] next: bash scripts/update_and_train.sh"
