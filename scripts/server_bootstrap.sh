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

echo "[bootstrap] installing training dependencies"
python -m pip install \
  "ms-swift" \
  "transformers" \
  "accelerate" \
  "deepspeed" \
  "vllm" \
  "qwen-vl-utils" \
  "pillow" \
  "datasets"

echo "[bootstrap] GPU check"
python - <<'PY'
try:
    import torch
    print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
except Exception as exc:
    print("torch_check_error:", repr(exc))
PY

echo "[bootstrap] done"
echo "[bootstrap] next: bash scripts/update_and_train.sh"
