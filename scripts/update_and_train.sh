#!/usr/bin/env bash
# Server entrypoint: optionally update code, activate env, then launch training.
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_NAME="${ENV_NAME:-tablereco}"
CONFIG="${1:-stage1_lora}"

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

if [ -d ".git" ] && git remote get-url origin >/dev/null 2>&1; then
  echo "[update] git pull origin main"
  git pull origin main
else
  echo "[update] no git remote checkout on server; assuming code was synced by scp/tar"
fi

CONDA_EXE="$(find_conda || true)"
if [ -n "$CONDA_EXE" ]; then
  CONDA_BASE="$("$CONDA_EXE" info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
  else
    echo "[env] conda env '$ENV_NAME' does not exist yet; run bash scripts/server_bootstrap.sh before real training"
  fi
else
  echo "[env] conda not found; continuing with current shell"
fi

echo "[env] python: $(python --version 2>&1 || true)"
echo "[train] config: $CONFIG"

if [ -f "configs/${CONFIG}.yaml" ]; then
  echo "[train] would run: swift sft --config configs/${CONFIG}.yaml"
  # swift sft --config "configs/${CONFIG}.yaml"
else
  echo "[train] config configs/${CONFIG}.yaml does not exist yet"
  echo "[train] sync path is working; training config will be added in the next phase"
fi

echo "[done]"
