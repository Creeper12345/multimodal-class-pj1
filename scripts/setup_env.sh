#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-multimodal}"
BACKEND="${2:-cuda124}"

echo "[PJ1] create conda env: ${ENV_NAME}"
conda env list >/dev/null
conda create -n "${ENV_NAME}" python=3.10 pip -y

case "${BACKEND}" in
  cuda124)
    echo "[PJ1] install torch/torchvision for CUDA 12.4"
    conda run -n "${ENV_NAME}" python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
    echo "[PJ1] install decord for Linux CUDA runtime"
    conda run -n "${ENV_NAME}" python -m pip install decord==0.6.0
    ;;
  cpu)
    echo "[PJ1] install torch/torchvision for CPU"
    conda run -n "${ENV_NAME}" python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
    echo "[PJ1] install decord for CPU runtime"
    conda run -n "${ENV_NAME}" python -m pip install decord==0.6.0
    ;;
  mps)
    echo "[PJ1] install torch/torchvision for macOS MPS"
    conda run -n "${ENV_NAME}" python -m pip install torch==2.5.1 torchvision==0.20.1
    echo "[PJ1] install decord from conda-forge for macOS"
    conda install -n "${ENV_NAME}" -c conda-forge decord -y
    ;;
  *)
    echo "Unsupported backend: ${BACKEND}" >&2
    echo "Usage: scripts/setup_env.sh [env_name] [cuda124|cpu|mps]" >&2
    exit 2
    ;;
esac

echo "[PJ1] install shared Python dependencies"
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

echo "[PJ1] environment ready"
echo "Next:"
echo "  conda run -n ${ENV_NAME} python scripts/check_environment.py --check-data"
echo "  conda run -n ${ENV_NAME} python code/pj1/task1/run_retrieval.py --dry-run --max-images 10"
