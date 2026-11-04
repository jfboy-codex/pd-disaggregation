#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${KVSERVE_ENV_DIR:-${HERE}/.venv}"
SOURCE_DIR="${KVSERVE_SOURCE_DIR:-${HERE}/KVServe}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
KVSERVE_REVISION="60f749aa039959db87dda6a6783b5e33b7b83ccc"
NVCOMP_PACKAGE="${NVCOMP_PACKAGE:-nvidia-nvcomp-cu13}"

command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
command -v git >/dev/null || { echo 'git is required' >&2; exit 1; }

if [[ -e "$ENV_DIR" ]]; then
  echo "Environment already exists: $ENV_DIR" >&2
  echo "Set KVSERVE_ENV_DIR to a new path to create an isolated copy." >&2
  exit 1
fi

uv venv --python 3.12 "$ENV_DIR"
uv pip install --python "$ENV_DIR/bin/python" "vllm==$VLLM_VERSION" --torch-backend=auto

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone https://github.com/hpdps-group/KVServe.git "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch origin "$KVSERVE_REVISION"
git -C "$SOURCE_DIR" checkout --detach "$KVSERVE_REVISION"

uv pip install --python "$ENV_DIR/bin/python" -e "$SOURCE_DIR" \
  fastapi httpx uvicorn pandas "$NVCOMP_PACKAGE"

"$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import kvserve_v1.connector.compressed_kv_connector
print("vLLM:", importlib.metadata.version("vllm"))
print("KVServe connector import: OK")
PY

echo "Environment ready: $ENV_DIR"
