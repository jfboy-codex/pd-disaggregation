#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the vLLM 0.30.0 Conda environment first." >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${KVSERVE_SOURCE_DIR:-${HERE}/KVServe}"
REVISION="60f749aa039959db87dda6a6783b5e33b7b83ccc"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
NVCOMP_PACKAGE="${NVCOMP_PACKAGE:-nvidia-nvcomp-cu13}"

python - <<'PY'
import importlib.metadata
version = importlib.metadata.version("vllm")
if version != "0.30.0":
    raise SystemExit(f"Expected vLLM 0.30.0, got {version}; run ../vllm_0_30/setup.sh first")
PY

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone https://github.com/hpdps-group/KVServe.git "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch origin "$REVISION"
git -C "$SOURCE_DIR" checkout --detach "$REVISION"

python -m pip install --index-url "$MIRROR" -e "$SOURCE_DIR" \
  fastapi httpx uvicorn pandas "$NVCOMP_PACKAGE"

python - <<'PY'
import kvserve_v1.connector.compressed_kv_connector
print("KVServe connector import: OK")
PY
