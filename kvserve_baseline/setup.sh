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

if [[ ! -d "$SOURCE_DIR" ]]; then
  git clone https://github.com/hpdps-group/KVServe.git "$SOURCE_DIR"
fi

if [[ -d "$SOURCE_DIR/.git" ]]; then
  if ! git -C "$SOURCE_DIR" cat-file -e "${REVISION}^{commit}" 2>/dev/null; then
    git -C "$SOURCE_DIR" fetch origin "$REVISION"
  fi
  git -C "$SOURCE_DIR" checkout --detach "$REVISION"
elif [[ -f "$SOURCE_DIR/pyproject.toml" && -f "$SOURCE_DIR/kvserve_v1/connector/compressed_kv_connector.py" ]]; then
  echo "Using uploaded KVServe source at $SOURCE_DIR (no Git metadata)."
  echo "Make sure it was downloaded from commit $REVISION."
else
  echo "Invalid KVServe source at $SOURCE_DIR; expected a Git checkout or extracted source archive." >&2
  exit 1
fi

# vLLM 0.30 FlashAttention stores K and V in the last dimension.
# The pinned KVServe utility expects a separate K/V axis, so install the
# compatibility adapter into the editable source checkout.
cp "$HERE/kv_utils_vllm030.py" "$SOURCE_DIR/kvserve_v1/utils/kv_utils.py"

python -m pip install --index-url "$MIRROR" -e "$SOURCE_DIR" \
  fastapi httpx uvicorn pandas "$NVCOMP_PACKAGE"

python - <<'PY'
import kvserve_v1.connector.compressed_kv_connector
print("KVServe connector import: OK")
PY

