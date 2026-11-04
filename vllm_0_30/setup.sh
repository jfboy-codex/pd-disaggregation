#!/usr/bin/env bash
set -euo pipefail

# Run only after activating the Conda environment used by the native baseline.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the baseline Conda environment first." >&2
  exit 1
fi

MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
python -m pip install --index-url "$MIRROR" --upgrade 'vllm==0.30.0' nixl fastapi httpx uvicorn
python - <<'PY'
import importlib.metadata
version = importlib.metadata.version("vllm")
if version != "0.30.0":
    raise SystemExit(f"Expected vLLM 0.30.0, got {version}")
print("vLLM", version)
PY
