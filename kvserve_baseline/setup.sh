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

# vLLM 0.30 binds one KV tensor directly to each attention layer; the older
# ForwardContext.virtual_engine selector no longer exists.
python - "$SOURCE_DIR" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "kvserve_v1/connector/compressed_kv_connector.py"
source = path.read_text(encoding="utf-8")
old = "kv_cache_layer = kv_cache[forward_context.virtual_engine]"
new = "kv_cache_layer = kv_cache  # vLLM 0.30 binds a tensor directly"
if source.count(old) == 1:
    path.write_text(source.replace(old, new), encoding="utf-8")
elif new not in source:
    raise SystemExit(f"Could not patch vLLM 0.30 KV cache access in {path}")
PY

# nvCOMP Codec instances hold mutable scratch state and must be thread-local.
python - "$SOURCE_DIR" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "kvserve_v1/compression/codec/nvcomp_func.py"
source = path.read_text(encoding="utf-8")
old = "key = (device.index, stream.cuda_stream)"
new = "key = (threading.get_ident(), device.index, stream.cuda_stream)"
if source.count(old) == 1:
    source = source.replace("import torch\n", "import threading\nimport torch\n", 1)
    path.write_text(source.replace(old, new), encoding="utf-8")
elif new not in source:
    raise SystemExit(f"Could not patch thread-local nvCOMP Codec in {path}")
PY

# Optional one-run diagnostic: compare the codec header before NCCL send and
# after NCCL receive. It is disabled unless KVSERVE_WIRE_DEBUG=1 is set.
python - "$SOURCE_DIR" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "kvserve_v1/connector/compressed_kv_connector.py"
source = path.read_text(encoding="utf-8")
send_anchor = "                    wire = build_wire(compressed, _max_nccl_chunk_bytes())\n"
send_probe = (
    "                    if os.environ.get('KVSERVE_WIRE_DEBUG') == '1':\n"
    "                        header = bytes(wire.body_chunks[0][:16].cpu().tolist()).hex()\n"
    "                        logger.warning('[KVSERVE_WIRE] SEND id=%s header=%s size=%d shape=%s',\n"
    "                                       transfer_id, header, wire.body_chunks[0].numel(),\n"
    "                                       wire.meta.get('metadata', {}).get('codec_shape'))\n"
)
recv_anchor = "                        compressed = restore_from_wire(wire)\n"
recv_probe = (
    "                        if os.environ.get('KVSERVE_WIRE_DEBUG') == '1':\n"
    "                            header = bytes(wire.body_chunks[0][:16].cpu().tolist()).hex()\n"
    "                            logger.warning('[KVSERVE_WIRE] RECV id=%s header=%s size=%d shape=%s',\n"
    "                                           transfer_id, header, wire.body_chunks[0].numel(),\n"
    "                                           wire.meta.get('metadata', {}).get('codec_shape'))\n"
)
for anchor, probe in ((send_anchor, send_probe), (recv_anchor, recv_probe)):
    if probe in source:
        continue
    if source.count(anchor) != 1:
        raise SystemExit(f"Could not insert wire diagnostic in {path}")
    source = source.replace(anchor, anchor + probe, 1)
path.write_text(source, encoding="utf-8")
PY

python -m pip install --index-url "$MIRROR" -e "$SOURCE_DIR" \
  fastapi httpx uvicorn pandas "$NVCOMP_PACKAGE"

python - <<'PY'
import kvserve_v1.connector.compressed_kv_connector
print("KVServe connector import: OK")
PY


