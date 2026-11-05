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

# Upstream's controller documents CR = original/compressed and V/(B*CR),
# but its predictor uses V*CR/B. Correct that one expression for online
# profile selection; no transport or codec code is changed here.
python - "$SOURCE_DIR" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "kvserve_v1/compression/controller/analytical_model.py"
source = path.read_text(encoding="utf-8")
old = "T_transfer_ms = (V_mb * cr / B_mbps) * 1000"
new = "T_transfer_ms = (V_mb / (cr * B_mbps)) * 1000"
if source.count(old) == 1:
    path.write_text(source.replace(old, new), encoding="utf-8")
elif new not in source:
    raise SystemExit(f"Could not patch controller transfer-volume formula in {path}")
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

# The first probe above runs before queuing. This second optional probe runs
# immediately before NCCL submits the bundle, so a changed header can be
# attributed to sender-side lifetime rather than the transfer itself.
python - "$SOURCE_DIR" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "kvserve_v1/transport/nccl_transport.py"
source = path.read_text(encoding="utf-8")
anchor = (
    "        with torch.cuda.stream(self._send_stream):\n"
    "            self._send_stream.wait_event(ready_event)\n"
    "            start_event = torch.cuda.Event(enable_timing=True)\n"
    "            start_event.record(self._send_stream)\n"
    "            for tensor in body_chunks + aux_tensors:\n"
)
probe = (
    "        if os.environ.get('KVSERVE_WIRE_DEBUG') == '1':\n"
    "            ready_event.synchronize()\n"
    "            header = bytes(body_chunks[0][:16].cpu().tolist()).hex()\n"
    "            logger.warning('[KVSERVE_WIRE] PRE_SEND id=%s header=%s size=%d',\n"
    "                           request_id, header, body_chunks[0].numel())\n"
)
if probe not in source:
    if source.count(anchor) != 1:
        raise SystemExit(f"Could not insert pre-send diagnostic in {path}")
    path.write_text(source.replace(anchor, probe + anchor, 1), encoding="utf-8")
PY

# Check whether the same buffers still contain the header after each NCCL
# operation completes, before another thread consumes or releases them.
python - "$SOURCE_DIR" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "kvserve_v1/transport/nccl_transport.py"
source = path.read_text(encoding="utf-8")
probes = (
    (
        "    def _send_completion_loop(self)",
        "    def _send_one(self",
        "POST_SEND",
        "                if os.environ.get('KVSERVE_WIRE_DEBUG') == '1' and row.get('kind') == 'bundle':\n"
        "                    body = keepalive[0][0]\n"
        "                    header = bytes(body[:16].cpu().tolist()).hex()\n"
        "                    logger.warning('[KVSERVE_WIRE] POST_SEND id=%s header=%s size=%d',\n"
        "                                   row['request_id'], header, body.numel())\n",
    ),
    (
        "    def _recv_completion_loop(self)",
        "    def _listen_loop(self)",
        "POST_RECV",
        "                if os.environ.get('KVSERVE_WIRE_DEBUG') == '1' and isinstance(payload, dict):\n"
        "                    body = payload['body_chunks'][0]\n"
        "                    header = bytes(body[:16].cpu().tolist()).hex()\n"
        "                    logger.warning('[KVSERVE_WIRE] POST_RECV id=%s header=%s size=%d',\n"
        "                                   rid, header, body.numel())\n",
    ),
)
for start_marker, end_marker, label, probe in probes:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    section = source[start:end]
    # Repair the earlier diagnostic patch, which was dedented out of try.
    if label == "POST_SEND":
        legacy_probe = "".join(line[4:] for line in probe.splitlines(keepends=True))
        if legacy_probe in section:
            section = section.replace(legacy_probe, probe, 1)
            source = source[:start] + section + source[end:]
            continue
    if probe in section:
        continue
    anchor = "                event.synchronize()\n"
    if section.count(anchor) != 1:
        raise SystemExit(f"Could not insert {label} diagnostic in {path}")
    section = section.replace(anchor, anchor + probe, 1)
    source = source[:start] + section + source[end:]
path.write_text(source, encoding="utf-8")
PY

python -m py_compile "$SOURCE_DIR/kvserve_v1/transport/nccl_transport.py"
if [[ "${KVSERVE_PATCH_ONLY:-0}" == "1" ]]; then
  echo "KVServe source patches applied; skipping package installation."
  exit 0
fi

python -m pip install --index-url "$MIRROR" -e "$SOURCE_DIR" \
  fastapi httpx uvicorn pandas packaging ninja "$NVCOMP_PACKAGE"

# The upstream Hadamard transformer imports this CUDA extension, but the
# pinned KVServe pyproject does not list it as a runtime dependency.
if ! python -c 'import fast_hadamard_transform' >/dev/null 2>&1; then
  python - <<'PY'
import re
import shutil
import subprocess
import torch

nvcc = shutil.which("nvcc")
if nvcc is None:
    raise SystemExit("Hadamard build needs nvcc matching PyTorch CUDA "
                     f"{torch.version.cuda}; set CUDA_HOME and PATH to that toolkit")
version_text = subprocess.check_output([nvcc, "--version"], text=True)
match = re.search(r"release (\d+\.\d+)", version_text)
if match is None or match.group(1) != torch.version.cuda:
    raise SystemExit(f"Hadamard build: nvcc {match.group(1) if match else 'unknown'} "
                     f"at {nvcc} differs from PyTorch CUDA {torch.version.cuda}. "
                     "Select the matching toolkit with CUDA_HOME and PATH.")
PY
  if [[ -n "${FHT_SOURCE_DIR:-}" ]]; then
    if [[ ! -f "$FHT_SOURCE_DIR/csrc/fast_hadamard_transform_cuda.cu" ]]; then
      echo "FHT_SOURCE_DIR must contain the full upstream source, including csrc/." >&2
      exit 1
    fi
    FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
      python -m pip install -v --no-build-isolation --no-deps "$FHT_SOURCE_DIR"
  else
    python -m pip install --index-url "$MIRROR" --no-build-isolation --no-deps \
      'fast-hadamard-transform==1.1.0'
  fi
fi

python - <<'PY'
import kvserve_v1.connector.compressed_kv_connector
import fast_hadamard_transform
import torch
tensor = torch.randn(2, 128, device="cuda", dtype=torch.float16)
fast_hadamard_transform.hadamard_transform(tensor)
print("KVServe connector import: OK")
print("Hadamard CUDA extension: OK")
PY
