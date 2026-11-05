# KVServe PD baseline

This is a KVServe baseline beside `vllm_0_30`. It uses the same
OpenAI endpoints and port numbers: prefill `8100`, decode `8200`, proxy `8192`.
The existing benchmark configuration therefore continues to use
`http://127.0.0.1:8192/v1` and model name `qwen2.5-7b`.

## What is integrated

The [KVServe paper](https://arxiv.org/abs/2605.13734) and its
[implementation](https://github.com/hpdps-group/KVServe) have these main parts:

| Part | Upstream implementation | Role here |
| --- | --- | --- |
| KV connector and transport | `kvserve_v1/connector`, `transport` | Producer on GPU 0, consumer on GPU 1; ZMQ control and NCCL KV transfer |
| Compression pipeline | `kvserve_v1/compression` | Transform, quantizer and codec stages, selected by `compression` in `config.json` |
| Offline profiling | `kvserve_v1/offline_search` | Available in the pinned upstream checkout for producing candidate profiles; not run during serving |
| Online controller | `kvserve_v1/compression/controller` | Can be selected with a measured profile library and service constraints |
| Request orchestration | `proxy.py`, `protocol.py` in this directory | Sends one-token prefill then streams decode; creates one stable `transfer_id` for both stages |

KVServe is an **out-of-tree vLLM connector**. Its upstream package requires
vLLM `>=0.18.0`; this baseline uses the same activated Conda environment as
the native `0.30.0` baseline and pins KVServe commit
`60f749aa039959db87dda6a6783b5e33b7b83ccc`. `setup.sh` installs only
KVServe and its additional packages, and checks out the
pinned source under `KVServe/`; the latter is not copied into this repo.
The pinned source supplies the complete connector and compression modules.
`setup.sh` also installs `kv_utils_vllm030.py` into the editable KVServe
checkout. This adapter converts vLLM 0.30 FlashAttention's
`[blocks, heads, tokens, 2*head_size]` cache layout to KVServe's
`[2, blocks, tokens, heads, head_size]` transfer layout and back. It is needed
for the Qwen model with the vLLM 0.30 FlashAttention backend.
The setup script also patches the decode connector to use the KV tensor
directly: vLLM 0.30 no longer exposes `ForwardContext.virtual_engine`.
It gives each nvCOMP caller thread a separate `Codec`, as required by
[NVIDIA's Python API](https://docs.nvidia.com/cuda/nvcomp/py_api.html),
to avoid sharing mutable scratch buffers under concurrent requests.
Rerun `setup.sh` after updating this baseline; the launcher checks for the
old access pattern and the nvCOMP patch before starting the servers.

## Install on the Linux server

First run `vllm_0_30/setup.sh` in the activated baseline Conda environment.
Then, from the repository root:

```bash
cd kvserve_baseline
bash setup.sh
```

Python packages are downloaded through the Tsinghua PyPI mirror; the pinned
source checkout still comes from GitHub. The default nvCOMP package is for
CUDA 13. Set `NVCOMP_PACKAGE=nvidia-nvcomp-cu12` if the Conda environment uses
CUDA 12. The script checks that vLLM 0.30.0 is already installed and does not
create another environment. Set `KVSERVE_SOURCE_DIR` for another source path.
If the server cannot clone from GitHub, download the archive for commit
`60f749aa039959db87dda6a6783b5e33b7b83ccc` on another machine, upload it,
and extract it as `kvserve_baseline/KVServe/`. The setup script accepts that
source directory without Git metadata and skips Git network access. In this
case the archive's commit must be checked before upload; the script cannot
verify the revision of an extracted archive.

## Start and benchmark

Edit `config.json` for your model path, GPU IDs, NUMA nodes and limits. A model
cache path or a concrete snapshot directory is accepted. The default model,
context window and scheduler settings match the existing native baseline.
Set both NUMA nodes to `null` if this server has no matching NUMA layout.

```bash
python start_pd.py --config config.json
```

The launcher starts both vLLM engines before waiting for readiness because
NCCL initialization needs both sides. It then starts the proxy. Press Ctrl+C
to stop all three process groups. Logs are in `pd_logs/`.

Smoke test from another shell:

```bash
curl -sS http://127.0.0.1:8192/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":false}'
```

Then run the existing benchmark without changing its backend URL:

```bash
cd ../vllm_pd_benchmark
python -u benchmark.py --config config.json
```

Stop the native baseline before starting KVServe because the ports are shared.
No GPU measurement was performed in this Windows workspace; confirm the smoke
test and inspect both engine logs on the server before collecting results.

## Compression settings

The bundled config uses KVServe's quantizer plus nvCOMP ANS codec with the
upstream default parameters and the bundled `Qwen2.5-7B-Instruct_scores.csv`.
The explicit model name matters because vLLM is launched from a Hugging Face
snapshot path whose basename is a commit hash; KVServe's `"default"` mode
would look for `<hash>_scores.csv` and fail on the first request. Use `null` to
test the KVServe connector and NCCL transport without compression. For a
different model, supply its own measured scores CSV and model name in the
quantizer config, or choose another supported compression pipeline from the
[upstream README](https://github.com/hpdps-group/KVServe#choose-a-compression-mode).

If decode reports `not compressed with NVCOMP_NATIVE bitstream kind`, run
`config_bitpack_diagnostic.json` with `--config` to test the same quantizer and
NCCL transfer using KVServe's 8-bit packing codec. This isolates the nvCOMP
path; results from the diagnostic config must be labeled as bitpack rather
than nvCOMP ANS. Keep `config.json` for the intended baseline.

For a failed bitpack run, rerun `setup.sh` after pulling updates, then set
`KVSERVE_WIRE_DEBUG=1` when starting `start_pd.py`. The optional log lines
`[KVSERVE_WIRE] SEND` and `RECV` record the first 16 payload bytes, size, and
codec shape for each transfer ID in `prefill.log` and `decode.log`. Compare
the same ID on both sides to tell whether the header changed during transfer.
`PRE_SEND` records the header again immediately before NCCL submits the send;
it distinguishes a sender-side buffer change after queuing from a transfer
or receive-side problem.
`POST_SEND` and `POST_RECV` record the same buffers after the respective NCCL
completion events, separating in-flight changes from later buffer handling.
Disable the variable for measured benchmarks because the probe copies header
bytes to the CPU and adds synchronization.

For the paper's adaptive controller, first produce a profile library from
offline profiling and configure:

```json
{
  "mode": "controller",
  "library_path": "/absolute/path/to/profiles.json",
  "service_config": {
    "bandwidth_mbps": 1000.0,
    "slo_ms": 200.0,
    "accuracy_requirement": 0.92
  }
}
```

Place that object at the `compression` key on both endpoints through this
single config file. The default mode is a fixed profile: it does not claim
adaptive decisions without an offline profile library. The benchmark measures
request latency and throughput; it does not report internal KV transfer volume
unless separate connector stats are collected.
