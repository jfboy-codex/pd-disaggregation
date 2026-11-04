# KVServe PD baseline

This is an isolated KVServe baseline beside `vllm_0_16`. It uses the same
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
vLLM `>=0.18.0`; this baseline pins vLLM `0.30.0` and KVServe commit
`60f749aa039959db87dda6a6783b5e33b7b83ccc`. The native `0.16.0` environment
is left alone. `setup_env.sh` creates a separate `.venv` and checks out the
pinned KVServe source under `KVServe/`; the latter is not copied into this repo.
The pinned source supplies the complete connector and compression modules.

## Install on the Linux server

From the repository root:

```bash
cd kvserve_baseline
bash setup_env.sh
```

This needs `uv`, `git`, Python 3.12, a CUDA-capable vLLM installation, and a
driver compatible with the selected CUDA wheels. The installer uses
`--torch-backend=auto` and defaults to `nvidia-nvcomp-cu13` for KVServe's
default ANS codec. Set `NVCOMP_PACKAGE=nvidia-nvcomp-cu12` if your isolated
vLLM environment uses CUDA 12. If you already have a tested vLLM wheel, you
can create `.venv` yourself, install that wheel and the pinned KVServe checkout,
then run `start_pd.py` with that interpreter after setting `vllm_version` in
`config.json` to the installed version.

For a different environment location, set `KVSERVE_ENV_DIR` before setup.
The installer refuses to overwrite an existing environment.

## Start and benchmark

Edit `config.json` for your model path, GPU IDs, NUMA nodes and limits. A model
cache path or a concrete snapshot directory is accepted. The default model,
context window and scheduler settings match the existing native baseline.
Set both NUMA nodes to `null` if this server has no matching NUMA layout.

```bash
.venv/bin/python start_pd.py --config config.json
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

`"compression": "default"` enables KVServe's built-in quantizer plus nvCOMP
ANS codec. Use `null` to test the KVServe connector and NCCL transport without
compression. For a custom pipeline, replace this field with the compression
dictionary from the [upstream README](https://github.com/hpdps-group/KVServe#choose-a-compression-mode).

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
