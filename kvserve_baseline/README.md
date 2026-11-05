# KVServe PD baseline

This baseline uses the pinned [upstream KVServe](https://github.com/hpdps-group/KVServe)
connector, compression pipeline, offline search results, profile library, and
online controller with vLLM 0.30.0. The proxy remains at `127.0.0.1:8192` with
model name `qwen2.5-7b`, so the existing benchmark config needs no URL change.
Prefill uses port 8100 and decode uses port 8200.

| Part | Implementation |
| --- | --- |
| KV transfer | Upstream `kvserve_v1/connector` and `transport` |
| Compression | Upstream transform, quantizer, and codec modules |
| Offline candidate search | Upstream `kvserve_v1/offline_search`; Qwen results are bundled |
| Controller | Upstream `ProfileLibrary` and `OnlineController` |
| Request routing | Local `proxy.py` supplies one `transfer_id` to prefill and decode |

The default `config.json` uses `mode: controller`. `prepare_profiles.py` converts
the bundled Qwen search candidates into the format that upstream `ProfileLibrary`
expects, measures each candidate's encode/decode speed and wire size using a real
Qwen KV sample, and writes `profiles/qwen2.5-7b.json`. The controller then
selects a profile per request and may choose no compression when its benefit or
SLO checks fail. `config_fixed.json` preserves the earlier fixed pipeline.

## Install and prepare on the server

Activate the same Conda environment used by `vllm_0_30`. It must have vLLM
0.30.0 and the local Qwen2.5-7B-Instruct snapshot. From the repository root:

```bash
cd ~/pd_disaggregation
KVSERVE_SOURCE_DIR="$PWD/kvserve_baseline/KVServe-upload" bash kvserve_baseline/setup.sh
python kvserve_baseline/prepare_profiles.py --config kvserve_baseline/config.json
```

Set `KVSERVE_SOURCE_DIR` to the actual uploaded KVServe directory if its name
differs. If there is no uploaded source, omit the variable; setup will clone
the pinned commit `60f749aa039959db87dda6a6783b5e33b7b83ccc`. The script
uses the Tsinghua PyPI mirror and installs into the active environment; it does
not create another environment. For CUDA 12, prefix setup with
`NVCOMP_PACKAGE=nvidia-nvcomp-cu12`. The pinned upstream checkout is installed
editable. Setup also applies the vLLM 0.30 KV layout and connector compatibility
patches and the existing nvCOMP thread-local patch. Uploaded source without Git
metadata cannot have its revision checked automatically.
Setup also corrects one upstream controller formula: transfer time is
`original_bytes / (compression_ratio * bandwidth)`, matching the upstream
docstring and benefit check.
The upstream package does not declare the CUDA Hadamard extension used by its
transformer, so setup installs `fast-hadamard-transform` separately and checks
that it runs on CUDA. Its package installer may fetch a matching binary from
GitHub or build one locally; if installation fails, keep the full pip error.
If GitHub downloads are unavailable on the server, upload the full
`fast-hadamard-transform` source and set `FHT_SOURCE_DIR` to that directory
when running setup.
When `FHT_SOURCE_DIR` is set, setup forces a local build and shows compiler
output, skipping upstream's attempt to download a prebuilt wheel first.
Compiling the extension requires an `nvcc` toolkit with the same CUDA version
as `torch.version.cuda`. Setup checks this before invoking pip and reports the
detected versions; point `CUDA_HOME` and `PATH` at the matching toolkit.

Stop serving processes before building profiles: model loading uses GPU 0.
`prepare_profiles.py` uses the upstream Qwen `tolerance_3_results.json` by
default. It checks each candidate by running upstream compression and
decompression on a local GPU KV sample. The script fails without writing the
final library if a candidate cannot be decoded. To use results from a fresh
upstream search, pass `--search-results /path/to/tolerance_3_results.json`.
The upstream search entry point and its model/task settings are documented in
`KVServe-upload/kvserve_v1/offline_search/README.md`; its default task is
`longbench_2wikimqa` with only five examples, so re-run that search with your
target tasks for research results.

The bundled upstream accuracy values are relative search scores, including a
value above 100 from the tiny evaluation sample. They are **not** validated
accuracy guarantees for ShareGPT, GSM8K, HumanEval, or all LongBench tasks.
The generated library records its source and speed measurement. Its
`accuracy_requirement` uses the same score scale; the example value `97.0`
matches the upstream search's tolerance threshold. The `bandwidth_mbps` and
`slo_ms` fields in `config.json` are examples. Set them to your measured
network bandwidth in MiB/s and desired KV-stage latency before comparing
controller behavior. The profiler measures one GPU; different decode hardware
needs separate performance validation.

## Start and benchmark

From the repository root, after the profile file exists:

```bash
python kvserve_baseline/start_pd.py --config kvserve_baseline/config.json
```

The launcher starts both vLLM engines, waits for them, then starts the proxy.
Use Ctrl+C to stop them. Logs are under `kvserve_baseline/pd_logs/`.
In another shell, test one request before a burst benchmark:

```bash
curl -sS --max-time 120 http://127.0.0.1:8192/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":false}'
cd ~/pd_disaggregation/vllm_pd_benchmark
python -u benchmark.py --config config.json
```

Stop the native baseline before starting KVServe because the ports overlap.
The server-side KV transfer corruption observed under burst load has not been
resolved; successful profile generation and service health checks do not prove
the benchmark will complete. No GPU serving test was run in this Windows
workspace. Check `prefill.log` and `decode.log` if the smoke test fails.

## Other modes and diagnostics

`config_fixed.json` uses a fixed quantizer plus nvCOMP ANS pipeline.
`config_bitpack_diagnostic.json` substitutes bitpack for transport diagnosis.
`compression: null` runs the upstream connector without compression. For a
single diagnostic run, `KVSERVE_WIRE_DEBUG=1` adds SEND, PRE_SEND, POST_SEND,
POST_RECV, and RECV header traces to the engine logs. Leave this unset for
performance measurements because it synchronizes GPU buffers to copy headers.
