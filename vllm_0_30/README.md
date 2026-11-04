# Native vLLM 0.30.0 PD baseline

Activate the Conda environment that previously ran the native baseline, then:

```bash
cd vllm_0_30
bash setup.sh
python start_pd.py
```

`setup.sh` upgrades vLLM to 0.30.0 **in the active Conda environment** and
installs NIXL and proxy dependencies from the Tsinghua PyPI mirror. It does not
create a virtual environment. Check that the server's Python, CUDA driver and
system libraries can run the vLLM 0.30.0 wheel before upgrading. The launcher
checks the installed version and resolves the existing Qwen model cache.

The vLLM 0.30 NIXL configuration uses `kv_producer` for prefill and
`kv_consumer` for decode; `kv_both` is deprecated. The prefill instance enables
chunked prefill, which vLLM 0.30 lists as compatible with NIXL PD. The proxy
still listens on `127.0.0.1:8192`, with P on `8100` and D on `8200`.
`start_pd.py` downloads the matching vLLM source from GitHub if the official
toy proxy script is not already under `~/vllm-0.30.0`. Python packages use
the Tsinghua mirror; this Git checkout uses GitHub.

The benchmark can keep `http://127.0.0.1:8192/v1`. Stop this baseline before
starting KVServe because they share the same ports. For comparable results,
record this baseline's vLLM version and its chunked-prefill setting alongside
the KVServe configuration.
