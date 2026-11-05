"""Start the isolated KVServe prefill/decode baseline on the native PD ports."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("prefill_port", "decode_port", "proxy_port", "kv_port"):
        if not 1 <= int(config[key]) <= 65535:
            raise ValueError(f"Invalid {key}: {config[key]}")
    ports = [int(config[key]) for key in ("prefill_port", "decode_port", "proxy_port", "kv_port")]
    if len(set(ports)) != len(ports):
        raise ValueError("The four service ports must differ")
    if config["prefill_gpu"] == config["decode_gpu"]:
        raise ValueError("Prefill and decode need distinct GPUs")
    return config


def resolve_model(cache: str) -> Path:
    path = Path(cache).expanduser()
    if (path / "config.json").is_file():
        return path.resolve()
    snapshots = path / "snapshots"
    ref = path / "refs" / "main"
    if ref.is_file():
        candidate = snapshots / ref.read_text(encoding="utf-8").strip()
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    if snapshots.is_dir():
        candidates = [p for p in snapshots.iterdir() if (p / "config.json").is_file()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime).resolve()
    raise FileNotFoundError(f"Model snapshot not found under {path}")


def check_runtime(config: dict) -> Path:
    version = importlib.metadata.version("vllm")
    if version != config["vllm_version"]:
        raise RuntimeError(f"Expected vLLM {config['vllm_version']}, got {version}")
    spec = importlib.util.find_spec("kvserve_v1")
    if spec is None or spec.origin is None:
        raise RuntimeError("KVServe is absent from this Python environment; run setup_env.sh")
    import kvserve_v1.connector.compressed_kv_connector  # noqa: F401
    source = Path(spec.origin).resolve().parents[1]
    if not (source / ".git").exists():
        raise RuntimeError("Install the pinned KVServe checkout in editable mode")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != config["kvserve_revision"]:
        raise RuntimeError(f"Expected KVServe {config['kvserve_revision']}, got {revision}")
    connector_source = source / "kvserve_v1/connector/compressed_kv_connector.py"
    if "kv_cache[forward_context.virtual_engine]" in connector_source.read_text(encoding="utf-8"):
        raise RuntimeError("KVServe needs the vLLM 0.30 connector patch; rerun kvserve_baseline/setup.sh")

    vllm_bin = Path(sys.executable).parent / "vllm"
    if not vllm_bin.is_file():
        raise RuntimeError(f"vllm executable not found beside {sys.executable}")
    if any(config.get(key) is not None for key in ("prefill_numa_node", "decode_numa_node")):
        if shutil.which("numactl") is None:
            raise RuntimeError("numactl is required for the configured NUMA binding")
    return vllm_bin


def check_ports(config: dict) -> None:
    for key in ("prefill_port", "decode_port", "proxy_port", "kv_port"):
        with socket.socket() as sock:
            try:
                sock.bind((config["host"], int(config[key])))
            except OSError as exc:
                raise RuntimeError(f"Port {config[key]} ({key}) is already in use") from exc


def kv_config(config: dict, producer: bool) -> str:
    return json.dumps(
        {
            "kv_connector": "CompressedKVConnector",
            "kv_connector_module_path": "kvserve_v1.connector.compressed_kv_connector",
            "kv_role": "kv_producer" if producer else "kv_consumer",
            "kv_rank": 0 if producer else 1,
            "kv_parallel_size": 2,
            "kv_ip": config["host"],
            "kv_port": int(config["kv_port"]),
            "kv_buffer_size": int(config["kv_buffer_size"]),
            "kv_connector_extra_config": {"compression": config["compression"]},
        },
        separators=(",", ":"),
    )


def backend_command(config: dict, model: Path, vllm_bin: Path, producer: bool) -> list[str]:
    node = config["prefill_numa_node"] if producer else config["decode_numa_node"]
    port = config["prefill_port"] if producer else config["decode_port"]
    command = []
    if node is not None:
        command += ["numactl", f"--cpunodebind={node}", f"--membind={node}"]
    command += [
        str(vllm_bin), "serve", str(model),
        "--host", config["host"],
        "--port", str(port),
        "--served-model-name", config["model_name"],
        "--dtype", config["dtype"],
        "--max-model-len", str(config["max_model_len"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "--kv-transfer-config", kv_config(config, producer),
    ]
    return command


def wait_ready(name: str, process: subprocess.Popen, url: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited with code {process.returncode}; check its log")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print(f"{name} ready: {url}", flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{name} did not become ready: {url}; check its log")


def run(config: dict, dry_run: bool) -> None:
    model = resolve_model(config["model_cache"])
    vllm_bin = check_runtime(config)
    log_dir = (HERE / config["log_dir"]).resolve()
    host = config["host"]
    # NCCL initialization is collective: start both sides before waiting for either.
    commands = [
        ("decode", backend_command(config, model, vllm_bin, False), config["decode_gpu"], config["decode_port"]),
        ("prefill", backend_command(config, model, vllm_bin, True), config["prefill_gpu"], config["prefill_port"]),
    ]
    proxy_command = [
        sys.executable, str(HERE / "proxy.py"),
        "--host", host,
        "--port", str(config["proxy_port"]),
        "--prefill-port", str(config["prefill_port"]),
        "--decode-port", str(config["decode_port"]),
        "--backend-timeout", str(config["backend_timeout_seconds"]),
    ]
    print(f"Model: {model}\nvLLM: {config['vllm_version']}\nKVServe: {config['kvserve_revision']}")
    print(f"Compression: {json.dumps(config['compression'], ensure_ascii=False)}")
    if dry_run:
        for name, command, gpu, _ in commands:
            print(f"{name} CUDA_VISIBLE_DEVICES={gpu}: {subprocess.list2cmdline(command)}")
        print(f"proxy: {subprocess.list2cmdline(proxy_command)}")
        return

    check_ports(config)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen]] = []
    logs = []

    def start(name: str, command: list[str], gpu: str | None = None) -> subprocess.Popen:
        output = open(log_dir / f"{name}.log", "a", encoding="utf-8", buffering=1)
        logs.append(output)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        process = subprocess.Popen(
            command, env=env, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((name, process))
        print(f"Started {name} (PID {process.pid}); log: {log_dir / (name + '.log')}", flush=True)
        return process

    try:
        for name, command, gpu, _ in commands:
            start(name, command, gpu)
        for (name, _, _, port), (_, process) in zip(commands, processes):
            wait_ready(name, process, f"http://{host}:{port}/health", config["startup_timeout_seconds"])
        process = start("proxy", proxy_command)
        wait_ready("proxy", process, f"http://{host}:{config['proxy_port']}/healthcheck", 30)
        print(f"KVServe API: http://{host}:{config['proxy_port']}/v1", flush=True)
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited with code {process.returncode}")
            time.sleep(2)
    finally:
        for _, process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in reversed(processes):
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for output in logs:
            output.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run(load_config(args.config), args.dry_run)
    except KeyboardInterrupt:
        print("KVServe services stopped", flush=True)


if __name__ == "__main__":
    main()

