
import os
import sys
import time
import json
import signal
import socket
import shutil
import subprocess
import traceback
import importlib.util
import importlib.metadata
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


# ============================================================
# 1. 基础配置
# ============================================================

MODEL_CACHE = (
    "/home/gzw/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
)

VLLM_SOURCE = Path.home() / "vllm-0.30.0"

LOG_DIR = Path(__file__).resolve().parent / "pd_logs"

MODEL_NAME = "qwen2.5-7b"

PREFILL_PORT = 8100
DECODE_PORT = 8200
PROXY_PORT = 8192

PREFILL_NIXL_PORT = 5600
DECODE_NIXL_PORT = 5601

MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 4
GPU_MEMORY_UTILIZATION = 0.80

STARTUP_TIMEOUT = 900


# ============================================================
# 2. 检查环境
# ============================================================

def check_environment():

    print("\n========== 检查环境 ==========")

    version = importlib.metadata.version("vllm")

    print("vLLM version:", version)

    if version != "0.30.0":
        raise RuntimeError(
            f"要求 vLLM 0.30.0，当前版本为 {version}"
        )

    if importlib.util.find_spec("nixl") is None:
        raise RuntimeError(
            "没有安装 NIXL，请先执行：uv pip install nixl"
        )

    for package in ["fastapi", "httpx", "uvicorn"]:
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(
                f"缺少依赖 {package}，请先安装"
            )

    if shutil.which("numactl") is None:
        raise RuntimeError("没有找到 numactl")

    vllm_bin = shutil.which("vllm")

    if vllm_bin is None:
        raise RuntimeError("没有找到 vllm 命令")

    print("NIXL: OK")
    print("numactl: OK")
    print("vllm executable:", vllm_bin)

    return vllm_bin


# ============================================================
# 3. 获取 HuggingFace 模型快照路径
# ============================================================

def get_model_path():

    refs_main = Path(MODEL_CACHE) / "refs" / "main"

    if refs_main.exists():

        revision = refs_main.read_text().strip()

        model_path = Path(MODEL_CACHE) / "snapshots" / revision

    else:

        snapshots = Path(MODEL_CACHE) / "snapshots"

        if not snapshots.exists():
            raise RuntimeError("没有找到模型 snapshots 目录")

        candidates = [
            p for p in snapshots.iterdir() if p.is_dir()
        ]

        if not candidates:
            raise RuntimeError("没有找到模型快照")

        model_path = max(
            candidates,
            key=lambda p: p.stat().st_mtime
        )

    if not (model_path / "config.json").exists():
        raise RuntimeError(
            f"模型路径无效: {model_path}"
        )

    print("\n模型路径:", model_path)

    return str(model_path)


# ============================================================
# 4. 获取 vLLM 官方 Proxy
# ============================================================

def get_proxy_script():

    proxy = (
        VLLM_SOURCE
        / "tests/v1/kv_connector/nixl_integration/"
        / "toy_proxy_server.py"
    )

    if proxy.exists():
        return proxy

    if VLLM_SOURCE.exists():
        raise RuntimeError(
            f"源码目录存在但没有 Proxy 脚本: {proxy}\n"
            "请检查源码版本是否为 v0.30.0"
        )

    print("\n下载 vLLM 0.30.0 官方源码...")

    if shutil.which("git") is None:
        raise RuntimeError("没有安装 git")

    subprocess.run(
        [
            "git",
            "clone",
            "--depth", "1",
            "--branch", "v0.30.0",
            "https://github.com/vllm-project/vllm.git",
            str(VLLM_SOURCE)
        ],
        check=True
    )

    if not proxy.exists():
        raise RuntimeError("下载源码后仍然找不到 Proxy 脚本")

    return proxy


# ============================================================
# 5. 检查端口
# ============================================================

def check_ports():

    ports = [
        PREFILL_PORT,
        DECODE_PORT,
        PROXY_PORT,
        PREFILL_NIXL_PORT,
        DECODE_NIXL_PORT
    ]

    for port in ports:

        with socket.socket() as s:

            try:
                s.bind(("127.0.0.1", port))

            except OSError:
                raise RuntimeError(
                    f"端口 {port} 已被占用"
                )

    print("\n端口检查通过")


# ============================================================
# 6. 启动进程
# ============================================================

processes = []
log_files = []


def start_process(name, command, env):

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"{name}.log"

    log_file = open(
        log_path,
        "a",
        buffering=1
    )

    log_files.append(log_file)

    print(f"\n启动 {name}")
    print("日志:", log_path)
    print("命令:", " ".join(map(str, command)))

    proc = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

    processes.append((name, proc))

    return proc


# ============================================================
# 7. 等待服务就绪
# ============================================================

def wait_http(name, proc, urls):

    print(f"\n等待 {name} 启动...")

    start_time = time.time()

    while time.time() - start_time < STARTUP_TIMEOUT:

        if proc.poll() is not None:
            raise RuntimeError(
                f"{name} 启动失败，退出码 {proc.returncode}，"
                f"请查看 pd_logs/{name}.log"
            )

        for url in urls:

            try:
                with urlopen(url, timeout=3) as response:

                    if response.status == 200:
                        print(f"{name} 已就绪: {url}")
                        return

            except Exception:
                pass

        time.sleep(3)

    raise TimeoutError(
        f"{name} 启动超时，请查看 pd_logs/{name}.log"
    )


# ============================================================
# 8. 启动 Prefill 和 Decode
# ============================================================

def start_vllm(vllm_bin, model_path):

    def kv_config(role):
        return json.dumps({
            "kv_connector": "NixlConnector",
            "kv_role": role,
            "kv_load_failure_policy": "fail"
        })

    common_args = [
        "serve",
        model_path,
        "--host", "127.0.0.1",
        "--dtype", "half",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--enforce-eager",
    ]

    # ---------------- Prefill ----------------

    p_env = os.environ.copy()

    p_env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "UCX_NET_DEVICES": "all",
        "VLLM_NIXL_SIDE_CHANNEL_PORT":
            str(PREFILL_NIXL_PORT)
    })

    p_command = [
        "numactl",
        "--cpunodebind=0",
        "--membind=0",
        vllm_bin,
        *common_args,
        "--port", str(PREFILL_PORT),
        "--enable-chunked-prefill",
        "--kv-transfer-config", kv_config("kv_producer"),
        "--served-model-name", MODEL_NAME
    ]

    p_proc = start_process(
        "prefill",
        p_command,
        p_env
    )

    # ---------------- Decode ----------------

    d_env = os.environ.copy()

    d_env.update({
        "CUDA_VISIBLE_DEVICES": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "UCX_NET_DEVICES": "all",
        "VLLM_NIXL_SIDE_CHANNEL_PORT":
            str(DECODE_NIXL_PORT)
    })

    d_command = [
        "numactl",
        "--cpunodebind=1",
        "--membind=1",
        vllm_bin,
        *common_args,
        "--port", str(DECODE_PORT),
        "--kv-transfer-config", kv_config("kv_consumer"),
        "--served-model-name", MODEL_NAME
    ]

    d_proc = start_process(
        "decode",
        d_command,
        d_env
    )

    # 等待两个实例都完成模型加载

    wait_http(
        "prefill",
        p_proc,
        [f"http://127.0.0.1:{PREFILL_PORT}/health"]
    )

    wait_http(
        "decode",
        d_proc,
        [f"http://127.0.0.1:{DECODE_PORT}/health"]
    )


# ============================================================
# 9. 启动 Proxy
# ============================================================

def start_proxy(proxy_script):

    env = os.environ.copy()

    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        str(proxy_script),
        "--host", "127.0.0.1",
        "--port", str(PROXY_PORT),
        "--prefiller-hosts", "localhost",
        "--prefiller-ports", str(PREFILL_PORT),
        "--decoder-hosts", "localhost",
        "--decoder-ports", str(DECODE_PORT)
    ]

    proc = start_process(
        "proxy",
        command,
        env
    )

    wait_http(
        "proxy",
        proc,
        [f"http://127.0.0.1:{PROXY_PORT}/healthcheck"]
    )


# ============================================================
# 10. 关闭所有进程
# ============================================================

def cleanup():

    print("\n正在关闭 PD 分离服务...")

    for name, proc in reversed(processes):

        if proc.poll() is None:

            print("停止:", name)

            try:
                os.killpg(proc.pid, signal.SIGTERM)

            except ProcessLookupError:
                pass

    for name, proc in reversed(processes):

        if proc.poll() is None:

            try:
                proc.wait(timeout=20)

            except subprocess.TimeoutExpired:

                print("强制关闭:", name)

                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

                proc.wait()

    for f in log_files:
        f.close()

    print("所有服务已关闭")


# ============================================================
# 11. 主函数
# ============================================================

def main():

    try:

        print("=" * 60)
        print("       vLLM PD Disaggregated Serving")
        print("=" * 60)

        vllm_bin = check_environment()

        model_path = get_model_path()

        proxy_script = get_proxy_script()

        check_ports()

        print("\n========== 启动 PD 服务 ==========")

        start_vllm(vllm_bin, model_path)

        start_proxy(proxy_script)

        print("\n" + "=" * 60)
        print("PD 分离服务启动成功")
        print("=" * 60)

        print(f"""
Prefill API : http://127.0.0.1:{PREFILL_PORT}
Decode API  : http://127.0.0.1:{DECODE_PORT}
Proxy API   : http://127.0.0.1:{PROXY_PORT}

日志目录: {LOG_DIR}

测试命令:

curl http://127.0.0.1:{PROXY_PORT}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{MODEL_NAME}",
    "messages": [
      {{"role": "user", "content": "介绍一下 Prefill 和 Decode"}}
    ],
    "max_tokens": 128,
    "temperature": 0
  }}'

按 Ctrl+C 停止所有服务。
""")

        while True:

            for name, proc in processes:

                if proc.poll() is not None:

                    raise RuntimeError(
                        f"{name} 进程意外退出，"
                        f"退出码: {proc.returncode}"
                    )

            time.sleep(2)

    except KeyboardInterrupt:

        print("\n收到 Ctrl+C")

    except Exception as e:

        print("\n启动失败:", repr(e))
        traceback.print_exc()

    finally:

        cleanup()


if __name__ == "__main__":
    main()

