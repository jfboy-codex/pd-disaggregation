from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config, config_path


def validate_config(config: dict[str, Any]) -> None:
    for section in ("backend", "benchmark", "datasets", "output"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    backend = config["backend"]
    for key in ("base_url", "model"):
        if not backend.get(key):
            raise ValueError(f"backend.{key} must be set")

    bench = config["benchmark"]
    for key in ("request_count", "concurrency", "client_batch_size", "max_output_tokens"):
        if int(bench.get(key, 0)) <= 0:
            raise ValueError(f"benchmark.{key} must be greater than zero")
    if int(bench.get("warmup_requests", 0)) < 0:
        raise ValueError("benchmark.warmup_requests cannot be negative")

    load = config.get("load", {})
    mode = load.get("mode", "closed_loop")
    if mode not in {"closed_loop", "burst", "poisson"}:
        raise ValueError("load.mode must be closed_loop, burst, or poisson")
    if mode == "poisson" and float(load.get("request_rate", 0)) <= 0:
        raise ValueError("load.request_rate must be positive in poisson mode")
    strategy = config.get("sampling", {}).get("strategy", "round_robin")
    if strategy not in {"round_robin", "shuffled", "sequential"}:
        raise ValueError("sampling.strategy must be round_robin, shuffled, or sequential")

    enabled = [item for item in config["datasets"] if item.get("enabled", True)]
    if not enabled:
        raise ValueError("At least one dataset must be enabled")
    supported = {"sharegpt", "gsm8k", "longbench", "humaneval"}
    unknown = {item.get("name") for item in enabled} - supported
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")


def snapshot_config(config: dict[str, Any], destination: Path) -> None:
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


