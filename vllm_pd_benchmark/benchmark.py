from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdbench import __version__
from pdbench.client import StreamingClient
from pdbench.config import load_config, snapshot_config, validate_config
from pdbench.datasets import load_and_sample
from pdbench.metrics import write_requests, write_summary, write_timeseries
from pdbench.runner import run_benchmark, run_warmup
from pdbench.tokenizer import TokenCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible vLLM streaming endpoint")
    parser.add_argument("--config", default="config.json", help="Path to JSON configuration")
    parser.add_argument("--output", help="Override output.directory")
    parser.add_argument("--requests", type=int, help="Override per-dataset benchmark.request_count")
    parser.add_argument("--concurrency", type=int, help="Override benchmark.concurrency")
    parser.add_argument("--seed", type=int, help="Override benchmark.seed")
    parser.add_argument("--dry-run", action="store_true", help="Load/tokenize data and write metadata, but send no requests")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output:
        config["output"]["directory"] = args.output
    if args.requests is not None:
        config["benchmark"]["request_count"] = args.requests
    if args.concurrency is not None:
        config["benchmark"]["concurrency"] = args.concurrency
    if args.seed is not None:
        config["benchmark"]["seed"] = args.seed


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


async def async_main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    apply_overrides(config, args)
    validate_config(config)

    samples, source_counts = load_and_sample(config)
    print(f"Loaded source samples: {source_counts}")
    print(
        f"Selected {config['benchmark']['request_count']} requests per dataset, "
        f"{len(samples)} total, with seed {config['benchmark'].get('seed')}"
    )

    print("Loading tokenizer...")
    counter = TokenCounter(config.get("tokenizer", {}), config["backend"]["model"])
    input_counts = [counter.count_messages(sample.messages) for sample in samples]
    print(f"Tokenizer: {counter.source}")
    print(f"Input tokens: min={min(input_counts)}, max={max(input_counts)}, mean={sum(input_counts) / len(input_counts):.1f}")

    output_root = Path(config["output"]["directory"]).expanduser()
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{config['output'].get('run_name', 'benchmark')}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_config(config, run_dir / "config.snapshot.json")

    metadata: dict[str, Any] = {
        "tool_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_directory": str(run_dir),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "packages": {name: package_version(name) for name in ("httpx", "transformers", "tokenizers")},
        "tokenizer_source": counter.source,
        "source_sample_counts": source_counts,
        "selected_requests": len(samples),
        "selected_sample_trace": [
            {"request_id": index, "dataset": sample.dataset, "task": sample.task, "sample_id": sample.sample_id, "source_index": sample.source_index}
            for index, sample in enumerate(samples)
        ],
        "metric_definitions": {
            "ttft_s": "Client wall time from request start until the first nonempty generated content arrives in the SSE stream.",
            "e2e_s": "Client wall time from request start until the streaming response finishes.",
            "tpot_s": "For successful responses with >=2 locally tokenized output tokens: (E2E - TTFT) / (output_tokens - 1).",
            "token_counts": "Local tokenizer counts; input uses apply_chat_template(..., add_generation_prompt=True), output tokenizes generated text without special tokens.",
            "throughput": "Successful requests or their locally counted tokens divided by measured wall-clock span. No server-side KV/cache metric is inferred.",
        },
    }

    if args.dry_run:
        metadata["dry_run"] = True
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Dry run complete: {run_dir}")
        return 0

    async with StreamingClient(config, counter) as client:
        warmup_count = int(config["benchmark"].get("warmup_requests", 0))
        if warmup_count:
            print(f"Running {warmup_count} warmup requests...")
            warmup = await run_warmup(warmup_count, samples, client)
            warmup_failures = [item for item in warmup if item.status != "ok"]
            metadata["warmup"] = {
                "requests": len(warmup),
                "failures": len(warmup_failures),
                "errors": [item.error for item in warmup_failures],
            }
            if warmup_failures:
                print(f"Warning: {len(warmup_failures)} warmup request(s) failed")

        print(f"Starting measured run: mode={config.get('load', {}).get('mode', 'closed_loop')}, concurrency={config['benchmark']['concurrency']}")
        run_origin = time.perf_counter()
        results = await run_benchmark(config, samples, client, run_origin)
        run_finished = time.perf_counter()

    save_output = bool(config["benchmark"].get("save_output_text", True))
    write_requests(run_dir / "requests.csv", results, save_output)
    summary_rows = write_summary(run_dir / "summary.csv", results)
    if config["benchmark"].get("timeseries", True):
        write_timeseries(
            run_dir / "timeseries.csv",
            results,
            float(config["benchmark"].get("timeseries_bucket_seconds", 1.0)),
        )
    metadata["measured_wall_time_s"] = run_finished - run_origin
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["result_files"] = ["requests.csv", "summary.csv"] + (["timeseries.csv"] if config["benchmark"].get("timeseries", True) else [])
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = summary_rows[0]
    print(f"Completed: {overall['successful_requests']}/{overall['requests']} successful")
    print(f"Request throughput: {overall['request_throughput_rps'] or 0:.3f} req/s")
    print(f"Output throughput: {overall['output_token_throughput_tps'] or 0:.3f} tok/s")
    print(f"Results: {run_dir}")
    return 0 if overall["failed_requests"] == 0 else 2


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()


