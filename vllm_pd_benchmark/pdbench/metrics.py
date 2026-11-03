from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .models import RequestResult


PERCENTILES = (50, 90, 95, 99)


def percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stats(prefix: str, values: Iterable[float | None]) -> dict[str, float | None]:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    row: dict[str, float | None] = {f"{prefix}_mean": sum(valid) / len(valid) if valid else None}
    row.update({f"{prefix}_p{p}": percentile(valid, p) for p in PERCENTILES})
    return row


def summarize_group(scope: str, name: str, results: list[RequestResult]) -> dict[str, Any]:
    successful = [item for item in results if item.status == "ok"]
    if results:
        duration = max(item.end_offset_s for item in results) - min(item.start_offset_s for item in results)
    else:
        duration = 0.0
    duration = max(duration, 0.0)
    input_tokens = sum(item.input_tokens for item in successful)
    output_tokens = sum(item.output_tokens for item in successful)
    row: dict[str, Any] = {
        "scope": scope,
        "name": name,
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "error_rate": (len(results) - len(successful)) / len(results) if results else None,
        "duration_s": duration,
        "request_throughput_rps": len(successful) / duration if duration else None,
        "offered_request_throughput_rps": len(results) / duration if duration else None,
        "input_token_throughput_tps": input_tokens / duration if duration else None,
        "output_token_throughput_tps": output_tokens / duration if duration else None,
        "total_token_throughput_tps": (input_tokens + output_tokens) / duration if duration else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    row.update(_stats("ttft_s", (item.ttft_s for item in successful)))
    row.update(_stats("tpot_s", (item.tpot_s for item in successful)))
    row.update(_stats("e2e_s", (item.e2e_s for item in successful)))
    row.update(_stats("input_tokens", (float(item.input_tokens) for item in successful)))
    row.update(_stats("output_tokens", (float(item.output_tokens) for item in successful)))
    return row


def build_summaries(results: list[RequestResult]) -> list[dict[str, Any]]:
    rows = [summarize_group("overall", "all", results)]
    datasets = sorted({item.dataset for item in results})
    for dataset in datasets:
        subset = [item for item in results if item.dataset == dataset]
        rows.append(summarize_group("dataset", dataset, subset))
        tasks = sorted({item.task for item in subset})
        for task in tasks:
            rows.append(summarize_group("task", f"{dataset}/{task}", [item for item in subset if item.task == task]))
    return rows


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_requests(path: Path, results: list[RequestResult], save_output: bool) -> None:
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("extra", None)
        if not save_output:
            row.pop("output_text", None)
        rows.append(row)
    _write_dict_rows(path, rows)


def write_summary(path: Path, results: list[RequestResult]) -> list[dict[str, Any]]:
    rows = build_summaries(results)
    _write_dict_rows(path, rows)
    return rows


def write_timeseries(path: Path, results: list[RequestResult], bucket_seconds: float) -> None:
    if not results:
        return
    final_time = max(item.end_offset_s for item in results)
    bucket_count = max(1, math.floor(final_time / bucket_seconds) + 1)
    rows = []
    for bucket in range(bucket_count):
        start = bucket * bucket_seconds
        end = start + bucket_seconds
        started = [item for item in results if start <= item.start_offset_s < end]
        completed = [item for item in results if start <= item.end_offset_s < end]
        successful = [item for item in completed if item.status == "ok"]
        active = sum(1 for item in results if item.start_offset_s < end and item.end_offset_s >= start)
        input_tokens = sum(item.input_tokens for item in successful)
        output_tokens = sum(item.output_tokens for item in successful)
        rows.append({
            "bucket_start_s": start,
            "bucket_end_s": end,
            "requests_started": len(started),
            "requests_completed": len(completed),
            "successful_requests": len(successful),
            "failed_requests": len(completed) - len(successful),
            "active_requests": active,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "request_throughput_rps": len(successful) / bucket_seconds,
            "output_token_throughput_tps": output_tokens / bucket_seconds,
        })
    _write_dict_rows(path, rows)

