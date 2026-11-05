"""Build a KVServe controller library from upstream search results and local GPU timings.

Run this once with the serving processes stopped. The upstream search result
supplies candidate configurations and quality scores; this script measures the
complete upstream compression/decompression path on a real Qwen KV sample.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_RESULTS = (
    "kvserve_v1/offline_search/evaluation/param_search/results/"
    "Qwen2.5-7B-Instruct/tolerance_3_results.json"
)


def model_snapshot(cache: str) -> Path:
    from start_pd import resolve_model

    return resolve_model(cache)


def collect_kv(model_path: Path, token_count: int):
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    text = (
        "This document describes a distributed language model service and "
        "the movement of key and value tensors between its stages. "
    ) * token_count
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :token_count]
    if input_ids.shape[1] != token_count:
        raise ValueError(f"Tokenized sample has {input_ids.shape[1]} tokens, expected {token_count}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16, trust_remote_code=False,
        local_files_only=True,
    ).to("cuda:0").eval()
    with torch.inference_mode():
        output = model(input_ids=input_ids.to("cuda:0"), use_cache=True)
    cache = output.past_key_values
    if hasattr(cache, "to_legacy_cache"):
        layers = cache.to_legacy_cache()
    elif hasattr(cache, "key_cache"):
        layers = zip(cache.key_cache, cache.value_cache)
    else:
        layers = cache

    blocks = []
    for layer in layers:
        key, value = layer[0], layer[1]
        if key.ndim != 4 or key.shape[0] != 1 or key.shape != value.shape:
            raise ValueError(f"Unexpected model KV shape: {tuple(key.shape)}")
        # HF: [batch, heads, tokens, dim]; connector: [2, blocks, 16, heads, dim].
        key = key[0].transpose(0, 1).contiguous()
        value = value[0].transpose(0, 1).contiguous()
        padding = (-key.shape[0]) % 16
        if padding:
            key = functional.pad(key, (0, 0, 0, 0, 0, padding))
            value = functional.pad(value, (0, 0, 0, 0, 0, padding))
        count = key.shape[0] // 16
        blocks.append(torch.stack((
            key.reshape(count, 16, key.shape[1], key.shape[2]),
            value.reshape(count, 16, value.shape[1], value.shape[2]),
        )))
    sample = torch.stack(blocks).contiguous()
    del output, cache, layers, model, blocks
    torch.cuda.empty_cache()
    return sample


def candidate_config(row: dict, model_name: str) -> dict:
    if row.get("transform_type") != "hadamard":
        raise ValueError("Only upstream Hadamard search results are supported")
    if row.get("axis_key") != [2] or row.get("axis_value") != [1, 3]:
        raise ValueError("Unsupported upstream search axes; inspect this candidate")
    fields = (
        "high_key_max_value", "high_value_max_value",
        "low_key_max_value", "low_value_max_value",
    )
    return {
        "enabled": True,
        "pipeline": ["transformer", "quantizer", "codec"],
        "transformer_config": {"transform_type": "hadamard", "seed": 0x3333},
        "quantizer_config": {
            "model_name": model_name,
            "hybrid_ratio": float(row["heads_selection"]),
            **{field: int(row[field]) for field in fields},
            "axis_key": "channel",
            "axis_value": "token",
            "split_type": "head",
        },
        "codec_config": {
            "codec_type": "nvcomp", "nvcomp_algorithm": "ANS", "data_type": "|u1"
        },
        "min_compress_size": 1024,
    }


def measure(sample, config: dict, repeats: int, warmup: int) -> dict:
    import torch
    from kvserve_v1.compression.manager import KVCompressionAdapter
    from kvserve_v1.compression.wire import build_wire

    adapter = KVCompressionAdapter(
        config, num_kv_heads=sample.shape[-2], head_size=sample.shape[-1],
        model_name=config["quantizer_config"]["model_name"],
    )
    encode_times = []
    decode_times = []
    wire_bytes = None
    for index in range(warmup + repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        compressed = adapter.compress(sample, f"profile-{index}")
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        if compressed is None:
            raise RuntimeError("KVServe compression returned no payload")
        wire = build_wire(compressed, 512 * 1024 * 1024)
        restored = adapter.decompress(compressed)
        torch.cuda.synchronize()
        decoded = time.perf_counter()
        if restored is None or restored.shape != sample.shape:
            raise RuntimeError("KVServe decompression failed during profiling")
        if index >= warmup:
            encode_times.append(encoded - started)
            decode_times.append(decoded - encoded)
            wire_bytes = wire.nbytes
        del compressed, wire, restored
    original_bytes = sample.numel() * sample.element_size()
    encode_s = statistics.median(encode_times)
    decode_s = statistics.median(decode_times)
    original_mib = original_bytes / (1024 * 1024)
    return {
        "compression_ratio": original_bytes / wire_bytes,
        "harmonic_speed_mbps": original_mib / (encode_s + decode_s),
        "encode_mbps": original_mib / encode_s,
        "decode_mbps": original_mib / decode_s,
        "original_bytes": original_bytes,
        "wire_bytes": wire_bytes,
    }


def make_intervals(profiles: list[dict]) -> list[dict]:
    # For x=1/B, each profile's transfer+codec time per MB is 1/S+x/cr.
    cuts = {0.0, 1.0}
    for left, right in combinations(profiles, 2):
        lp, rp = left["performance_metrics"], right["performance_metrics"]
        slope_delta = 1 / lp["compression_ratio"] - 1 / rp["compression_ratio"]
        if abs(slope_delta) < 1e-15:
            continue
        crossing = (1 / rp["harmonic_speed_mbps"] -
                    1 / lp["harmonic_speed_mbps"]) / slope_delta
        if 0 < crossing < 1:
            cuts.add(crossing)
    ordered = sorted(cuts)
    intervals = []
    for index, (lower, upper) in enumerate(zip(ordered, ordered[1:])):
        midpoint = (lower + upper) / 2
        ranked = sorted(profiles, key=lambda item: (
            1 / item["performance_metrics"]["harmonic_speed_mbps"] +
            midpoint / item["performance_metrics"]["compression_ratio"]
        ))
        intervals.append({
            "interval_id": index,
            "x_range": [lower, upper],
            "candidate_profile_ids": [p["profile_id"] for p in ranked[:3]],
            "model_optimal_profile_id": ranked[0]["profile_id"],
        })
    return intervals


def build_library(profiles: list[dict], model_name: str, tokens: int,
                  source: Path) -> dict:
    scores = sorted({p["quality_metrics"]["accuracy"] for p in profiles})
    boundaries = [0.0] + [math.nextafter(score, math.inf) for score in scores]
    buckets = []
    for index, (lower, upper) in enumerate(zip(boundaries, boundaries[1:])):
        eligible = [p for p in profiles if p["quality_metrics"]["accuracy"] >= lower]
        if eligible:
            buckets.append({
                "bucket_id": index,
                "acc_range": [lower, upper],
                "pareto_profiles": eligible,
                "bandwidth_intervals": make_intervals(eligible),
            })
    return {
        "library_metadata": {
            "model_name": model_name,
            "dataset": "upstream offline search tasks",
            "accuracy_unit": "upstream search score; use the same unit for accuracy_requirement",
            "speed_source": "local real-model KV sample, one GPU",
            "sample_tokens": tokens,
            "search_results": str(source),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "accuracy_buckets": buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--search-results", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "profiles/qwen2.5-7b.json")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.input_tokens < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("input-tokens and repeats must be positive; warmup must be nonnegative")

    import kvserve_v1
    from kvserve_v1.compression.controller.profile_library import ProfileLibrary

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = "Qwen2.5-7B-Instruct"
    source = args.search_results or Path(kvserve_v1.__file__).resolve().parent / SOURCE_RESULTS.removeprefix("kvserve_v1/")
    search_rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(search_rows, list) or not search_rows:
        raise ValueError("Offline search results must be a nonempty JSON list")
    print(f"Loading model and collecting {args.input_tokens} real KV tokens...", flush=True)
    sample = collect_kv(model_snapshot(config["model_cache"]), args.input_tokens)
    print(f"KV sample shape: {tuple(sample.shape)}", flush=True)
    profiles = []
    for index, row in enumerate(search_rows):
        candidate = candidate_config(row, model_name)
        print(f"Measuring profile {index + 1}/{len(search_rows)}...", flush=True)
        metrics = measure(sample, candidate, args.repeats, args.warmup)
        profiles.append({
            "profile_id": f"{model_name}-search-{row.get('config_id', index)}",
            "metadata": {"model_name": model_name, "source_config_id": row.get("config_id", index),
                         "upstream_compression_ratio": row.get("cr")},
            "compression_config": candidate,
            "performance_metrics": metrics,
            "quality_metrics": {"accuracy": float(row["accuracy"])},
        })
    library = build_library(profiles, model_name, args.input_tokens, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(library, indent=2) + "\n", encoding="utf-8")
    loaded = ProfileLibrary(str(temporary))
    temporary.replace(args.output)
    print(f"Saved {loaded.get_num_profiles()} controller profiles: {args.output}", flush=True)


if __name__ == "__main__":
    main()
