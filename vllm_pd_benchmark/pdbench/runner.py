from __future__ import annotations

import asyncio
import random
from typing import Any

from .client import StreamingClient
from .models import RequestResult, Sample


async def _run_closed_loop(
    samples: list[Sample],
    concurrency: int,
    batch_size: int,
    submit: Any,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    pending: set[asyncio.Task[RequestResult]] = set()
    next_index = 0
    while next_index < len(samples) or pending:
        capacity = concurrency - len(pending)
        while capacity > 0 and next_index < len(samples):
            count = min(capacity, batch_size, len(samples) - next_index)
            for _ in range(count):
                index = next_index
                pending.add(asyncio.create_task(submit(index, samples[index])))
                next_index += 1
            capacity = concurrency - len(pending)
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        results.extend(task.result() for task in done)
    return results


async def _run_bursts(
    samples: list[Sample],
    batch_size: int,
    submit: Any,
) -> list[RequestResult]:
    results = []
    for start in range(0, len(samples), batch_size):
        tasks = [submit(index, samples[index]) for index in range(start, min(start + batch_size, len(samples)))]
        results.extend(await asyncio.gather(*tasks))
    return results


async def _run_poisson(
    samples: list[Sample],
    rate: float,
    seed: int,
    submit: Any,
) -> list[RequestResult]:
    rng = random.Random(seed)

    async def delayed(index: int, sample: Sample, delay: float) -> RequestResult:
        await asyncio.sleep(delay)
        return await submit(index, sample)

    elapsed = 0.0
    tasks = []
    for index, sample in enumerate(samples):
        if index:
            elapsed += rng.expovariate(rate)
        tasks.append(asyncio.create_task(delayed(index, sample, elapsed)))
    return list(await asyncio.gather(*tasks))


async def run_benchmark(
    config: dict[str, Any],
    samples: list[Sample],
    client: StreamingClient,
    run_origin: float,
) -> list[RequestResult]:
    bench = config["benchmark"]
    concurrency = int(bench["concurrency"])
    batch_size = int(bench["client_batch_size"])
    semaphore = asyncio.Semaphore(concurrency)

    async def submit(index: int, sample: Sample) -> RequestResult:
        async with semaphore:
            return await client.request(index, sample, run_origin)

    mode = config.get("load", {}).get("mode", "closed_loop")
    if mode == "closed_loop":
        results = await _run_closed_loop(samples, concurrency, batch_size, submit)
    elif mode == "burst":
        results = await _run_bursts(samples, batch_size, submit)
    else:
        results = await _run_poisson(
            samples,
            float(config["load"]["request_rate"]),
            int(bench.get("seed", 0)),
            submit,
        )
    return sorted(results, key=lambda item: item.request_id)


async def run_warmup(
    count: int,
    samples: list[Sample],
    client: StreamingClient,
) -> list[RequestResult]:
    import time

    if count <= 0:
        return []
    origin = time.perf_counter()
    warm_samples = [samples[index % len(samples)] for index in range(count)]
    semaphore = asyncio.Semaphore(int(client.bench["concurrency"]))

    async def submit(index: int, sample: Sample) -> RequestResult:
        async with semaphore:
            return await client.request(-(index + 1), sample, origin)

    return list(await asyncio.gather(*[
        submit(index, sample) for index, sample in enumerate(warm_samples)
    ]))

