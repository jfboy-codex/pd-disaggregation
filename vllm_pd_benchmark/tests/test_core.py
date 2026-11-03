from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pdbench.runner import run_benchmark

try:
    import httpx
    from pdbench.client import StreamingClient
except ModuleNotFoundError:
    httpx = None
    StreamingClient = None
from pdbench.datasets import load_gsm8k, load_humaneval, load_longbench, load_sharegpt
from pdbench.metrics import build_summaries
from pdbench.models import RequestResult
from pdbench.models import Sample


class DatasetTests(unittest.TestCase):
    def test_all_adapters(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as raw:
            root = Path(raw)
            sharegpt = root / "sharegpt.json"
            sharegpt.write_text(json.dumps([{"id": "s1", "conversations": [
                {"from": "human", "value": "hello"},
                {"from": "gpt", "value": "hi"},
                {"from": "human", "value": "continue"},
                {"from": "gpt", "value": "not part of prompt"},
            ]}]), encoding="utf-8")
            self.assertEqual(load_sharegpt({"path": str(sharegpt)})[0].messages[-1]["content"], "continue")

            gsm = root / "gsm.jsonl"
            gsm.write_text(json.dumps({"question": "1+1?", "answer": "2"}) + "\n", encoding="utf-8")
            self.assertIn("1+1?", load_gsm8k({"path": str(gsm)})[0].messages[0]["content"])

            human = root / "human.jsonl"
            human.write_text(json.dumps({"task_id": "x", "prompt": "def f():\n"}) + "\n", encoding="utf-8")
            self.assertEqual(load_humaneval({"path": str(human)})[0].sample_id, "x")

            long = root / "long"
            long.mkdir()
            (long / "narrativeqa.jsonl").write_text(json.dumps({"input": "why?", "context": "story"}) + "\n", encoding="utf-8")
            prompt = load_longbench({"path": str(long), "tasks": ["narrativeqa"]})[0].messages[0]["content"]
            self.assertIn("story", prompt)
            self.assertIn("why?", prompt)


class MetricTests(unittest.TestCase):
    def test_summary(self) -> None:
        result = RequestResult(
            request_id=0, sample_id="a", dataset="gsm8k", task="test", status="ok",
            start_offset_s=0.0, end_offset_s=2.0, ttft_s=0.5, e2e_s=2.0,
            tpot_s=0.5, input_tokens=10, output_tokens=4, total_tokens=14,
        )
        row = build_summaries([result])[0]
        self.assertEqual(row["successful_requests"], 1)
        self.assertEqual(row["request_throughput_rps"], 0.5)
        self.assertEqual(row["ttft_s_p99"], 0.5)


class _FakeCounter:
    def count_messages(self, messages):
        return 7

    def count_text(self, text):
        return len(text.split())


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_dataset_phases_wait_for_completion(self) -> None:
        events = []

        class FakeClient:
            async def request(self, request_id, sample, run_origin):
                events.append(("start", sample.dataset))
                events.append(("done", sample.dataset))
                return RequestResult(request_id, sample.sample_id, sample.dataset, sample.task, "ok")

        samples = [
            Sample(str(i), dataset, "task", [{"role": "user", "content": "q"}], i)
            for i, dataset in enumerate(("sharegpt", "sharegpt", "gsm8k", "gsm8k"))
        ]
        config = {
            "benchmark": {"concurrency": 2, "client_batch_size": 2, "seed": 1},
            "load": {"mode": "burst", "separate_datasets": True},
        }
        results = await run_benchmark(config, samples, FakeClient(), 0.0)
        self.assertEqual([result.request_id for result in results], [0, 1, 2, 3])
        first_gsm = events.index(("start", "gsm8k"))
        self.assertEqual(events[:first_gsm].count(("done", "sharegpt")), 2)


@unittest.skipUnless(httpx is not None, "httpx is not installed")
class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_sse_and_usage(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = "".join([
                'data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}\n\n',
                'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n',
                'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\n',
                "data: [DONE]\n\n",
            ])
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        config = {
            "backend": {"base_url": "http://test/v1", "model": "m", "timeout_seconds": 5},
            "benchmark": {"concurrency": 1, "max_output_tokens": 8, "retries": 0},
        }
        client = StreamingClient(config, _FakeCounter())
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            sample = Sample("s", "gsm8k", "test", [{"role": "user", "content": "q"}], 0)
            import time
            result = await client.request(0, sample, time.perf_counter())
        finally:
            await client.http.aclose()
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output_text, "hello world")
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.server_completion_tokens, 2)
        self.assertIsNotNone(result.ttft_s)
        self.assertIsNotNone(result.tpot_s)


if __name__ == "__main__":
    unittest.main()


