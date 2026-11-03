from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from .models import RequestResult, Sample
from .tokenizer import TokenCounter


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return ""


class StreamingClient:
    def __init__(self, config: dict[str, Any], token_counter: TokenCounter):
        backend = config["backend"]
        self.url = backend["base_url"].rstrip("/") + "/chat/completions"
        self.model = backend["model"]
        self.api_key = backend.get("api_key", "EMPTY")
        self.timeout = float(backend.get("timeout_seconds", 600))
        self.verify_tls = bool(backend.get("verify_tls", True))
        self.extra_body = dict(backend.get("extra_body", {}))
        self.bench = config["benchmark"]
        self.counter = token_counter
        self.http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "StreamingClient":
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            verify=self.verify_tls,
            limits=httpx.Limits(max_connections=int(self.bench["concurrency"]), max_keepalive_connections=int(self.bench["concurrency"])),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.http is not None:
            await self.http.aclose()

    async def request(
        self,
        request_id: int,
        sample: Sample,
        run_origin: float,
    ) -> RequestResult:
        retries = int(self.bench.get("retries", 0))
        last: RequestResult | None = None
        for attempt in range(1, retries + 2):
            last = await self._request_once(request_id, sample, run_origin, attempt)
            if last.status == "ok":
                return last
            if attempt <= retries:
                await asyncio.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0))
        assert last is not None
        return last

    async def _request_once(
        self,
        request_id: int,
        sample: Sample,
        run_origin: float,
        attempt: int,
    ) -> RequestResult:
        if self.http is None:
            raise RuntimeError("StreamingClient must be used as an async context manager")

        started = time.perf_counter()
        input_tokens = self.counter.count_messages(sample.messages)
        result = RequestResult(
            request_id=request_id,
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            task=sample.task,
            status="error",
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            start_offset_s=started - run_origin,
            input_tokens=input_tokens,
            attempts=attempt,
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": sample.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": int(self.bench["max_output_tokens"]),
            "temperature": float(self.bench.get("temperature", 0.0)),
            "top_p": float(self.bench.get("top_p", 1.0)),
        }
        if self.bench.get("ignore_eos", False):
            body["ignore_eos"] = True
        body.update(self.extra_body)

        chunks: list[str] = []
        first_content_at: float | None = None
        try:
            async with self.http.stream("POST", self.url, json=body) as response:
                result.http_status = response.status_code
                if response.is_error:
                    raw = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {raw[:1000]}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage") or {}
                    if usage.get("prompt_tokens") is not None:
                        result.server_prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens") is not None:
                        result.server_completion_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    text = _content_text((choice.get("delta") or {}).get("content"))
                    if text:
                        now = time.perf_counter()
                        if first_content_at is None:
                            first_content_at = now
                        chunks.append(text)
                    if choice.get("finish_reason") is not None:
                        result.finish_reason = str(choice["finish_reason"])

            ended = time.perf_counter()
            result.output_text = "".join(chunks)
            result.output_tokens = self.counter.count_text(result.output_text)
            result.total_tokens = result.input_tokens + result.output_tokens
            result.e2e_s = ended - started
            result.end_offset_s = ended - run_origin
            if first_content_at is None:
                result.error = "Stream completed without nonempty generated content"
                return result
            result.ttft_s = first_content_at - started
            if result.output_tokens >= 2:
                result.tpot_s = (result.e2e_s - result.ttft_s) / (result.output_tokens - 1)
            result.status = "ok"
            return result
        except (httpx.HTTPError, asyncio.TimeoutError, RuntimeError, OSError) as exc:
            ended = time.perf_counter()
            result.e2e_s = ended - started
            result.end_offset_s = ended - run_origin
            result.error = f"{type(exc).__name__}: {exc}"
            result.output_text = "".join(chunks)
            result.output_tokens = self.counter.count_text(result.output_text)
            result.total_tokens = result.input_tokens + result.output_tokens
            if first_content_at is not None:
                result.ttft_s = first_content_at - started
            return result

