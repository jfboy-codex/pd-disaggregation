from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Sample:
    sample_id: str
    dataset: str
    task: str
    messages: list[dict[str, str]]
    source_index: int


@dataclass(slots=True)
class RequestResult:
    request_id: int
    sample_id: str
    dataset: str
    task: str
    status: str
    error: str = ""
    http_status: int | None = None
    started_at_utc: str = ""
    start_offset_s: float = 0.0
    end_offset_s: float = 0.0
    ttft_s: float | None = None
    e2e_s: float = 0.0
    tpot_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    server_prompt_tokens: int | None = None
    server_completion_tokens: int | None = None
    finish_reason: str = ""
    output_text: str = ""
    attempts: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

