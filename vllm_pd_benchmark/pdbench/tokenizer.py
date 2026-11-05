from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def resolve_hf_snapshot(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if (candidate / "tokenizer_config.json").exists():
        return candidate.resolve()

    refs_main = candidate / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = candidate / "snapshots" / revision
        if snapshot.exists():
            return snapshot.resolve()

    snapshots = candidate / "snapshots"
    if snapshots.is_dir():
        choices = sorted(
            (item for item in snapshots.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if choices:
            return choices[0].resolve()
    return candidate.resolve()


class TokenCounter:
    def __init__(self, config: dict[str, Any], model: str):
        configured = config.get("path") or model
        local_path = resolve_hf_snapshot(configured)
        source = str(local_path) if local_path.exists() else configured
        self.source = source
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            local_files_only=local_path.exists(),
        )

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        try:
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if getattr(ids, "ndim", None) == 2:
                return int(ids.shape[1])
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                return len(ids[0])
            return len(ids)
        except (ValueError, TypeError, AttributeError, KeyError):
            text = "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            ) + "\nassistant:"
            return len(self.tokenizer.encode(text, add_special_tokens=True))

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))


