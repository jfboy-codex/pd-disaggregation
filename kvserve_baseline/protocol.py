"""Build the two HTTP requests for one KVServe transfer."""

from __future__ import annotations


def phase_payloads(body: dict, transfer_id: str) -> tuple[dict, dict]:
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    if not isinstance(transfer_id, str) or not transfer_id:
        raise ValueError("transfer_id is required")

    prefill = body.copy()
    prefill["stream"] = False
    prefill.pop("stream_options", None)
    prefill["max_tokens"] = 1
    if "max_completion_tokens" in prefill:
        prefill["max_completion_tokens"] = 1
    prefill["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": transfer_id,
    }

    decode = body.copy()
    decode["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "transfer_id": transfer_id,
    }
    return prefill, decode
