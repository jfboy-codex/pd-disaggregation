# SPDX-License-Identifier: Apache-2.0
"""KVServe KV utilities adapted for vLLM 0.30 FlashAttention layouts."""

import torch


def make_slot_mapping(token_ids: list[int], block_ids: list[int],
                      block_size: int) -> torch.Tensor:
    """Compute absolute slot indices for the given tokens.

    Returns a CPU LongTensor of shape [num_tokens].
    """
    num_tokens = len(token_ids)
    block_ids_t = torch.tensor(block_ids, dtype=torch.long)
    offsets = torch.arange(block_size, dtype=torch.long)
    # slots[b, o] = block_ids[b] * block_size + o
    slots = (block_ids_t.unsqueeze(1) * block_size + offsets).flatten()
    return slots[:num_tokens]


def extract_kv_from_layer(
    kv_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Extract tokens for one request from a paged KV buffer.

    Args:
        kv_layer: [2, num_pages, page_size, ...] on GPU.
        slot_mapping: [num_tokens] absolute slot indices (CPU or GPU).

    Returns:
        [2, num_tokens, kv_dim] on the same device as kv_layer.
    """
    num_pages = kv_layer.shape[1]
    page_size = kv_layer.shape[2]
    flat = kv_layer.reshape(2, num_pages * page_size, -1)
    sm = slot_mapping.to(flat.device)
    return flat[:, sm, :].clone()


def inject_kv_into_layer(
    kv_cache_layer: torch.Tensor,
    kv_data: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Inject KV tokens for one request into a paged KV buffer in-place.

    Args:
        kv_cache_layer: [2, num_pages, page_size, ...] on GPU.
        kv_data: [2, num_tokens, kv_dim] (CPU or GPU).
        slot_mapping: [num_tokens] absolute slot indices (CPU or GPU).
    """
    num_pages = kv_cache_layer.shape[1]
    page_size = kv_cache_layer.shape[2]
    flat = kv_cache_layer.reshape(2, num_pages * page_size, -1)
    sm = slot_mapping.to(flat.device)
    flat[:, sm, :] = kv_data.to(flat.device)


def extract_kv_from_layer_by_blocks(
    kv_layer: torch.Tensor,
    block_ids: list[int],
) -> torch.Tensor:
    """Extract KV blocks for one request using block IDs.

    Mirrors vLLM p2p connector behavior:
    - FlashAttention layout: [2, num_blocks, ...] -> kv[:, block_ids, ...]
    - MLA/FlashInfer layout: [num_blocks, 2, ...] -> kv[block_ids, ...]
    - vLLM 0.30 FlashAttention: [blocks, heads, tokens, 2*head_size]
      -> [2, selected_blocks, tokens, heads, head_size]
    """
    if not block_ids:
        # Keep a valid empty tensor with compatible rank.
        if kv_layer.shape[0] == 2:
            return kv_layer[:, :0, ...].clone()
        if kv_layer.ndim == 4 and kv_layer.shape[1] != 2:
            return kv_layer.new_empty((2, 0, kv_layer.shape[2],
                                       kv_layer.shape[1], kv_layer.shape[3] // 2))
        return kv_layer[:0, ...].clone()

    idx = torch.tensor(block_ids, device=kv_layer.device, dtype=torch.long)
    if kv_layer.shape[0] == 2:  # FlashAttention
        return kv_layer[:, idx, ...].clone()
    if kv_layer.shape[1] == 2:  # MLA/FlashInfer
        return kv_layer[idx, ...].clone()
    if kv_layer.ndim == 4 and kv_layer.shape[-1] % 2 == 0:
        blocks = kv_layer.index_select(0, idx).transpose(1, 2)
        key, value = blocks.chunk(2, dim=-1)
        return torch.stack((key, value), dim=0).clone()
    raise RuntimeError(f"Unsupported KV layout shape: {tuple(kv_layer.shape)}")


def inject_kv_into_layer_by_blocks(
    kv_cache_layer: torch.Tensor,
    kv_data: torch.Tensor,
    block_ids: list[int],
    request_id: str = "",
) -> None:
    """Inject KV blocks for one request using block IDs.

    If producer/consumer block counts differ by tail blocks, inject overlap only
    (same behavior as vLLM p2p connector warnings).
    """
    if not block_ids:
        return

    idx = torch.tensor(block_ids, device=kv_cache_layer.device, dtype=torch.long)
    payload = kv_data.to(kv_cache_layer.device)

    if kv_cache_layer.shape[0] == 2:  # FlashAttention
        num_blocks = payload.shape[1]
        if idx.numel() == num_blocks:
            kv_cache_layer[:, idx, ...] = payload
        else:
            kv_cache_layer[:, idx[:num_blocks], ...] = payload
        return

    if kv_cache_layer.shape[1] == 2:  # MLA/FlashInfer
        num_blocks = payload.shape[0]
        if idx.numel() == num_blocks:
            kv_cache_layer[idx, ...] = payload
        else:
            kv_cache_layer[idx[:num_blocks], ...] = payload
        return

    if kv_cache_layer.ndim == 4 and kv_cache_layer.shape[-1] % 2 == 0:
        if payload.shape[0] != 2 or payload.ndim != 5:
            raise ValueError(f"Invalid vLLM 0.30 KV payload: {tuple(payload.shape)}")
        num_blocks = payload.shape[1]
        if num_blocks > idx.numel():
            raise ValueError("KV payload has more blocks than destination")
        blocks = torch.cat((payload[0], payload[1]), dim=-1).transpose(1, 2)
        if blocks.shape[1:] != kv_cache_layer.shape[1:]:
            raise ValueError("KV payload shape does not match destination")
        kv_cache_layer.index_copy_(0, idx[:num_blocks], blocks)
        return

    raise RuntimeError(
        f"Unsupported KV layout for request {request_id}: "
        f"{tuple(kv_cache_layer.shape)}"
    )

