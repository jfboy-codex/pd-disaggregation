"""CPU checks for the vLLM 0.30 FlashAttention KV layout adapter."""

import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parents[1] / "kv_utils_vllm030.py"
spec = importlib.util.spec_from_file_location("kv_utils_vllm030", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extract_and_inject_new_layout():
    source = torch.arange(7 * 4 * 16 * 256, dtype=torch.float32).reshape(7, 4, 16, 256)
    extracted = module.extract_kv_from_layer_by_blocks(source, [5, 1])
    assert extracted.shape == (2, 2, 16, 4, 128)
    target = torch.zeros_like(source)
    module.inject_kv_into_layer_by_blocks(target, extracted, [2, 6])
    assert torch.equal(target[2], source[5])
    assert torch.equal(target[6], source[1])
    assert torch.count_nonzero(target[[0, 1, 3, 4, 5]]) == 0


def test_empty_blocks_new_layout():
    source = torch.zeros((7, 4, 16, 256))
    assert module.extract_kv_from_layer_by_blocks(source, []).shape == (2, 0, 16, 4, 128)

