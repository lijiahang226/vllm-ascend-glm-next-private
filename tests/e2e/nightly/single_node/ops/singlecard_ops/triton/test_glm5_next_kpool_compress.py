# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.glm5_next_kpool_compress import glm5_next_kpool_compress_and_write_cache


def _reference_compress(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    scores = slot_score.float() + ape.float().unsqueeze(0)
    return (torch.softmax(scores, dim=1) * slot_k.float()).sum(dim=1).to(torch.bfloat16)


def _reference_scatter(
    kv_cache: torch.Tensor,
    compressed: torch.Tensor,
    loc: torch.Tensor,
) -> torch.Tensor:
    expected = kv_cache.clone()
    block_ids = torch.div(loc, kv_cache.shape[1], rounding_mode="floor")
    block_offsets = torch.remainder(loc, kv_cache.shape[1])
    expected[block_ids, block_offsets, 0, :] = compressed
    return expected


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is required")
@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU is required")
def test_glm5_next_kpool_compress_triton_writes_cache_like_reference():
    device = "npu"
    num_pools = 5
    pool_size = 16
    head_dim = 128
    block_size = 4
    num_blocks = 6

    slot_k_storage = torch.empty(
        (num_pools, pool_size, head_dim * 2),
        dtype=torch.bfloat16,
        device=device,
    )
    slot_score_storage = torch.empty_like(slot_k_storage)
    pool_offsets = torch.arange(pool_size, dtype=torch.float32, device=device)[None, :, None]
    dim_offsets = torch.arange(head_dim, dtype=torch.float32, device=device)[None, None, :]
    row_offsets = torch.arange(num_pools, dtype=torch.float32, device=device)[:, None, None]
    slot_k = slot_k_storage[:, :, ::2]
    slot_score = slot_score_storage[:, :, ::2]
    slot_k.copy_((row_offsets + 1) * 0.25 + pool_offsets * 0.5 + (dim_offsets % 7) * 0.125)
    slot_score.copy_((row_offsets % 3) * 0.2 + pool_offsets * 0.35 - (dim_offsets % 5) * 0.1)

    ape_storage = torch.empty((pool_size, head_dim * 2), dtype=torch.float32, device=device)
    ape = ape_storage[:, ::2]
    ape_pool = torch.arange(pool_size, dtype=torch.float32, device=device)[:, None]
    ape_dim = torch.arange(head_dim, dtype=torch.float32, device=device)[None, :]
    ape.copy_((ape_pool - 1.5) * 0.17 + (ape_dim % 11) * 0.03)

    kv_cache = torch.full(
        (num_blocks, block_size, 1, head_dim),
        -7.0,
        dtype=torch.bfloat16,
        device=device,
    )
    loc = torch.tensor([1, 4, 7, 13, 20], dtype=torch.int64, device=device)
    expected_compressed = _reference_compress(slot_k, slot_score, ape)
    expected_cache = _reference_scatter(kv_cache, expected_compressed, loc)

    result = glm5_next_kpool_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
    )
    torch.npu.synchronize()

    assert result is None
    torch.testing.assert_close(kv_cache, expected_cache, rtol=2e-2, atol=2e-2)
