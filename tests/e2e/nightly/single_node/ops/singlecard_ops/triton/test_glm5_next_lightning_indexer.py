# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.glm5_next_lightning_indexer import glm5_next_lightning_indexer


def _sorted_history(row: torch.Tensor, history_width: int) -> list[int]:
    return sorted(value for value in row[:history_width].tolist() if value >= 0)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is required")
@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU is required")
@pytest.mark.parametrize(
    ("index_topk", "index_kpool", "num_heads", "max_pool_seq_len"),
    (
        (8, 2, 1, 4096),
        (2048, 4, 16, 4096),
        (30, 3, 7, 2051),
    ),
)
def test_glm5_next_lightning_indexer_triton_chunked_matches_fallback_beyond_2048(
    index_topk: int,
    index_kpool: int,
    num_heads: int,
    max_pool_seq_len: int,
):
    """Triton agrees with fallback across online and non-aligned shapes."""
    head_dim = 128
    cache_block_size = 1120
    num_blocks = (max_pool_seq_len + cache_block_size - 1) // cache_block_size + 1

    rng = torch.Generator().manual_seed(123)
    indexer_cache = torch.randn(
        (num_blocks, cache_block_size, 1, head_dim),
        dtype=torch.bfloat16,
        generator=rng,
    )
    indexer_block_table = (
        torch.arange(num_blocks, dtype=torch.int32)
        .unsqueeze(0)
        .expand(2, -1)
        .contiguous()
    )
    query = torch.randn((2, num_heads, head_dim), dtype=torch.bfloat16, generator=rng)
    weights = torch.randn((2, num_heads), dtype=torch.bfloat16, generator=rng)
    cum_query_lens = torch.tensor([1, 2], dtype=torch.int32)
    indexer_seq_lens = torch.tensor([max_pool_seq_len, 100], dtype=torch.int32)
    positions = torch.tensor([max_pool_seq_len * index_kpool - 3, 99], dtype=torch.int64)

    kwargs = {
        "index_topk": index_topk,
        "index_kpool": index_kpool,
        "max_pool_seq_len": max_pool_seq_len,
    }
    npu_args = (
        query.to("npu"),
        indexer_cache.to("npu"),
        weights.to("npu"),
        cum_query_lens.to("npu"),
        indexer_seq_lens.to("npu"),
        indexer_block_table.to("npu"),
        positions.to("npu"),
    )
    env_name = "VLLM_ASCEND_ENABLE_GLM5_NEXT_TRITON_INDEXER"
    old_env = os.environ.get(env_name)
    try:
        os.environ[env_name] = "0"
        expected = glm5_next_lightning_indexer(*npu_args, **kwargs).cpu()
        os.environ[env_name] = "1"
        result = glm5_next_lightning_indexer(*npu_args, **kwargs).cpu()
    finally:
        if old_env is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_env
    torch.npu.synchronize()

    assert result.dtype == torch.int32
    assert result.shape == expected.shape
    for row_idx in range(2):
        assert _sorted_history(result[row_idx, 0], index_topk) == _sorted_history(
            expected[row_idx, 0],
            index_topk,
        )
        torch.testing.assert_close(
            result[row_idx, 0, index_topk:].to("cpu"),
            expected[row_idx, 0, index_topk:],
        )
