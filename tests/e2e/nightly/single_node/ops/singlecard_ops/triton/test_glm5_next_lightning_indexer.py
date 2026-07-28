# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.glm5_next_lightning_indexer import (
    _append_tail_to_topk,
    _expand_pools_to_tokens,
    _pool_topk,
)

if HAS_TRITON:
    from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (
        glm5_next_lightning_indexer_triton,
    )
else:
    glm5_next_lightning_indexer_triton = None


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is required")
@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU is required")
def test_glm5_next_lightning_indexer_triton_matches_reference_chain():
    assert glm5_next_lightning_indexer_triton is not None
    device = "npu"
    num_tokens = 2
    num_heads = 2
    head_dim = 128
    index_topk = 4
    index_kpool = 2
    max_pool_seq_len = 5

    query = torch.zeros(
        num_tokens,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    query[:, 0, 0] = 1
    weights = torch.zeros(
        num_tokens,
        num_heads,
        dtype=torch.bfloat16,
        device=device,
    )
    weights[:, 0] = 1
    logical_keys = torch.zeros(
        max_pool_seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    logical_keys[:, 0] = torch.tensor([1, 4, 3, 2, 5], dtype=torch.bfloat16, device=device)
    indexer_cache = torch.zeros(
        (3, 2, 1, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    block_mapping = [2, 0, 1]
    indexer_block_table = torch.tensor([block_mapping], dtype=torch.int32, device=device)
    for pool_id in range(max_pool_seq_len):
        logical_page, offset = divmod(pool_id, 2)
        physical_page = block_mapping[logical_page]
        indexer_cache[physical_page, offset, 0] = logical_keys[pool_id]

    cum_query_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    indexer_seq_lens = torch.tensor([max_pool_seq_len], dtype=torch.int32, device=device)
    positions = torch.tensor([4, 8], dtype=torch.int64, device=device)

    pool_ids = _pool_topk(
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        index_topk // index_kpool,
        index_kpool,
        max_pool_seq_len,
    )
    expected = _expand_pools_to_tokens(
        pool_ids,
        index_topk,
        index_kpool,
    )
    expected = _append_tail_to_topk(
        expected,
        positions,
        index_kpool,
    ).unsqueeze(1)

    result = glm5_next_lightning_indexer_triton(
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        index_topk=index_topk,
        index_kpool=index_kpool,
        max_pool_seq_len=max_pool_seq_len,
    )
    torch.npu.synchronize()

    assert result.dtype == torch.int32
    assert result.shape == expected.shape
    torch.testing.assert_close(
        torch.sort(result[:, :, :index_topk], dim=-1).values,
        torch.sort(expected[:, :, :index_topk], dim=-1).values,
    )
    torch.testing.assert_close(result[:, :, index_topk:], expected[:, :, index_topk:])
