# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-unbounded tiled Triton scorer for the GLM5 Next indexer.

The launch grid scans 64 compressed pools per program.  Length and query-row
changes only resize the grid/workspace and never create a new specialization.
The weighted query is formed by PyTorch in BF16 to preserve model precision;
the expensive paged-cache dot products are performed here in Triton.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

TRITON_POOL_TILE_SIZE = 64


@triton.jit
def _glm5_next_lightning_indexer_score_kernel(
    weighted_query_ptr,
    indexer_cache_ptr,
    request_ids_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    score_out_ptr,
    query_stride_t,
    query_stride_d,
    cache_stride_block,
    cache_stride_offset,
    cache_stride_d,
    block_table_stride_req,
    block_table_stride_page,
    score_stride_t,
    score_stride_tile,
    pool_block_size,
    index_kpool,
    max_pool_seq_len,
    num_blocks,
    HEAD_DIM: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    token_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    tile_start = tile_idx * BLOCK_POOL

    req_id = tl.load(request_ids_ptr + token_idx).to(tl.int32)
    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    request_pool_len = tl.load(indexer_seq_lens_ptr + req_id).to(tl.int32)
    causal_pool_len = (pos + 1) // index_kpool
    visible_pool_len = tl.minimum(causal_pool_len, request_pool_len)

    dim_offsets = tl.arange(0, HEAD_DIM)
    weighted_query = tl.load(
        weighted_query_ptr
        + token_idx * query_stride_t
        + dim_offsets * query_stride_d
    ).to(tl.float32)

    pool_offsets = tl.arange(0, BLOCK_POOL)
    global_pool_ids = tile_start + pool_offsets
    valid_pool = (
        (global_pool_ids < max_pool_seq_len)
        & (global_pool_ids < visible_pool_len)
    )
    page_offsets = global_pool_ids % pool_block_size
    logical_pages = global_pool_ids // pool_block_size
    physical_blocks = tl.load(
        indexer_block_table_ptr
        + req_id * block_table_stride_req
        + logical_pages * block_table_stride_page,
        mask=valid_pool,
        other=0,
    ).to(tl.int32)
    physical_blocks = tl.maximum(physical_blocks, 0)
    physical_blocks = tl.minimum(physical_blocks, num_blocks - 1)

    keys = tl.load(
        indexer_cache_ptr
        + physical_blocks[None, :] * cache_stride_block
        + page_offsets[None, :] * cache_stride_offset
        + dim_offsets[:, None] * cache_stride_d,
        mask=valid_pool[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(weighted_query[:, None] * keys, axis=0)
    # NPU BF16 matmul returns BF16; the fallback then casts it to FP32.
    scores = scores.to(tl.bfloat16).to(tl.float32)
    scores = tl.where(valid_pool, scores, float("-inf"))
    tl.store(
        score_out_ptr
        + token_idx * score_stride_t
        + tile_idx * score_stride_tile
        + pool_offsets,
        scores,
    )


def glm5_next_lightning_indexer_triton_scores(
    weighted_query: torch.Tensor,
    indexer_cache: torch.Tensor,
    request_ids: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    pool_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    """Return padded global pool scores for one bounded query-row chunk."""
    score_pool_len = max(max_pool_seq_len, pool_topk)
    num_tiles = (
        score_pool_len + TRITON_POOL_TILE_SIZE - 1
    ) // TRITON_POOL_TILE_SIZE
    score_out = torch.empty(
        (weighted_query.shape[0], num_tiles, TRITON_POOL_TILE_SIZE),
        dtype=torch.float32,
        device=weighted_query.device,
    )
    if weighted_query.shape[0] == 0:
        return score_out.view(weighted_query.shape[0], -1)

    _glm5_next_lightning_indexer_score_kernel[
        (weighted_query.shape[0], num_tiles)
    ](
        weighted_query,
        indexer_cache,
        request_ids,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        score_out,
        weighted_query.stride(0),
        weighted_query.stride(1),
        indexer_cache.stride(0),
        indexer_cache.stride(1),
        indexer_cache.stride(3),
        indexer_block_table.stride(0),
        indexer_block_table.stride(1),
        score_out.stride(0),
        score_out.stride(1),
        indexer_cache.shape[1],
        index_kpool,
        max_pool_seq_len,
        indexer_cache.shape[0],
        HEAD_DIM=weighted_query.shape[1],
        BLOCK_POOL=TRITON_POOL_TILE_SIZE,
    )
    return score_out.view(weighted_query.shape[0], -1)
