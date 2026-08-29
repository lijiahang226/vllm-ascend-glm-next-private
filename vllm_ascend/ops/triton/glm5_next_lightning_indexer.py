# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for the narrow GLM5 Next KPool lightning indexer.

The kernel covers one pool sub-range (``chunk_start`` .. ``chunk_start +
``chunk_len``) of the compressed indexer cache and returns the chunk-local
top-k pool ids with their scores.  The Python wrapper merges the chunk-local
results so arbitrarily long compressed histories can be scanned without ever
materializing more than ``TRITON_MAX_POOL_SEQ_LEN`` pool slots per kernel
launch (the NPU register/UB limit that previously forced a PyTorch fallback).
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element

TRITON_MAX_POOL_SEQ_LEN = 2048


@triton.jit
def _glm5_next_lightning_indexer_kernel(
    query_ptr,
    indexer_cache_ptr,
    weights_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    pool_id_out_ptr,
    score_out_ptr,
    query_stride_t: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    weights_stride_t: tl.constexpr,
    weights_stride_h: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    NUM_REQS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    POOL_TOPK: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    chunk_start,
    chunk_len,
):
    token_idx = tl.program_id(0)
    chunk_start = chunk_start.to(tl.int32)
    chunk_len = chunk_len.to(tl.int32)

    req_id = 0
    for req in tl.range(NUM_REQS):
        query_end = tl.load(cum_query_lens_ptr + req).to(tl.int32)
        req_id += tl.where(token_idx >= query_end, 1, 0)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    request_pool_len = tl.load(indexer_seq_lens_ptr + req_id).to(tl.int32)
    causal_pool_len = (pos + 1) // INDEX_KPOOL
    visible_pool_len = tl.minimum(causal_pool_len, request_pool_len)

    dim_offsets = tl.arange(0, HEAD_DIM)
    qbar = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for head_idx in tl.range(NUM_HEADS):
        weight = tl.load(weights_ptr + token_idx * weights_stride_t + head_idx * weights_stride_h).to(tl.float32)
        q = tl.load(
            query_ptr + token_idx * query_stride_t + head_idx * query_stride_h + dim_offsets * query_stride_d
        ).to(tl.float32)
        qbar += q * weight

    pool_offsets = tl.arange(0, BLOCK_POOL)
    chunk_global = chunk_start + pool_offsets
    valid_pool = (pool_offsets < chunk_len) & (chunk_global < visible_pool_len)
    page_offsets = chunk_global % pool_block_size
    logical_pages = chunk_global // pool_block_size
    physical_blocks = tl.load(
        indexer_block_table_ptr + req_id * block_table_stride_req + logical_pages * block_table_stride_page,
        mask=pool_offsets < chunk_len,
        other=0,
    ).to(tl.int64)
    physical_blocks = tl.maximum(physical_blocks, 0)

    scores = tl.zeros((BLOCK_POOL,), dtype=tl.float32)
    for dim_idx in tl.range(HEAD_DIM):
        q_value = get_element(qbar, (dim_idx,))
        k = tl.load(
            indexer_cache_ptr
            + physical_blocks * cache_stride_block
            + page_offsets * cache_stride_offset
            + dim_idx * cache_stride_d,
            mask=valid_pool,
            other=0.0,
        ).to(tl.float32)
        scores += q_value * k
    scores = tl.where(valid_pool, scores, float("-inf"))

    for topk_idx in tl.range(POOL_TOPK):
        best_value = tl.max(scores, axis=0)
        best_idx = tl.argmax(scores, axis=0).to(tl.int32)
        chunk_pool_id = tl.where(best_value == float("-inf"), -1, best_idx)
        tl.store(pool_id_out_ptr + token_idx * POOL_TOPK + topk_idx, chunk_pool_id)
        tl.store(score_out_ptr + token_idx * POOL_TOPK + topk_idx, best_value)
        scores = tl.where(pool_offsets == best_idx, float("-inf"), scores)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def glm5_next_lightning_indexer_triton_chunk(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    chunk_start: int,
    chunk_len: int,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the chunk-local pool top-k and return ``(pool_ids, scores)``.

    ``pool_ids`` contains chunk-relative pool indices with -1 for invalid
    slots and ``scores`` the matching fp32 scores (-inf for invalid slots).
    Both tensors have shape ``[T, POOL_TOPK]``.
    """
    pool_topk = index_topk // index_kpool
    pool_id_out = torch.empty(
        (query.shape[0], pool_topk),
        dtype=torch.int32,
        device=query.device,
    )
    score_out = torch.empty(
        (query.shape[0], pool_topk),
        dtype=torch.float32,
        device=query.device,
    )
    if query.shape[0] == 0:
        return pool_id_out, score_out

    block_pool = _next_power_of_2(
        max(1, min(max_pool_seq_len, TRITON_MAX_POOL_SEQ_LEN))
    )
    _glm5_next_lightning_indexer_kernel[(query.shape[0],)](
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        pool_id_out,
        score_out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        indexer_cache.stride(0),
        indexer_cache.stride(1),
        indexer_cache.stride(3),
        weights.stride(0),
        weights.stride(1),
        indexer_block_table.stride(0),
        indexer_block_table.stride(1),
        indexer_cache.shape[1],
        cum_query_lens.shape[0],
        query.shape[1],
        query.shape[2],
        index_kpool,
        pool_topk,
        block_pool,
        chunk_start,
        chunk_len,
    )
    return pool_id_out, score_out
