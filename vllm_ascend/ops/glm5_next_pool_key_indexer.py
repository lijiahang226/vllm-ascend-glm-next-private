# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM5 Next pool-key indexer CANN op wrapper.

The CANN ``aclnnPoolKeyIndexer`` custom op selects the top-k pools for every
query token from a pool-compressed key cache, expands the selected pools to
token indices and appends the causal tail.  This module dispatches to
``torch.ops._C_ascend.npu_pool_key_indexer`` (the custom ops are always
built together with vllm-ascend).
"""

from __future__ import annotations

import torch


def glm5_next_pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="TND",
    layout_k="PA_BBND",
    topk=128,
    pool_size=16,
    mask_mode=3,
) -> torch.Tensor:
    """Select top-k pool token indices with the CANN PoolKeyIndexer op.

    Returns ``sparse_indices`` with shape ``[T1, topk + pool_size - 1]``
    (TND query layout); invalid entries are filled with ``-1``.
    """
    if layout_q not in ("BSND", "TND"):
        raise ValueError(f"unsupported layout_q: {layout_q}")
    if layout_k not in ("BSND", "TND", "PA_BBND"):
        raise ValueError(f"unsupported layout_k: {layout_k}")
    if mask_mode not in (0, 3):
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if topk % pool_size != 0:
        raise ValueError(f"topk({topk}) must be divisible by pool_size({pool_size})")
    if pool_key.shape[-2] != 1:
        raise ValueError(f"pool_key N2 must be 1, got {pool_key.shape[-2]}")
    if query.shape[-1] != pool_key.shape[-1]:
        raise ValueError(
            f"headDim mismatch: query {query.shape[-1]} vs pool_key {pool_key.shape[-1]}"
        )

    sparse_indices, _ = torch.ops._C_ascend.npu_pool_key_indexer(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=False,
    )
    return sparse_indices
