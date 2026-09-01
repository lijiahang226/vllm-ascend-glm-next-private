# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM5 Next KeyPool fused CANN op wrapper.

The CANN ``aclnnKeyPool`` custom op fuses the indexer K/Gate projection,
optional K LayerNorm, paged state-cache update and pool compression into a
single kernel.  This module dispatches to
``torch.ops._C_ascend.npu_key_pool`` (the custom ops are always built
together with vllm-ascend).
"""

from __future__ import annotations

from typing import Optional

import torch

SUPPORTED_RATIOS = (2, 4, 8, 16, 32, 64, 128)


def glm5_next_key_pool(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Compress indexer K pools with the fused CANN KeyPool op.

    ``state_cache`` is updated in place (tail-only semantics): only the
    tokens that do not yet form a complete pool are written back.
    """
    if cmp_ratio not in SUPPORTED_RATIOS:
        raise ValueError(f"unsupported cmp_ratio={cmp_ratio}")
    if (norm_weight is None) != (norm_bias is None):
        raise ValueError("norm_weight and norm_bias must be passed as a pair")
    if wk.dim() != 2 or gate_weight.shape != wk.shape:
        raise ValueError("wk and gate_weight must have the same rank-2 shape")
    if ape.shape != (cmp_ratio, wk.size(0)):
        raise ValueError(f"ape shape must be [{cmp_ratio}, {wk.size(0)}]")
    if state_cache.dim() != 3 or state_cache.size(-1) != 2 * wk.size(0):
        raise ValueError("state_cache shape must be [block_num, block_size, 2*D]")
    if cache_block_table.dim() != 2:
        raise ValueError("cache_block_table shape must be [B, max_block_num_per_batch]")
    if start_pos.dim() != 1 or start_pos.numel() != cache_block_table.size(0):
        raise ValueError("start_pos shape must be [B]")

    return torch.ops._C_ascend.npu_key_pool(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        cu_seqlens=cu_seqlens,
        cmp_ratio=cmp_ratio,
        norm_eps=norm_eps,
    )
