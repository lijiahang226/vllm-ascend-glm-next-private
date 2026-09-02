# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Accuracy regressions for the GLM5 Next CANN indexer operators."""

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.glm5_next_key_pool import glm5_next_key_pool
from vllm_ascend.ops.glm5_next_kpool_compress import (
    glm5_next_kpool_compress_and_write_cache,
)
from vllm_ascend.ops.glm5_next_lightning_indexer import (
    glm5_next_lightning_indexer,
)
from vllm_ascend.ops.glm5_next_pool_key_indexer import (
    glm5_next_pool_key_indexer,
)


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required",
)


def test_cann_key_pool_matches_previous_compression_op():
    """Fused projection/norm/pooling matches the previous Triton/small op."""
    torch.manual_seed(7)
    device = torch.device("npu")
    pool_size, head_dim, hidden_dim = 4, 128, 256

    hidden_states = torch.randn(
        pool_size, hidden_dim, dtype=torch.bfloat16, device=device
    )
    wk = torch.randn(head_dim, hidden_dim, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn_like(wk)
    ape = torch.randn(pool_size, head_dim, dtype=torch.float32, device=device)
    norm_weight = torch.randn(head_dim, dtype=torch.float32, device=device)
    norm_bias = torch.randn(head_dim, dtype=torch.float32, device=device)

    state_cache = torch.zeros(
        1, pool_size, 2 * head_dim, dtype=torch.float32, device=device
    )
    pooled = glm5_next_key_pool(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        torch.tensor([[0]], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        cu_seqlens=torch.tensor([0, pool_size], dtype=torch.int32, device=device),
        cmp_ratio=pool_size,
    )

    projected_k = F.linear(hidden_states, wk)
    projected_k = F.layer_norm(
        projected_k.float(),
        (head_dim,),
        norm_weight,
        norm_bias,
        1e-6,
    ).to(torch.bfloat16)
    projected_gate = F.linear(hidden_states, gate_weight)
    previous_cache = torch.zeros(
        1, 1, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    glm5_next_kpool_compress_and_write_cache(
        previous_cache,
        projected_k.unsqueeze(0),
        projected_gate.unsqueeze(0),
        ape,
        torch.tensor([0], dtype=torch.int64, device=device),
    )
    torch.npu.synchronize()

    assert pooled.shape == (1, 1, head_dim)
    torch.testing.assert_close(
        pooled[0, 0].float(),
        previous_cache[0, 0, 0].float(),
        rtol=5e-2,
        atol=5e-2,
    )


def test_cann_pool_key_indexer_matches_previous_indexer_chain():
    """CANN pool TopK/expand/tail semantics match the previous indexer op."""
    torch.manual_seed(11)
    device = torch.device("npu")
    pool_size, index_topk = 4, 8
    num_tokens, num_heads, head_dim = 2, 4, 128
    completed_pools = 4
    cache_block_size = 2

    query = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    weights = torch.randn(
        num_tokens, num_heads, dtype=torch.bfloat16, device=device
    )
    cache = torch.randn(
        2,
        cache_block_size,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    positions = torch.tensor([16, 17], dtype=torch.int64, device=device)

    previous = glm5_next_lightning_indexer(
        query,
        cache,
        weights,
        torch.tensor([num_tokens], dtype=torch.int32, device=device),
        torch.tensor([completed_pools], dtype=torch.int32, device=device),
        block_table,
        positions,
        index_topk=index_topk,
        index_kpool=pool_size,
        max_pool_seq_len=completed_pools,
    ).squeeze(1)
    cann = glm5_next_pool_key_indexer(
        query,
        cache,
        weights,
        torch.tensor([2], dtype=torch.int64, device=device),
        actual_seq_q=torch.tensor([num_tokens], dtype=torch.int64, device=device),
        actual_seq_k=torch.tensor([completed_pools], dtype=torch.int64, device=device),
        block_table=block_table,
        layout_q="TND",
        layout_k="PA_BBND",
        topk=index_topk,
        pool_size=pool_size,
        mask_mode=3,
    )
    torch.npu.synchronize()

    assert cann.shape == previous.shape
    # Pool TopK ordering is not part of the contract; compare the selected
    # history as a set and the ordered causal tail exactly.
    torch.testing.assert_close(
        torch.sort(cann[:, :index_topk], dim=-1).values,
        torch.sort(previous[:, :index_topk], dim=-1).values,
    )
    torch.testing.assert_close(cann[:, index_topk:], previous[:, index_topk:])
