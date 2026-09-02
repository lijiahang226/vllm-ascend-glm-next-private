# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the GLM5 Next CANN KeyPool / PoolKeyIndexer wrappers.

The CANN custom ops (``aclnnKeyPool`` / ``aclnnPoolKeyIndexer``) require NPU
hardware.  These tests verify the CANN dispatch and input validation on CPU
with mocks.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.glm5_next_key_pool import glm5_next_key_pool
from vllm_ascend.ops.glm5_next_pool_key_indexer import glm5_next_pool_key_indexer


def _make_bsh_inputs():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    wk = torch.randn(8, 32, dtype=torch.bfloat16)
    gate_weight = torch.randn(8, 32, dtype=torch.bfloat16)
    ape = torch.randn(4, 8, dtype=torch.float32)
    state_cache = torch.zeros(5, 4, 16, dtype=torch.float32)
    block_ids = torch.arange(0, 4, dtype=torch.int32)
    cache_block_table = block_ids.reshape(2, 2)
    start_pos = torch.zeros(2, dtype=torch.int32)
    return (
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
    )


def _make_pki_inputs():
    torch.manual_seed(2)
    query = torch.randn(16, 4, 8, dtype=torch.bfloat16)
    paged = torch.randn(4, 2, 1, 8, dtype=torch.bfloat16)
    weights = torch.randn(16, 4, dtype=torch.bfloat16)
    pool_tail_k = torch.tensor([0, 2], dtype=torch.int64)
    actual_seq_q = torch.tensor([6, 16], dtype=torch.int64)
    actual_seq_k = torch.tensor([3, 5], dtype=torch.int64)
    block_table = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    return (
        query,
        paged,
        weights,
        pool_tail_k,
        actual_seq_q,
        actual_seq_k,
        block_table,
    )


def test_key_pool_dispatches_to_cann_op():
    (
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
    ) = _make_bsh_inputs()
    hidden_states = hidden_states.to("meta")
    wk = wk.to("meta")
    gate_weight = gate_weight.to("meta")
    ape = ape.to("meta")
    state_cache = state_cache.to("meta")
    cache_block_table = cache_block_table.to("meta")
    start_pos = start_pos.to("meta")
    expected = torch.empty(2, 2, 8, dtype=torch.bfloat16, device="meta")
    with patch.object(torch.ops, "_C_ascend", create=True) as mock_ops:
        mock_ops.npu_key_pool = MagicMock(return_value=expected)
        result = glm5_next_key_pool(
            hidden_states,
            wk,
            gate_weight,
            ape,
            state_cache,
            cache_block_table,
            start_pos,
            cmp_ratio=4,
        )
        assert result is expected
        mock_ops.npu_key_pool.assert_called_once()
        call_kwargs = mock_ops.npu_key_pool.call_args.kwargs
        assert call_kwargs["cmp_ratio"] == 4
        assert call_kwargs["norm_weight"] is None
        assert call_kwargs["cu_seqlens"] is None


def test_key_pool_validates_inputs():
    (
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
    ) = _make_bsh_inputs()
    with patch.object(torch.ops, "_C_ascend", create=True) as mock_ops:
        mock_ops.npu_key_pool = MagicMock()
        with pytest.raises(ValueError, match="unsupported cmp_ratio"):
            glm5_next_key_pool(
                hidden_states,
                wk,
                gate_weight,
                ape,
                state_cache,
                cache_block_table,
                start_pos,
                cmp_ratio=3,
            )
        with pytest.raises(ValueError, match="must be passed as a pair"):
            glm5_next_key_pool(
                hidden_states,
                wk,
                gate_weight,
                ape,
                state_cache,
                cache_block_table,
                start_pos,
                norm_weight=torch.ones(8),
            )
        with pytest.raises(ValueError, match="ape shape"):
            glm5_next_key_pool(
                hidden_states,
                wk,
                gate_weight,
                torch.randn(2, 8),
                state_cache,
                cache_block_table,
                start_pos,
            )


def test_pool_key_indexer_dispatches_to_cann_op():
    (
        query,
        paged,
        weights,
        pool_tail_k,
        actual_seq_q,
        actual_seq_k,
        block_table,
    ) = _make_pki_inputs()
    query = query.to("meta")
    paged = paged.to("meta")
    weights = weights.to("meta")
    pool_tail_k = pool_tail_k.to("meta")
    block_table = block_table.to("meta")
    expected = torch.empty(16, 11, dtype=torch.int32, device="meta")
    with patch.object(torch.ops, "_C_ascend", create=True) as mock_ops:
        mock_ops.npu_pool_key_indexer = MagicMock(return_value=(expected, torch.empty(0)))
        result = glm5_next_pool_key_indexer(
            query,
            paged,
            weights,
            pool_tail_k,
            actual_seq_q=actual_seq_q,
            actual_seq_k=actual_seq_k,
            block_table=block_table,
            layout_q="TND",
            layout_k="PA_BBND",
            topk=8,
            pool_size=4,
            mask_mode=3,
        )
        assert result is expected
        mock_ops.npu_pool_key_indexer.assert_called_once()
        call_kwargs = mock_ops.npu_pool_key_indexer.call_args.kwargs
        assert call_kwargs["layout_q"] == "TND"
        assert call_kwargs["layout_k"] == "PA_BBND"
        assert call_kwargs["topk"] == 8
        assert call_kwargs["pool_size"] == 4
        assert call_kwargs["mask_mode"] == 3
        assert call_kwargs["quant_mode"] == -1
        assert call_kwargs["return_value"] is False


def test_pool_key_indexer_validates_inputs():
    (
        query,
        paged,
        weights,
        pool_tail_k,
        actual_seq_q,
        actual_seq_k,
        block_table,
    ) = _make_pki_inputs()
    with patch.object(torch.ops, "_C_ascend", create=True) as mock_ops:
        mock_ops.npu_pool_key_indexer = MagicMock()
        with pytest.raises(ValueError, match="must be divisible by pool_size"):
            glm5_next_pool_key_indexer(
                query,
                paged,
                weights,
                pool_tail_k,
                layout_q="TND",
                layout_k="TND",
                topk=5,
                pool_size=2,
            )
        with pytest.raises(ValueError, match="N2 must be 1"):
            glm5_next_pool_key_indexer(
                query,
                paged.repeat(1, 1, 2, 1),
                weights,
                pool_tail_k,
                layout_q="TND",
                layout_k="PA_BBND",
                block_table=block_table,
                topk=8,
                pool_size=4,
            )
