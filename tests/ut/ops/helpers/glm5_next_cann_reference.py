# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone PyTorch references for the GLM-5 Next CANN operators.

These implement the operator contracts (plan §9.1) independently of the CANN
ops and of the framework wrappers they replace. They are CPU-run UTs here and
serve as the oracle for the NPU accuracy tests in
``tests/e2e/models/test_glm5_next_key_pool.py``.

Contract notes (from ``aclnnKeyPool`` / ``aclnnKpoolIndexer`` docs and the
op_host tiling):
- ``key_pool`` writes each incoming chunk's ``[K, gate]`` rows into an FP32
  state cache addressed through the state block table (``0`` = no update),
  then compresses every pool that becomes complete inside the chunk: a
  per-row softmax over the pool axis of ``gate + ape`` followed by a weighted
  sum of K.
- ``pool_key_indexer`` scores each visible pool as the head-weighted sum of
  per-head ReLU dot products scaled by ``1 / sqrt(head_dim)``, selects the
  top ``topk // pool_size`` pools causally per query token, expands pool ids
  into token ids, and appends the request-level tail ``[L_orig - pool_tail_k,
  L_orig)`` (kernel ``ExpandAndAppendIndices``; causally capped to rows whose
  position has progressed past the expanded region).
"""

from __future__ import annotations

import math

import torch


def _layer_norm(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    if weight is None or bias is None:
        return x
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight, bias, eps).to(x.dtype)


def key_pool_reference(
    x: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    cmp_ratio: int = 4,
) -> torch.Tensor:
    """Reference of the CANN ``key_pool`` contract (TND layout, coff=1).

    Args:
        x: ``[T, H]`` chunk of hidden states (BF16/FP16).
        wk: ``[D, H]`` K projection weight (rows = head dim).
        gate_weight: ``[D, H]`` gate projection weight.
        ape: ``[cmp_ratio, D]`` positional bias (FP32).
        state_cache: ``[num_blocks, block_size, 2 * D]`` FP32 ``[K, gate]``
            tail state; block id 0 is reserved as the invalid sentinel and
            never written.
        state_block_table: ``[B, pages]`` INT32; values are the framework
            converted key_pool block ids (vLLM b -> b+1, -1 -> 0).
        start_pos: ``[B]`` INT32 absolute position the chunk starts at.
        cu_seqlens: ``[B + 1]`` INT32 ``[0, cumsum(query_lens)]``; required
            for the TND layout.
        norm_weight / norm_bias / norm_eps: optional LayerNorm for K.
        cmp_ratio: pool size (``index_kpool``).

    Returns:
        ``[B, ceil(pages * block_size / cmp_ratio), D]`` pooled keys; only
        pools that complete in this call are filled.
    """
    input_dtype = x.dtype  # captured before the computation cast below
    x = x.float()
    wk = wk.float()
    gate_weight = gate_weight.float()
    ape = ape.float()
    if cu_seqlens is None:
        raise ValueError("key_pool_reference requires cu_seqlens for the TND layout")
    chunk_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch = state_block_table.shape[0]
    block_size = state_cache.shape[1]
    head_dim = wk.shape[0]
    pool_capacity = (
        state_block_table.shape[1] * block_size + cmp_ratio - 1
    ) // cmp_ratio
    pooled = torch.zeros(batch, pool_capacity, head_dim, dtype=torch.float32)

    k = _layer_norm(x @ wk.t(), norm_weight, norm_bias, norm_eps)
    gate = x @ gate_weight.t()

    # Split the chunk per request.
    chunk_rows: list[torch.Tensor] = []
    offset = 0
    for length in chunk_lens.tolist():
        chunk_rows.append((offset, offset + int(length)))
        offset += int(length)
    assert offset == x.shape[0], "cu_seqlens[-1] must equal the chunk length"

    # 1) Persist the incomplete-pool tail of this chunk into the FP32 state
    #    cache (cross-chunk state, plan §9.1): only the last
    #    ``len % cmp_ratio`` tokens need to survive for the next call.
    #    Matches the op golden (key_pool_golden._write_cache).
    for b in range(batch):
        length = int(chunk_lens[b])
        end_pos = int(start_pos[b]) + length
        tail_len = end_pos % cmp_ratio
        if tail_len == 0:
            continue
        tail_start = max(int(start_pos[b]), end_pos - tail_len)
        for local in range(length):
            abs_pos = int(start_pos[b]) + local
            if abs_pos < tail_start:
                continue
            page = abs_pos // block_size
            block_id = int(state_block_table[b, page])
            if block_id == 0:
                continue
            row_start, row_end = chunk_rows[b]
            state_cache[block_id, abs_pos % block_size, :head_dim] = k[row_start + local]
            state_cache[block_id, abs_pos % block_size, head_dim:] = gate[row_start + local]

    # 2) Compress every pool that completes inside this chunk. Tokens before
    #    start_pos are read back from the state cache (cross-chunk tail).
    for b in range(batch):
        length = int(chunk_lens[b])
        chunk_start = int(start_pos[b])
        chunk_end = chunk_start + length
        first_pool = chunk_start // cmp_ratio
        last_pool = (chunk_end - 1) // cmp_ratio
        for pool in range(first_pool, last_pool + 1):
            pool_end = (pool + 1) * cmp_ratio
            if pool_end > chunk_end:
                continue  # pool is still incomplete
            pool_k = torch.zeros(cmp_ratio, head_dim, dtype=torch.float32)
            pool_gate = torch.zeros(cmp_ratio, head_dim, dtype=torch.float32)
            fill = 0
            for token_pos in range(pool * cmp_ratio, pool_end):
                if token_pos < chunk_start:
                    page = token_pos // block_size
                    block_id = int(state_block_table[b, page])
                    if block_id == 0:
                        raise ValueError(
                            f"request {b}: history token {token_pos} has an invalid state block"
                        )
                    pool_k[fill] = state_cache[block_id, token_pos % block_size, :head_dim]
                    pool_gate[fill] = state_cache[block_id, token_pos % block_size, head_dim:]
                else:
                    row_start, _ = chunk_rows[b]
                    pool_k[fill] = k[row_start + token_pos - chunk_start]
                    pool_gate[fill] = gate[row_start + token_pos - chunk_start]
                fill += 1
            scores = (pool_gate + ape).softmax(dim=0)
            pooled[b, pool - first_pool] = (scores * pool_k).sum(dim=0)

    return pooled.to(input_dtype)


def _visible_pool_count(pos: int, pool_size: int, total_pools: int) -> int:
    return min((pos + 1) // pool_size, total_pools)


def pool_key_indexer_reference(
    query: torch.Tensor,
    pool_key: torch.Tensor,
    weights: torch.Tensor,
    pool_tail_k: torch.Tensor,
    *,
    actual_seq_q: torch.Tensor,
    actual_seq_k: torch.Tensor,
    block_table: torch.Tensor | None,
    topk: int,
    pool_size: int,
    mask_mode: int = 3,
) -> torch.Tensor:
    """Reference of the CANN ``pool_key_indexer`` contract (TND query,
    PA_BBND paged key, causal mask; plan §9.1).

    Args:
        query: ``[T, H, D]`` BF16/FP16 query.
        pool_key: ``[blocks, block_size, 1, D]`` BF16 compressed K cache.
        weights: ``[T, H]`` head weights; the framework-side factor
            ``num_heads**-0.5`` is expected to be applied by the caller (the
            op itself applies ``head_dim**-0.5`` and per-head ReLU).
        pool_tail_k: ``[B]`` INT64 ``seq_lens % pool_size``.
        actual_seq_q: ``[B]`` INT64 cumulative query ends (TND prefix sums).
        actual_seq_k: ``[B]`` INT64 per-request pool counts
            ``floor(seq_lens / pool_size)`` (PA, not prefix sums).
        block_table: ``[B, pages]`` INT32 paged block table (``0``-based
            physical ids, padded entries must be masked by ``actual_seq_k``).
        topk: token budget; must be divisible by ``pool_size``.
        pool_size: ``index_kpool``.
        mask_mode: 0 (no mask) or 3 (causal).

    Returns:
        ``[T, topk + pool_size - 1]`` INT32 sparse token indices, ``-1``
        padded: selected pools expanded to token ids followed by the
        request-level tail ``[L_orig - pool_tail_k, L_orig)`` (plan §7,
        kernel ``ExpandAndAppendIndices``).
    """
    query = query.float()
    pool_key = pool_key.float()
    weights = weights.float()
    head_dim = query.shape[-1]
    batch = actual_seq_q.numel()
    sparse_count = topk // pool_size
    out_width = topk + pool_size - 1
    output = torch.full((query.shape[0], out_width), -1, dtype=torch.int32)
    query_ends = [0] + [int(v) for v in actual_seq_q.tolist()]
    scale = 1.0 / math.sqrt(head_dim)

    for b in range(batch):
        q_start, q_end = query_ends[b], query_ends[b + 1]
        tail = int(pool_tail_k[b])
        total_pools = int(actual_seq_k[b])
        seq_len = total_pools * pool_size + tail
        qlen = q_end - q_start
        for j in range(q_start, q_end):
            # Reconstruct the token's absolute position inside the request.
            pos = seq_len - qlen + (j - q_start)
            visible = _visible_pool_count(pos, pool_size, total_pools) if mask_mode == 3 else total_pools
            scores = torch.full((visible,), float("-inf"), dtype=torch.float32)
            for p in range(visible):
                page = p // pool_key.shape[1]
                if block_table is not None and page < block_table.shape[1]:
                    block_id = int(block_table[b, page])
                else:
                    block_id = p  # fall back to identity addressing for tests
                if block_id < 0:
                    continue
                k_vec = pool_key[block_id, p % pool_key.shape[1], 0]
                per_head = torch.nn.functional.relu(
                    (query[j] * k_vec).sum(dim=-1) * scale
                )
                scores[p] = (per_head * weights[j]).sum()
            # Top-k with first-wins tie breaking in ascending pool order.
            k = min(sparse_count, visible)
            selected = torch.topk(scores, k, largest=True).indices if k > 0 else torch.empty(0, dtype=torch.long)
            selected = torch.sort(selected).values  # deterministic ascending order
            col = 0
            for pool in selected.tolist():
                for token in range(pool * pool_size, (pool + 1) * pool_size):
                    output[j, col] = token
                    col += 1
            # Tail append (kernel ExpandAndAppendIndices): the request-level
            # tail [L_orig - pool_tail_k, L_orig) is written at columns
            # [topk, topk + pool_size). Under causal masking the count is
            # capped by (pos - topk + 1) so early prefill rows do not see
            # tail columns before the expanded region is complete.
            if tail > 0:
                if mask_mode == 0:
                    visible_tail_k = tail
                else:
                    visible_tail_k = max(0, min(tail, pos - topk + 1))
                for t in range(visible_tail_k):
                    output[j, topk + t] = seq_len - tail + t
    return output


def sparse_to_dense_token_ids(
    sparse_indices: torch.Tensor,
    seq_len: int,
) -> list[list[int]]:
    """Expand sparse indices to unique valid token ids (test helper)."""
    valid = (sparse_indices >= 0) & (sparse_indices < seq_len)
    return [sparse_indices[i][valid[i]].tolist() for i in range(sparse_indices.shape[0])]
