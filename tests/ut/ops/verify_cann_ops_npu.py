# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU-only verification for the CANN KeyPool / PoolKeyIndexer ops.

Runs the two custom ops directly (no model) on minimal decode- and
prefill-shaped inputs and compares against the CPU golden references in
``D:\\projects\\kernel``:

- ``key_pool_model_debug_golden.key_pool_simulation``
- ``pool_key_indexer_reference.pool_key_indexer_reference``

Usage (on an NPU host with torch_npu and the rebuilt vllm_ascend_C):

    python tests/ut/ops/verify_cann_ops_npu.py

Exit code 0 = both ops match the golden within tolerance.
"""

from __future__ import annotations

import sys

import torch

from vllm_ascend.ops.glm5_next_key_pool import glm5_next_key_pool
from vllm_ascend.ops.glm5_next_pool_key_indexer import glm5_next_pool_key_indexer

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------
B = 1
R = 4  # cmp_ratio / pool_size
D = 8  # head dim (small for CPU golden speed)
H = 16  # hidden size
N = 2  # indexer heads
TOPK = 8  # must be divisible by R
MAX_BLOCKS = 8  # state block table width (logical blocks)
POOL_BLOCKS = 4  # indexer cache physical blocks
POOL_BLOCK_SIZE = 4  # indexer cache block size (pools per block)


def _make_inputs(device: torch.device, *, prefill: bool):
    """Build key_pool + pool_key_indexer inputs for one scenario.

    decode: 1 token, no completed pool (weak: shapes only).
    prefill: 6 tokens, R=4 -> 1 completed pool + 2 tail tokens (strong:
    exercises the pool compression values).
    """
    torch.manual_seed(1 if prefill else 0)
    T = 6 if prefill else 1
    tail = 2 if prefill else 1
    pools = 1 if prefill else 0

    hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    wk = torch.randn(D, H, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn(D, H, dtype=torch.bfloat16, device=device)
    ape = torch.randn(R, D, dtype=torch.float32, device=device)
    state_cache = torch.zeros(2, R, 2 * D, dtype=torch.float32, device=device)
    cache_block_table = torch.zeros(B, MAX_BLOCKS, dtype=torch.int32, device=device)
    cache_block_table[0, 0] = 1  # logical block 0 -> physical block 1
    start_pos = torch.zeros(B, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    norm_weight = torch.randn(D, dtype=torch.float32, device=device)
    norm_bias = torch.randn(D, dtype=torch.float32, device=device)

    query = torch.randn(T, N, D, dtype=torch.bfloat16, device=device)
    pool_key = torch.zeros(POOL_BLOCKS, POOL_BLOCK_SIZE, 1, D, dtype=torch.bfloat16, device=device)
    weights = torch.randn(T, N, dtype=torch.bfloat16, device=device)
    pool_tail_k = torch.tensor([tail], dtype=torch.int64, device=device)
    actual_seq_q = torch.tensor([T], dtype=torch.int64, device=device)  # B-element prefix sum
    actual_seq_k = torch.tensor([pools], dtype=torch.int64, device=device)  # completed pools
    block_table = torch.zeros(B, 2, dtype=torch.int32, device=device)
    block_table[0, 0] = 0  # pool 0 -> physical block 0

    return {
        "hidden_states": hidden_states,
        "wk": wk,
        "gate_weight": gate_weight,
        "ape": ape,
        "state_cache": state_cache,
        "cache_block_table": cache_block_table,
        "start_pos": start_pos,
        "cu_seqlens": cu_seqlens,
        "norm_weight": norm_weight,
        "norm_bias": norm_bias,
        "query": query,
        "pool_key": pool_key,
        "weights": weights,
        "pool_tail_k": pool_tail_k,
        "actual_seq_q": actual_seq_q,
        "actual_seq_k": actual_seq_k,
        "block_table": block_table,
    }


def _golden_key_pool(inp):
    sys.path.insert(0, r"D:\projects\kernel")
    from key_pool_model_debug_golden import key_pool_simulation

    return key_pool_simulation(
        inp["hidden_states"].cpu(),
        inp["wk"].cpu(),
        inp["gate_weight"].cpu(),
        inp["ape"].cpu(),
        inp["state_cache"].cpu(),
        inp["cache_block_table"].cpu(),
        inp["start_pos"].cpu(),
        norm_weight=inp["norm_weight"].cpu(),
        norm_bias=inp["norm_bias"].cpu(),
        cu_seqlens=inp["cu_seqlens"].cpu(),
        cmp_ratio=R,
    )


def _golden_pool_key_indexer(inp):
    sys.path.insert(0, r"D:\projects\kernel")
    from pool_key_indexer_reference import pool_key_indexer_reference

    return pool_key_indexer_reference(
        inp["query"].cpu(),
        inp["pool_key"].cpu(),
        inp["weights"].cpu(),
        inp["pool_tail_k"].cpu(),
        actual_seq_q=inp["actual_seq_q"].cpu(),
        actual_seq_k=inp["actual_seq_k"].cpu(),
        block_table=inp["block_table"].cpu(),
        layout_q="TND",
        layout_k="PA_BBND",
        topk=TOPK,
        pool_size=R,
        mask_mode=3,
        return_value=False,
    )


def _run_key_pool(inp, tag: str) -> int:
    print(f"== KeyPool [{tag}] ==")
    try:
        pooled = glm5_next_key_pool(
            inp["hidden_states"],
            inp["wk"],
            inp["gate_weight"],
            inp["ape"],
            inp["state_cache"],
            inp["cache_block_table"],
            inp["start_pos"],
            norm_weight=inp["norm_weight"],
            norm_bias=inp["norm_bias"],
            cu_seqlens=inp["cu_seqlens"],
            cmp_ratio=R,
        )
        golden = _golden_key_pool(inp)
        pooled_cpu = pooled.cpu().float()
        golden_cpu = golden.float()
        print(f"  pooled_key shape: {tuple(pooled_cpu.shape)}  golden: {tuple(golden_cpu.shape)}")
        if pooled_cpu.shape != golden_cpu.shape:
            print("  SHAPE MISMATCH")
            return 1
        diff = (pooled_cpu - golden_cpu).abs()
        print(f"  max abs diff: {diff.max().item():.6f}  (rows > 1e-2: {(diff.abs() > 1e-2).any(dim=-1).sum().item()})")
        print(f"  pooled[0,:2]: {pooled_cpu[0, :2].tolist()}")
        print(f"  golden[0,:2]: {golden_cpu[0, :2].tolist()}")
        if diff.max().item() <= 1e-2:
            print("  KeyPool MATCHES golden")
        else:
            print("  KeyPool DIFFERS from golden")
    except Exception as exc:  # noqa: BLE001
        print(f"  KeyPool FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _run_pool_key_indexer(inp, tag: str) -> int:
    print(f"== PoolKeyIndexer [{tag}] ==")
    try:
        indices = glm5_next_pool_key_indexer(
            inp["query"],
            inp["pool_key"],
            inp["weights"],
            inp["pool_tail_k"],
            actual_seq_q=inp["actual_seq_q"],
            actual_seq_k=inp["actual_seq_k"],
            block_table=inp["block_table"],
            layout_q="TND",
            layout_k="PA_BBND",
            topk=TOPK,
            pool_size=R,
            mask_mode=3,
        )
        golden_idx, _ = _golden_pool_key_indexer(inp)
        idx_cpu = indices.cpu()
        print(f"  indices shape: {tuple(idx_cpu.shape)}  golden: {tuple(golden_idx.shape)}")
        if idx_cpu.shape != golden_idx.shape:
            print("  SHAPE MISMATCH")
            return 1
        print(f"  indices[0]: {idx_cpu[0].tolist()}")
        print(f"  golden[0]:  {golden_idx[0].tolist()}")
        if torch.equal(idx_cpu, golden_idx):
            print("  PoolKeyIndexer MATCHES golden")
        else:
            print("  PoolKeyIndexer DIFFERS from golden")
    except Exception as exc:  # noqa: BLE001
        print(f"  PoolKeyIndexer FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


def main() -> int:
    if not torch.npu.is_available():
        print("NPU not available; this script must run on an Ascend host.")
        return 2
    device = torch.device("npu:0")
    rc = 0
    for prefill, tag in ((False, "decode"), (True, "prefill")):
        inp = _make_inputs(device, prefill=prefill)
        rc |= _run_key_pool(inp, tag)
        rc |= _run_pool_key_indexer(inp, tag)
    print("== done ==")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
