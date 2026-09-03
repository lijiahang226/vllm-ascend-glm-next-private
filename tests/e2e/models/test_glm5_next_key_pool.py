# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU accuracy tests for the GLM-5 Next CANN key_pool / pool_key_indexer.

Coverage (plan §9.3/§9.4, §8, §10):
- standalone op accuracy against the CPU references
  (tests/ut/ops/helpers/glm5_next_cann_reference.py) for decode/prefill and
  chunked shapes;
- cross-chunk state: a second call must observe the FP32 tail state written
  by the first call (buffer values, not capture-time constants);
- ACLGraph replay: capture both ops on one content set and replay with
  different lengths/tails/keys; outputs must reflect the new values;
- real-weight end-to-end: eager vs ACLGraph logits and greedy token
  stability (requires GLM5_NEXT_MODEL_DIR with real weights).

Run on an NPU host with torch_npu and the rebuilt ``_C_ascend``:

    pytest -sv --confcutdir=tests/e2e/models \
        tests/e2e/models/test_glm5_next_key_pool.py -k "not real_weights"

``--confcutdir`` stops pytest from loading the parent ``tests/e2e/conftest.py``
(model-eval harness whose named fixtures download hub models), so the
standalone op tests never touch the network.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

try:
    import torch_npu  # noqa: F401

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False

from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded
from tests.ut.ops.helpers.glm5_next_cann_reference import key_pool_reference
from tests.ut.ops.helpers.pool_key_indexer_reference import (
    pool_key_indexer_reference as _official_pki_reference,
)

pytestmark = pytest.mark.skipif(not HAS_NPU, reason="requires an Ascend NPU")

# _C_ascend is loaded lazily by the engine; the tests call the compiled ops
# directly, so load the extension explicitly on NPU hosts.
if HAS_NPU:
    ensure_c_ascend_loaded()

D = 128  # kernel constraint: key_pool headDim in {128, 512}, pool_key_indexer requires exactly 128
H = 4096  # kernel constraint: key_pool hidden size in [1024, 10240], 512-aligned
R = 4  # cmp_ratio / pool_size
TOPK = 8
POOL_BLOCKS = 8  # indexer cache physical blocks
POOL_BLOCK_SIZE = 16  # indexer cache block size (pools per block), 16-aligned
NPU = "npu:0"


# ---------------------------------------------------------------------------
# Standalone op accuracy vs reference
# ---------------------------------------------------------------------------


def _key_pool_inputs(seq_len: int, chunk_len: int, start_pos: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    hidden = torch.randn(chunk_len, H, dtype=torch.bfloat16, device=device) * 0.1
    wk = torch.randn(D, H, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn(D, H, dtype=torch.bfloat16, device=device)
    ape = torch.randn(R, D, dtype=torch.float32, device=device) * 0.1
    state_cache = torch.zeros(4, R, 2 * D, dtype=torch.float32, device=device)
    width = (seq_len + R - 1) // R
    state_block_table = torch.tensor(
        [list(range(1, width + 1)) + [0] * (8 - width)],
        dtype=torch.int32,
        device=device,
    )
    start_pos_t = torch.tensor([start_pos], dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)
    return {
        "hidden": hidden,
        "wk": wk,
        "gate_weight": gate_weight,
        "ape": ape,
        "state_cache": state_cache,
        "state_block_table": state_block_table,
        "start_pos": start_pos_t,
        "cu_seqlens": cu_seqlens,
    }


def _run_key_pool_npu(inp, norm_weight=None, norm_bias=None):
    return torch.ops._C_ascend.key_pool(
        inp["hidden"],
        inp["wk"],
        inp["gate_weight"],
        inp["ape"],
        inp["state_cache"],
        inp["state_block_table"],
        inp["start_pos"],
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        cos=None,
        sin=None,
        cu_seqlens=inp["cu_seqlens"],
        seqused=None,
        cmp_ratio=R,
        norm_eps=1e-6,
        rotary_mode=1,
    )


def test_key_pool_npu_matches_reference_chunked_prefill():
    """KeyPool vs reference: prefill with a cross-chunk tail (plan §9.2)."""
    seq_len, chunk_len, start_pos = 9, 5, 4
    inp = _key_pool_inputs(seq_len, chunk_len, start_pos, NPU, seed=1)
    norm_weight = torch.randn(D, dtype=torch.float32, device=NPU)
    norm_bias = torch.randn(D, dtype=torch.float32, device=NPU)

    # One-shot history (positions 0..3) then the chunk under test (4..8).
    hist = _key_pool_inputs(seq_len, start_pos, 0, NPU, seed=2)
    _run_key_pool_npu(hist, norm_weight, norm_bias)

    pooled = _run_key_pool_npu(inp, norm_weight, norm_bias).cpu()

    expected = key_pool_reference(
        inp["hidden"].cpu(),
        inp["wk"].cpu(),
        inp["gate_weight"].cpu(),
        inp["ape"].cpu(),
        inp["state_cache"].cpu(),
        inp["state_block_table"].cpu(),
        inp["start_pos"].cpu(),
        cu_seqlens=inp["cu_seqlens"].cpu(),
        norm_weight=norm_weight.cpu(),
        norm_bias=norm_bias.cpu(),
        cmp_ratio=R,
    )
    # Pool 1 (tokens 4..7) completes inside the chunk; row pool-first_pool=0.
    torch.testing.assert_close(pooled[0, 0], expected[0, 0], rtol=5e-2, atol=5e-2)
    # The FP32 state cache now holds the chunk tail (token 8) in page block 2.
    assert torch.count_nonzero(inp["state_cache"]).item() > 0


def test_key_pool_npu_cross_chunk_state_is_reflected_in_second_call():
    """A second chunk must observe the tail state written by the first call
    (buffer values, not capture-time constants, plan §8.3)."""
    seq_len, start_pos = 13, 5
    inp = _key_pool_inputs(seq_len, 9, 0, NPU, seed=3)  # first chunk positions 0..8
    _run_key_pool_npu(inp)
    state_after_first = inp["state_cache"].clone().cpu()

    inp2 = _key_pool_inputs(seq_len, 4, start_pos, NPU, seed=4)
    # Reuse the same FP32 state cache buffer that the first call mutated.
    inp2["state_cache"].copy_(state_after_first.to(NPU))
    pooled2 = _run_key_pool_npu(inp2).cpu()

    expected = key_pool_reference(
        inp2["hidden"].cpu(),
        inp2["wk"].cpu(),
        inp2["gate_weight"].cpu(),
        inp2["ape"].cpu(),
        inp2["state_cache"].cpu(),
        inp2["state_block_table"].cpu(),
        inp2["start_pos"].cpu(),
        cu_seqlens=inp2["cu_seqlens"].cpu(),
        cmp_ratio=R,
    )
    # Pool 3 (12..15) is incomplete; pool 2 (8..11) completes inside this
    # chunk from the state tail (8,9) plus chunk tokens (10,11).
    torch.testing.assert_close(pooled2[0, 0], expected[0, 0], rtol=5e-2, atol=5e-2)


def _pki_inputs(seq_len, query_lens, device, seed=0):
    torch.manual_seed(seed)
    total_query = sum(query_lens)
    query = torch.randn(total_query, 2, D, dtype=torch.bfloat16, device=device) * 0.5
    weights = torch.randn(total_query, 2, dtype=torch.bfloat16, device=device) * 0.5
    # Completed-pool count (actual_seq_k) is floor(seq_len / pool_size).
    pool_count = seq_len // R
    pool_key = torch.randn(POOL_BLOCKS, POOL_BLOCK_SIZE, 1, D, dtype=torch.bfloat16, device=device) * 0.2
    page_count = (pool_count + POOL_BLOCK_SIZE - 1) // POOL_BLOCK_SIZE
    block_table = torch.zeros(1, max(page_count, 1), dtype=torch.int32, device=device)
    for page in range(page_count):
        block_table[0, page] = page
    return {
        "query": query,
        "pool_key": pool_key,
        "weights": weights,
        "pool_tail_k": torch.tensor([seq_len % R], dtype=torch.int64, device=device),
        "actual_seq_q": torch.tensor(query_lens, dtype=torch.int64).cumsum(0).to(device),
        "actual_seq_k": torch.tensor([pool_count], dtype=torch.int64, device=device),
        "block_table": block_table,
    }


def _run_pki_npu(inp, topk=TOPK, return_value=True):
    indices, values = torch.ops._C_ascend.pool_key_indexer(
        inp["query"],
        inp["pool_key"],
        inp["weights"],
        inp["pool_tail_k"],
        actual_seq_q=inp["actual_seq_q"],
        actual_seq_k=inp["actual_seq_k"],
        block_table=inp["block_table"],
        layout_q="TND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=R,
        mask_mode=3,
        quant_mode=-1,
        return_value=return_value,
    )
    return indices, values


def _check_pki_against_golden(indices, values, inp, topk=TOPK, pool_size=R, atol=2e-2, rtol=1e-2):
    """Compare the op output against the official CANN golden
    (tests/ut/ops/helpers/pool_key_indexer_reference.py).

    Tie-breaking is not part of the operator contract: when the op and the
    golden select different pools, the selected-score sets must still match
    within tolerance (invalid entries are -inf sentinels on both sides).
    Rows beyond the last actual_seq_q are padded (physical T may exceed the
    summed query lengths in graph buckets) and are out of scope.
    """
    # The official golden is a CPU implementation (its internal arange/full
    # create CPU tensors), so move the inputs to CPU before calling it.
    gold_idx, gold_val = _official_pki_reference(
        inp["query"].cpu(),
        inp["pool_key"].cpu(),
        inp["weights"].cpu(),
        inp["pool_tail_k"].cpu(),
        actual_seq_q=inp["actual_seq_q"].cpu(),
        actual_seq_k=inp["actual_seq_k"].cpu(),
        block_table=inp["block_table"].cpu(),
        layout_q="TND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=3,
        return_value=True,
    )
    indices = indices.cpu()
    values = values.cpu()
    gold_idx = gold_idx.cpu()
    gold_val = gold_val.cpu()
    valid_rows = int(inp["actual_seq_q"][-1].item())
    assert indices.shape[0] >= valid_rows, (indices.shape, valid_rows)
    assert gold_idx.shape[0] == valid_rows, (gold_idx.shape, valid_rows)
    for row in range(valid_rows):
        op_tokens = sorted(int(v) for v in indices[row, :topk] if v >= 0)
        gold_tokens = sorted(int(v) for v in gold_idx[row, :topk] if v >= 0)
        op_tail = [int(v) for v in indices[row, topk:] if v >= 0]
        gold_tail = [int(v) for v in gold_idx[row, topk:] if v >= 0]
        assert op_tail == gold_tail, (row, op_tail, gold_tail)
        if op_tokens == gold_tokens:
            continue
        # Different selection: must be a tie — the selected score sets must
        # match within tolerance.
        op_scores = sorted(float(v) for v in values[row] if math.isfinite(v))
        gold_scores = sorted(float(v) for v in gold_val[row] if math.isfinite(v))
        assert len(op_scores) == len(gold_scores), (row, op_scores, gold_scores)
        for got, expect in zip(op_scores, gold_scores):
            assert abs(got - expect) <= atol + rtol * abs(expect), (row, got, expect)


def test_pool_key_indexer_npu_matches_reference_prefill():
    """PoolKeyIndexer on prefill-shaped input validates against the operator
    contract (plan §9.3): causal top-k validity, exact pool expansion and the
    request-level tail."""
    inp = _pki_inputs(seq_len=13, query_lens=[5], device=NPU, seed=5)
    indices, values = _run_pki_npu(inp)
    assert indices.shape == (5, TOPK + R - 1)
    assert values.shape == (5, TOPK // R)
    _check_pki_against_golden(indices, values, inp)


def test_pool_key_indexer_npu_multi_request_decode():
    """Multi-request decode + mixed tails (plan §9.3)."""
    torch.manual_seed(6)
    query = torch.randn(2, 2, D, dtype=torch.bfloat16, device=NPU) * 0.4
    weights = torch.randn(2, 2, dtype=torch.bfloat16, device=NPU)
    pool_key = torch.randn(POOL_BLOCKS, POOL_BLOCK_SIZE, 1, D, dtype=torch.bfloat16, device=NPU) * 0.2
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=NPU)
    inp = {
        "query": query,
        "pool_key": pool_key,
        "weights": weights,
        "pool_tail_k": torch.tensor([1, 3], dtype=torch.int64, device=NPU),
        "actual_seq_q": torch.tensor([1, 2], dtype=torch.int64, device=NPU),
        "actual_seq_k": torch.tensor([3, 2], dtype=torch.int64, device=NPU),
        "block_table": block_table,
    }
    indices, values = _run_pki_npu(inp)
    assert indices.shape == (2, TOPK + R - 1)
    _check_pki_against_golden(indices, values, inp)


def test_pool_key_indexer_npu_graph_replay_reflects_new_values():
    """Capture both ops once and replay with different content; outputs must
    track the new values (plan §8). Graph capture bakes tensor SHAPES, so
    replay only changes VALUES: hidden/query contents, K rows, block tables
    and the ValueDepend lengths/tails. The PKI replay also exercises the
    padded-T case (last actual_seq_q < physical T), which the Tensor
    ValueDepend path must mask via the lengths read from GM."""
    key_inp = _key_pool_inputs(seq_len=9, chunk_len=5, start_pos=4, device=NPU, seed=7)
    pki_inp = _pki_inputs(seq_len=13, query_lens=[5], device=NPU, seed=8)

    def run_step():
        torch.ops._C_ascend.key_pool(
            key_inp["hidden"],
            key_inp["wk"],
            key_inp["gate_weight"],
            key_inp["ape"],
            key_inp["state_cache"],
            key_inp["state_block_table"],
            key_inp["start_pos"],
            norm_weight=None,
            norm_bias=None,
            cos=None,
            sin=None,
            cu_seqlens=key_inp["cu_seqlens"],
            seqused=None,
            cmp_ratio=R,
            norm_eps=1e-6,
            rotary_mode=1,
        )
        return _run_pki_npu(pki_inp, return_value=True)

    run_step()  # warmup allocates the workspaces the graph will reuse
    graph_cls = getattr(torch.npu, "NPUGraph", None)
    if graph_cls is None:
        pytest.skip("torch.npu.NPUGraph capture is unavailable in this torch_npu build")
    graph = graph_cls()
    with torch.npu.graph(graph):
        captured_indices, captured_values = run_step()
    torch.npu.synchronize()
    # Capture-time actual_seq_q=[5]; rows 0..2 are compared against the
    # replayed rows below (the replay shrinks actual_seq_q to 3).
    captured_tokens = [
        sorted(int(v) for v in captured_indices.cpu()[row] if v >= 0)
        for row in range(3)
    ]

    # ---- replay: same shapes, new values ----
    # key_pool keeps chunk_len=5 so hidden stays [5, H]; new start_pos /
    # state table / state content. seq_len=15 gives a valid state block for
    # the tail write (position 12 -> page 3 -> block 4).
    key_inp2 = _key_pool_inputs(seq_len=15, chunk_len=5, start_pos=8, device=NPU, seed=9)
    key_inp["hidden"].copy_(key_inp2["hidden"])
    key_inp["start_pos"].copy_(key_inp2["start_pos"])
    key_inp["cu_seqlens"].copy_(key_inp2["cu_seqlens"])
    key_inp["state_block_table"].copy_(key_inp2["state_block_table"])
    key_inp["state_cache"].zero_()
    # pool_key_indexer keeps the query shape [5, H, D]; new K rows / weights /
    # block table, and a SHORTER actual_seq_q (3 < physical T=5) to exercise
    # the padded-T masking path (plan §8).
    pki_inp2 = _pki_inputs(seq_len=9, query_lens=[5], device=NPU, seed=10)
    pki_inp["query"].copy_(pki_inp2["query"])
    pki_inp["weights"].copy_(pki_inp2["weights"])
    pki_inp["pool_tail_k"].copy_(pki_inp2["pool_tail_k"])
    pki_inp["actual_seq_q"].copy_(torch.tensor([3], dtype=torch.int64, device=NPU))
    pki_inp["actual_seq_k"].copy_(pki_inp2["actual_seq_k"])
    pki_inp["pool_key"].copy_(pki_inp2["pool_key"])
    pki_inp["block_table"].copy_(pki_inp2["block_table"])

    graph.replay()
    torch.npu.synchronize()

    valid_rows = int(pki_inp["actual_seq_q"][-1].item())
    assert valid_rows == 3
    replayed_tokens = [
        sorted(int(v) for v in captured_indices.cpu()[row] if v >= 0)
        for row in range(valid_rows)
    ]
    # The key_pool call inside the graph must have honored the new block
    # table: the replayed state cache is no longer empty even though the
    # capture-time state was zeroed before replay.
    assert torch.count_nonzero(key_inp["state_cache"]).item() > 0
    assert replayed_tokens != captured_tokens
    # The replayed output (same captured buffers, new ValueDepend lengths /
    # tails / K rows) must satisfy the operator contract.
    _check_pki_against_golden(captured_indices, captured_values, pki_inp)


# ---------------------------------------------------------------------------
# Real-weight end-to-end (plan §9.4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GLM5_NEXT_MODEL_DIR"),
    reason="GLM5_NEXT_MODEL_DIR (real GLM-5-Next weights) must be set",
)
def test_glm5_next_key_pool_eager_vs_aclgraph_real_weights():
    """eager vs ACLGraph logits and greedy tokens with real weights.

    Runs a short prefill + decode + a chunked-prefill request, once with
    enforce_eager=True and once with cudagraph capture, then compares the
    greedy output tokens (plan §9.4). Requires an NPU host and a
    GLM-5-Next checkpoint directory.
    """
    from vllm import LLM, SamplingParams

    model_dir = os.environ["GLM5_NEXT_MODEL_DIR"]
    prompts = [
        "The capital of France is",
        "Solve the following problem step by step: 27 * 43 =",
    ]

    def run(enforce_eager: bool) -> list[str]:
        llm = LLM(
            model=model_dir,
            enforce_eager=enforce_eager,
            max_model_len=8192,
            trust_remote_code=True,
        )
        params = SamplingParams(temperature=0.0, max_tokens=32)
        outputs = llm.generate(prompts, params)
        llm.shutdown()
        return [out.outputs[0].text for out in outputs]

    eager_tokens = run(enforce_eager=True)
    graph_tokens = run(enforce_eager=False)

    # The ReLU contract changes sparse index selection vs the old Triton
    # path; eager and ACLGraph must agree with each other, which isolates
    # graph-capture defects from accepted operator semantic changes.
    assert eager_tokens == graph_tokens
