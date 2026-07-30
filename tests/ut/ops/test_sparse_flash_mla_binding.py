# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

pytest.importorskip("vllm_ascend.vllm_ascend_C")


def test_sparse_flash_mla_meta_output_shapes():
    q = torch.empty((2, 4, 512), dtype=torch.bfloat16, device="meta")
    cmp_kv = torch.empty((4, 128, 1, 512), dtype=torch.bfloat16, device="meta")
    topk = torch.empty((2, 32), dtype=torch.int32, device="meta")
    block_table = torch.empty((2, 4), dtype=torch.int32, device="meta")
    metadata = torch.empty((1024,), dtype=torch.int32, device="meta")

    output, softmax_lse = torch.ops._C_ascend.npu_sparse_flash_mla(
        q,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=topk,
        cmp_block_table=block_table,
        metadata=metadata,
        layout_q="TND",
        layout_kv="PA_BBND",
        return_softmax_lse=True,
    )

    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert output.device.type == "meta"
    assert softmax_lse.shape == (1, 2, 4)
    assert softmax_lse.dtype == torch.float32


def test_sparse_flash_mla_metadata_meta_shape_and_dtype():
    cu_seqlens_q = torch.empty((3,), dtype=torch.int32, device="meta")
    seqused_cmp_kv = torch.empty((2,), dtype=torch.int32, device="meta")
    cmp_residual_kv = torch.empty((2,), dtype=torch.int32, device="meta")
    cmp_topk_length = torch.empty((2,), dtype=torch.int32, device="meta")

    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        4,
        1,
        512,
        cu_seqlens_q=cu_seqlens_q,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        cmp_topk_length=cmp_topk_length,
        batch_size=2,
        max_seqlen_q=1,
        max_seqlen_cmp_kv=128,
        cmp_topk=32,
        cmp_ratio=4,
        cmp_mask_mode=3,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_cmp_kv=True,
    )

    assert metadata.shape == (1024,)
    assert metadata.dtype == torch.int32
    assert metadata.device.type == "meta"
