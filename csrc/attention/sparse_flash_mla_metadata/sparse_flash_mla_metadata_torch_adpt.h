/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef SPARSE_FLASH_MLA_METADATA_TORCH_ADPT_H
#define SPARSE_FLASH_MLA_METADATA_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_sparse_flash_mla_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv,
    const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_ori_kv,
    int64_t max_seqlen_cmp_kv, int64_t ori_topk, int64_t cmp_topk,
    int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q,
    c10::string_view layout_kv, bool has_ori_kv, bool has_cmp_kv,
    c10::string_view device)
{
    constexpr int64_t METADATA_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    const c10::optional<at::Tensor> *inputs[] = {
        &cu_seqlens_q,      &cu_seqlens_ori_kv, &cu_seqlens_cmp_kv,
        &seqused_q,         &seqused_ori_kv,     &seqused_cmp_kv,
        &cmp_residual_kv,   &ori_topk_length,    &cmp_topk_length,
    };
    for (const auto *input : inputs) {
        if (input->has_value()) {
            output_device = input->value().device();
            break;
        }
    }
    at::Tensor metadata = at::empty(
        {METADATA_SIZE}, at::TensorOptions().dtype(at::kInt).device(output_device));

    std::string layout_q_str(layout_q);
    std::string layout_kv_str(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnSparseFlashMlaMetadata, cu_seqlens_q, cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv,
        cmp_residual_kv, ori_topk_length, cmp_topk_length, num_heads_q,
        num_heads_kv, head_dim, batch_size, max_seqlen_q,
        max_seqlen_ori_kv, max_seqlen_cmp_kv, ori_topk, cmp_topk, cmp_ratio,
        ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
        layout_q_ptr, layout_kv_ptr, has_ori_kv, has_cmp_kv, metadata);
    return metadata;
}

}  // namespace vllm_ascend

#endif  // SPARSE_FLASH_MLA_METADATA_TORCH_ADPT_H
