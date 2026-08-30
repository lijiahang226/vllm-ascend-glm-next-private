/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef POOL_KEY_INDEXER_TORCH_ADPT_H
#define POOL_KEY_INDEXER_TORCH_ADPT_H

#include <cstdint>
#include <string>
#include <tuple>

namespace vllm_ascend {

// pool_tail_k / actual_seq_q / actual_seq_k are ValueDepend(OPTIONAL) in the op
// definition, so the auto-generated aclnn API expects aclIntArray* (not
// aclTensor*). Convert 1-D int64 tensors to at::IntArrayRef so ConvertType
// produces aclIntArray*.
std::tuple<at::Tensor, at::Tensor> npu_pool_key_indexer(
    const at::Tensor& query, const at::Tensor& pool_key, const at::Tensor& weights,
    const at::Tensor& pool_tail_k, const c10::optional<at::Tensor>& actual_seq_q,
    const c10::optional<at::Tensor>& actual_seq_k,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& q_descale,
    const c10::optional<at::Tensor>& k_descale, c10::string_view layout_q,
    c10::string_view layout_k, int64_t topk, int64_t pool_size, int64_t mask_mode,
    int64_t quant_mode, bool return_value, int64_t key_stride0)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(pool_key.numel() > 0, "Tensor pool_key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
    TORCH_CHECK(pool_tail_k.numel() > 0, "Tensor pool_tail_k is empty.");
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    TORCH_CHECK(pool_size > 0, "pool_size should be greater than 0, but now is ", pool_size);
    TORCH_CHECK(topk % pool_size == 0, "topk(", topk, ") should be divisible by pool_size(",
                pool_size, ").");

    std::string query_layout_str = std::string(layout_q);
    std::string key_layout_str = std::string(layout_k);

    int64_t indices_len = topk + pool_size - 1;
    int64_t values_len = topk / pool_size;
    at::SmallVector<int64_t, 8> indices_shape;
    at::SmallVector<int64_t, 8> values_shape;
    if (query_layout_str == "BSND") {
        indices_shape = {query.size(0), query.size(1), indices_len};
        values_shape = {query.size(0), query.size(1), values_len};
    } else {
        indices_shape = {query.size(0), indices_len};
        values_shape = {query.size(0), values_len};
    }
    at::Tensor sparse_indices_out =
        at::empty(indices_shape, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        // Spec requires sparseValuesOut to be FLOAT (not query dtype).
        sparse_values_out = at::empty(values_shape, query.options().dtype(at::kFloat));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
    }

    // ValueDepend host arrays: the op definition requires INT64 semantics.
    // The GLM-5 path passes prebuilt CPU tensors from the metadata builder, so
    // no host-device sync happens here (ACLGraph capture forbids it); a device
    // input (legacy/fallback callers) still converts with the sync overhead.
    at::Tensor pool_tail_k_cpu = (
        pool_tail_k.is_cpu()
            ? pool_tail_k.to(at::kLong).contiguous()
            : pool_tail_k.to(at::kLong).cpu().contiguous()
    );
    at::IntArrayRef pool_tail_k_arr(pool_tail_k_cpu.data_ptr<int64_t>(),
                                    pool_tail_k_cpu.numel());
    c10::optional<at::IntArrayRef> actual_seq_q_arr = c10::nullopt;
    at::Tensor actual_seq_q_cpu;
    if (actual_seq_q.has_value() && actual_seq_q.value().defined()) {
        actual_seq_q_cpu = (
            actual_seq_q.value().is_cpu()
                ? actual_seq_q.value().to(at::kLong).contiguous()
                : actual_seq_q.value().to(at::kLong).cpu().contiguous()
        );
        actual_seq_q_arr = at::IntArrayRef(actual_seq_q_cpu.data_ptr<int64_t>(),
                                           actual_seq_q_cpu.numel());
    }
    c10::optional<at::IntArrayRef> actual_seq_k_arr = c10::nullopt;
    at::Tensor actual_seq_k_cpu;
    if (actual_seq_k.has_value() && actual_seq_k.value().defined()) {
        actual_seq_k_cpu = (
            actual_seq_k.value().is_cpu()
                ? actual_seq_k.value().to(at::kLong).contiguous()
                : actual_seq_k.value().to(at::kLong).cpu().contiguous()
        );
        actual_seq_k_arr = at::IntArrayRef(actual_seq_k_cpu.data_ptr<int64_t>(),
                                           actual_seq_k_cpu.numel());
    }

    char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

    // pool_key 0-axis non-contiguous: read stride(0) directly as the
    // key_stride0 attr (missing -> -1, tiling falls back to shape-derived).
    int64_t key_stride0_attr = key_stride0;
    if (!pool_key.is_contiguous()) {
        key_stride0_attr = pool_key.stride(0);
    }

    EXEC_NPU_CMD(aclnnPoolKeyIndexer, query, pool_key, weights, pool_tail_k_arr,
                 actual_seq_q_arr, actual_seq_k_arr, block_table, q_descale, k_descale,
                 topk, pool_size, query_layout_ptr, key_layout_ptr, mask_mode, quant_mode,
                 return_value, key_stride0_attr, sparse_indices_out, sparse_values_out);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

}  // namespace vllm_ascend

#endif  // POOL_KEY_INDEXER_TORCH_ADPT_H
