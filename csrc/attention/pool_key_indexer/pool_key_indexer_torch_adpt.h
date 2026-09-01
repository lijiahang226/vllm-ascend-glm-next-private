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

namespace vllm_ascend {

constexpr int64_t POOL_KEY_INDEXER_SIZE = 8;
constexpr int64_t POOL_KEY_INDEXER_DIM_0 = 0;
constexpr int64_t POOL_KEY_INDEXER_DIM_1 = 1;

// Derive output shapes from query shape + layout.
//   BSND: query (B,S1,N1,D) -> indices (B,S1,topk+poolSize-1) / values (B,S1,topk/poolSize)
//   TND:  query (T1,N1,D)   -> indices (T1,topk+poolSize-1)     / values (T1,topk/poolSize)
// Note: PKI output has no N2 dimension (N2 is fixed to 1), unlike LIV2 which emits keyHeadNum.
std::tuple<at::Tensor, at::Tensor> ConstructPoolKeyIndexerOutputTensor(
    const at::Tensor& query, int64_t topk, int64_t pool_size,
    const std::string& query_layout_str, bool return_value)
{
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    TORCH_CHECK(pool_size > 0,
                "pool_size should be greater than 0, but now is ", pool_size);
    TORCH_CHECK(topk % pool_size == 0, "topk(", topk,
                ") should be divisible by pool_size(", pool_size, ").");

    int64_t indices_len = topk + pool_size - 1;
    int64_t values_len = topk / pool_size;

    at::SmallVector<int64_t, POOL_KEY_INDEXER_SIZE> indices_shape;
    at::SmallVector<int64_t, POOL_KEY_INDEXER_SIZE> values_shape;

    if (query_layout_str == "BSND") {
        indices_shape = {query.size(POOL_KEY_INDEXER_DIM_0),
                         query.size(POOL_KEY_INDEXER_DIM_1), indices_len};
        values_shape = {query.size(POOL_KEY_INDEXER_DIM_0),
                        query.size(POOL_KEY_INDEXER_DIM_1), values_len};
    } else {
        indices_shape = {query.size(POOL_KEY_INDEXER_DIM_0), indices_len};
        values_shape = {query.size(POOL_KEY_INDEXER_DIM_0), values_len};
    }

    at::Tensor sparse_indices_out =
        at::empty(indices_shape, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        // Spec requires sparseValuesOut to be FLOAT (not query dtype)
        sparse_values_out =
            at::empty(values_shape, query.options().dtype(at::kFloat));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                              sparse_values_out);
}

std::vector<bool> IsContiguousAxes(const at::Tensor& tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();
    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguous_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguous_stride[i] = contiguous_stride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguous_stride[i]);
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor> npu_pool_key_indexer(
    const at::Tensor& query, const at::Tensor& pool_key,
    const at::Tensor& weights, const at::Tensor& pool_tail_k,
    const c10::optional<at::Tensor>& actual_seq_q,
    const c10::optional<at::Tensor>& actual_seq_k,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& q_descale,
    const c10::optional<at::Tensor>& k_descale, c10::string_view layout_q,
    c10::string_view layout_k, int64_t topk, int64_t pool_size,
    int64_t mask_mode, int64_t quant_mode, bool return_value)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(pool_key.numel() > 0, "Tensor pool_key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
    TORCH_CHECK(pool_tail_k.numel() > 0, "Tensor pool_tail_k is empty.");

    std::string query_layout_str = std::string(layout_q);
    std::string key_layout_str = std::string(layout_k);

    std::tuple<at::Tensor, at::Tensor> pool_key_indexer_output =
        ConstructPoolKeyIndexerOutputTensor(query, topk, pool_size,
                                            query_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(pool_key_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(pool_key_indexer_output);

    char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

    // Non-contiguous pool_key on axis 0 (following the compressor approach):
    // the eager/aclnn tiling context does not report the runtime stride, so
    // the torch extension reads at::Tensor::stride(0) directly and passes it
    // down as the key_stride0 attribute; non-zero axes must be contiguous.
    // In PA_BBND, axis-0 stride >= blockSize*N2*D means non-contiguous
    // addressing, otherwise it is equivalent to contiguous.
    int64_t key_stride0 = -1;
    if (!pool_key.is_contiguous()) {
        auto contiguous_axes = IsContiguousAxes(pool_key);
        bool is_pa_bbnd = (key_layout_str == "PA_BBND");
        for (int64_t i = 1; i < static_cast<int64_t>(contiguous_axes.size());
             i++) {
            TORCH_CHECK(contiguous_axes[i],
                        "pool_key only supports non-contiguous tensor on the 0-axis, axis[",
                        i, "] stride is not contiguous");
        }
        key_stride0 = pool_key.stride(0);
        if (is_pa_bbnd) {
            int64_t contiguous_stride0 =
                pool_key.size(1) * pool_key.size(2) * pool_key.size(3);
            TORCH_CHECK(key_stride0 >= contiguous_stride0,
                        "pool_key stride0(", key_stride0,
                        ") must be >= contiguous stride(", contiguous_stride0,
                        ") in PA_BBND scenarios");
        }
    }

    // pool_tail_k / actual_seq_q / actual_seq_k are plain device tensors
    // (no ValueDepend in the op definition): the generated aclnn API accepts
    // aclTensor* and the kernel reads the values from GM at runtime.  This
    // keeps the op ACLGraph-compatible — values may change between capture
    // and replay without being baked into the tiling data.
    EXEC_NPU_CMD(aclnnPoolKeyIndexer, query, pool_key, weights, pool_tail_k,
                 actual_seq_q, actual_seq_k, block_table, q_descale, k_descale,
                 topk, pool_size, query_layout_ptr, key_layout_ptr, mask_mode,
                 quant_mode, return_value, key_stride0, sparse_indices_out,
                 sparse_values_out);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                              sparse_values_out);
}
}  // namespace vllm_ascend

#endif  // POOL_KEY_INDEXER_TORCH_ADPT_H
