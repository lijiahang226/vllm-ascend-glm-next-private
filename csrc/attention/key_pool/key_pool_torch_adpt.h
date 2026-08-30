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
#ifndef KEY_POOL_TORCH_ADPT_H
#define KEY_POOL_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_key_pool(
    const at::Tensor& hidden_states, const at::Tensor& wk, const at::Tensor& gate_weight,
    const at::Tensor& ape, at::Tensor& state_cache, const at::Tensor& cache_block_table,
    const at::Tensor& start_pos, const c10::optional<at::Tensor>& norm_weight,
    const c10::optional<at::Tensor>& norm_bias, const c10::optional<at::Tensor>& cos,
    const c10::optional<at::Tensor>& sin, const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& seqused, int64_t cmp_ratio, double norm_eps,
    int64_t rotary_mode)
{
    TORCH_CHECK(hidden_states.defined(), "Check hidden_states != nullptr failed");
    TORCH_CHECK(wk.defined(), "Check wk != nullptr failed");
    auto hidden_states_dim = hidden_states.dim();
    TORCH_CHECK(hidden_states_dim == 2 || hidden_states_dim == 3,
                "hidden_states dim num[", hidden_states_dim, "] should be 2 or 3");
    TORCH_CHECK(wk.dim() == 2, "wk dim num[", wk.dim(), "] should be 2");
    TORCH_CHECK(state_cache.dim() == 3, "state_cache dim num[", state_cache.dim(),
                "] should be 3");
    TORCH_CHECK(cache_block_table.dim() == 2, "cache_block_table dim num[",
                cache_block_table.dim(), "] should be 2");
    TORCH_CHECK(cmp_ratio > 0, "cmp_ratio should be greater than 0");
    TORCH_CHECK(norm_weight.has_value() == norm_bias.has_value(),
                "norm_weight and norm_bias must be provided as a pair");
    TORCH_CHECK(cos.has_value() == sin.has_value(),
                "cos and sin must be provided as a pair");
    TORCH_CHECK(!cos.has_value(), "KeyPool RoPE is not implemented in this stage");
    TORCH_CHECK(hidden_states_dim != 2 || cu_seqlens.has_value(),
                "cu_seqlens is required for TH layout");
    TORCH_CHECK(hidden_states_dim != 3 || !cu_seqlens.has_value(),
                "cu_seqlens must be absent for BSH layout");
    TORCH_CHECK(!seqused.has_value(), "seqused is reserved and must be None in this stage");

    int64_t pcap = (cache_block_table.size(1) * state_cache.size(1) + cmp_ratio - 1) / cmp_ratio;
    at::Tensor pooled_key =
        at::zeros({cache_block_table.size(0), pcap, wk.size(0)},
                  hidden_states.options().dtype(hidden_states.dtype()));

    // state_cache 支持非连续(0 轴 stride 由 aclnnKeyPool 的 stateCacheStrideDim0
    // 属性下传, 与 wheel 封装一致)。
    int64_t state_cache_stride_dim0 = state_cache.stride(0);

    EXEC_NPU_CMD(aclnnKeyPool, hidden_states, wk, gate_weight, ape, state_cache,
                 cache_block_table, start_pos, norm_weight, norm_bias, cos, sin,
                 cu_seqlens, seqused, cmp_ratio, norm_eps, rotary_mode,
                 state_cache_stride_dim0, pooled_key);
    return pooled_key;
}

}  // namespace vllm_ascend

#endif  // KEY_POOL_TORCH_ADPT_H
