/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_POOL_KEY_INDEXER_H_
#define OPS_OP_PROTO_INC_POOL_KEY_INDEXER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Selects top-k pooled keys and expands them to token indices.
 *
 * Inputs use BSND/TND query layouts and BSND/TND/PA_BBND key layouts.
 * pool_tail_k and actual_seq_q/actual_seq_k use INT64 ValueDepend semantics;
 * the generated Tensor ACLNN variant keeps device values graph-replayable.
 */
REG_OP(PoolKeyIndexer)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(pool_key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(pool_tail_k, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_q, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_k, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(q_descale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(k_descale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(sparse_indices, TensorType({DT_INT32}))
    .OUTPUT(sparse_values, TensorType({DT_FLOAT}))
    .ATTR(topk, Int, 2048)
    .ATTR(pool_size, Int, 16)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_k, String, "BSND")
    .ATTR(mask_mode, Int, 3)
    .ATTR(quant_mode, Int, -1)
    .ATTR(return_value, Bool, false)
    .ATTR(key_stride0, Int, -1)
    .OP_END_FACTORY_REG(PoolKeyIndexer)
}  // namespace ge

#endif  // OPS_OP_PROTO_INC_POOL_KEY_INDEXER_H_
