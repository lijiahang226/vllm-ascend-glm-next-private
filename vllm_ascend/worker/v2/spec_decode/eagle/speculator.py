# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Ascend Eagle speculator for the v2 model runner.

All Ascend-specific behavior lives in ``AscendSpeculatorMixin`` (see
``vllm_ascend.worker.v2.spec_decode.base``); Eagle only differs from MTP in
the draft-model loader, which is identical (``load_eagle_model``).
"""

from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator  # type: ignore[import-not-found]

from vllm_ascend.worker.v2.spec_decode.base import (
    AscendSpeculatorMixin,
    build_attn_metadata_wrapper,  # noqa: F401  (re-exported for compatibility)
    graph_manager_wrapper,  # noqa: F401  (re-exported for compatibility)
    torch_gather_wrapper,  # noqa: F401  (re-exported for compatibility)
)


class AscendEagleSpeculator(AscendSpeculatorMixin, EagleSpeculator):
    """Eagle speculator for Ascend NPUs on the v2 model runner."""
