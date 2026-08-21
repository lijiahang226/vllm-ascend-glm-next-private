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
"""Ascend MTP speculator for the v2 model runner.

MTP shares the ``AutoRegressiveSpeculator`` base and the ``load_eagle_model``
draft loader with Eagle; all Ascend-specific behavior is provided by
``AscendSpeculatorMixin``.
"""

from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator  # type: ignore[import-not-found]

from vllm_ascend.worker.v2.spec_decode.base import AscendSpeculatorMixin


class AscendMTPSpeculator(AscendSpeculatorMixin, MTPSpeculator):
    """MTP speculator for Ascend NPUs on the v2 model runner."""
