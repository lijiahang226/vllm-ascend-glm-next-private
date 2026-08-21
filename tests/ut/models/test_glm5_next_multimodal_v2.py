# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UTs for GLM-5-Next multimodal support on the v2 model runner.

The v2 runner's multimodal path is inherited from vLLM's GPUModelRunner:
``supports_mm_inputs`` gates the encoder cache/runner, and ``execute_model``
feeds ``scheduled_encoder_inputs`` through ``model_state.get_mm_embeddings``
into ``embed_multimodal``.  GLM-5-Next only needs its processor registered and
the ``SupportsMultiModal`` interface intact.
"""

import pytest
from vllm.model_executor.models.interfaces import SupportsMultiModal

from vllm_ascend.models.glm5_next_multimodal import (
    AscendGlm5NextForConditionalGeneration,
)


def test_glm5_next_multimodal_processor_registered():
    # register_processor attaches the factory to the model class; the v2
    # runner's supports_multimodal_inputs resolves it via
    # _create_processing_info.
    assert hasattr(AscendGlm5NextForConditionalGeneration, "_processor_factory")


def test_glm5_next_multimodal_implements_v2_mm_encoder_interface():
    # execute_mm_encoder calls model.embed_multimodal(**kwargs); the
    # implementation is inherited from Glm4vForConditionalGeneration.
    assert issubclass(AscendGlm5NextForConditionalGeneration, SupportsMultiModal)
    assert callable(getattr(AscendGlm5NextForConditionalGeneration, "embed_multimodal"))


def test_glm5_next_multimodal_has_language_model_contract():
    # model_state.get_mm_embeddings requires the language model to expose
    # requires_raw_input_tokens (defaults to False) and embed paths; the
    # class-level default keeps input_ids=None once inputs_embeds is set.
    assert not SupportsMultiModal.requires_raw_input_tokens
    assert not getattr(
        AscendGlm5NextForConditionalGeneration,
        "requires_raw_input_tokens",
        False,
    )
