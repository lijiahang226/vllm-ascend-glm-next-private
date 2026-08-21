# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UTs for GLM-5 Indexer KPool MLA support in the v2 model runner's attn_utils.

These cover the three P0 pieces that gate GLM-5 on the v2 runner:
1. ``get_kv_cache_spec`` collects the three GLM-5 cache-role layers and the
   KDA/Mamba layers.
2. ``_allocate_kv_cache`` allocates one raw tensor per GLM-5 cache spec
   without the K/V split used by other MLA models.
3. ``_reshape_kv_cache_v2`` reshapes the main MLA, compressed indexer,
   compressor state and Mamba caches with their page-padding semantics.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.attention.indexer_kpool_mla_v1 import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolMLABackend,
    AscendIndexerKPoolStateBackend,
)
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolStateSpec
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _is_glm5_indexer_kpool_cache_spec,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)

GLM5_MAIN_SPEC = MLAAttentionSpec(
    block_size=384,
    num_kv_heads=1,
    head_size=576,
    dtype=torch.bfloat16,
    compress_ratio=1,
    model_version="glm5_next",
)
GLM5_INDEXER_SPEC = MLAAttentionSpec(
    block_size=384,
    num_kv_heads=1,
    head_size=128,
    dtype=torch.bfloat16,
    compress_ratio=16,
    model_version="glm5_next",
)
GLM5_STATE_SPEC = AscendIndexerKPoolStateSpec(
    block_size=16,
    num_kv_heads=1,
    head_size=256,
    dtype=torch.bfloat16,
    sliding_window=16,
    cache_role="indexer_state",
    model_version="glm5_next",
)
GLM5_MAMBA_SPEC = MambaSpec(
    block_size=384,
    shapes=((3, 1536), (4, 128, 128)),
    dtypes=(torch.bfloat16, torch.float32),
    num_speculative_blocks=0,
)

# The GLM-5 cache planner aligns specs into two physical page-size classes
# (patch_kv_cache_utils._align_glm5_cache_specs): the main class covers the
# MLA and Mamba pages, the small class covers the indexer and state pages.
MAIN_PAGE_SIZE = GLM5_MAIN_SPEC.page_size_bytes
SMALL_PAGE_SIZE = max(GLM5_INDEXER_SPEC.page_size_bytes, GLM5_STATE_SPEC.page_size_bytes)
assert MAIN_PAGE_SIZE >= GLM5_MAMBA_SPEC.page_size_bytes

GLM5_MAIN_SPEC_PADDED = replace(GLM5_MAIN_SPEC, page_size_padded=MAIN_PAGE_SIZE)
GLM5_INDEXER_SPEC_PADDED = replace(GLM5_INDEXER_SPEC, page_size_padded=SMALL_PAGE_SIZE)
GLM5_STATE_SPEC_PADDED = replace(GLM5_STATE_SPEC, page_size_padded=SMALL_PAGE_SIZE)
GLM5_MAMBA_SPEC_PADDED = replace(GLM5_MAMBA_SPEC, page_size_padded=MAIN_PAGE_SIZE)


def _make_glm5_layer(name: str, spec, cache_role: str):
    layer = MagicMock()
    layer.cache_role = cache_role
    layer.get_kv_cache_spec.return_value = spec
    layer.get_attn_backend.return_value = {
        "kv": AscendIndexerKPoolMLABackend,
        "indexer": AscendIndexerKPoolBackend,
        "state": AscendIndexerKPoolStateBackend,
    }[cache_role]
    layer.layer_name = name
    return layer


def _make_mamba_layer(name: str):
    layer = MagicMock()
    layer.get_kv_cache_spec.return_value = GLM5_MAMBA_SPEC
    layer.get_attn_backend.return_value = MagicMock()
    layer.layer_name = name
    return layer


def _make_glm5_kv_cache_config(num_blocks: int = 64) -> KVCacheConfig:
    """Build a GLM-5 KVCacheConfig with the two-page-class tensor plan."""
    main_names = [f"model.layers.{i}.self_attn.attn" for i in range(0, 44, 4)]
    indexer_names = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in range(0, 44, 4)]
    state_names = [
        f"model.layers.{i}.self_attn.indexer.compressor.state_cache" for i in range(0, 44, 4)
    ]
    mamba_names = [f"model.layers.{i}.mamba" for i in range(0, 44, 4)]

    full_specs = {}
    for main_name, indexer_name in zip(main_names, indexer_names):
        full_specs[main_name] = GLM5_MAIN_SPEC_PADDED
        full_specs[indexer_name] = GLM5_INDEXER_SPEC_PADDED
    state_specs = {name: GLM5_STATE_SPEC_PADDED for name in state_names}
    mamba_specs = {name: GLM5_MAMBA_SPEC_PADDED for name in mamba_names}

    kv_cache_tensors = [
        KVCacheTensor(size=MAIN_PAGE_SIZE * num_blocks, shared_by=[name])
        for name in main_names
    ]
    kv_cache_tensors += [
        KVCacheTensor(size=SMALL_PAGE_SIZE * num_blocks, shared_by=[indexer_name, state_name])
        for indexer_name, state_name in zip(indexer_names, state_names)
    ]
    kv_cache_tensors += [
        KVCacheTensor(size=MAIN_PAGE_SIZE * num_blocks, shared_by=[name]) for name in mamba_names
    ]

    groups = [
        KVCacheGroupSpec(
            [name for pair in zip(main_names, indexer_names) for name in pair],
            UniformTypeKVCacheSpecs.from_specs(full_specs),
        ),
        KVCacheGroupSpec(state_names, UniformTypeKVCacheSpecs.from_specs(state_specs)),
        KVCacheGroupSpec(mamba_names, UniformTypeKVCacheSpecs.from_specs(mamba_specs)),
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=groups,
    )


def _make_vllm_config():
    return SimpleNamespace(
        kv_transfer_config=None,
        quant_config=None,
        model_config=SimpleNamespace(),
    )


def test_is_glm5_indexer_kpool_cache_spec():
    assert _is_glm5_indexer_kpool_cache_spec(GLM5_MAIN_SPEC)
    assert _is_glm5_indexer_kpool_cache_spec(GLM5_INDEXER_SPEC)
    assert _is_glm5_indexer_kpool_cache_spec(GLM5_STATE_SPEC)
    assert not _is_glm5_indexer_kpool_cache_spec(GLM5_MAMBA_SPEC)
    assert not _is_glm5_indexer_kpool_cache_spec(
        MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        )
    )


def test_get_kv_cache_spec_collects_glm5_cache_roles_and_mamba():
    layers = {
        "model.layers.0.self_attn.attn": _make_glm5_layer(
            "model.layers.0.self_attn.attn", GLM5_MAIN_SPEC, "kv"
        ),
        "model.layers.0.self_attn.indexer.k_cache": _make_glm5_layer(
            "model.layers.0.self_attn.indexer.k_cache", GLM5_INDEXER_SPEC, "indexer"
        ),
        "model.layers.0.self_attn.indexer.compressor.state_cache": _make_glm5_layer(
            "model.layers.0.self_attn.indexer.compressor.state_cache",
            GLM5_STATE_SPEC,
            "state",
        ),
        "model.layers.0.mamba": _make_mamba_layer("model.layers.0.mamba"),
    }
    vllm_config = _make_vllm_config()
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config",
        return_value=layers,
    ):
        specs = get_kv_cache_spec(vllm_config)

    assert specs["model.layers.0.self_attn.attn"] is GLM5_MAIN_SPEC
    assert specs["model.layers.0.self_attn.indexer.k_cache"] is GLM5_INDEXER_SPEC
    assert specs["model.layers.0.self_attn.indexer.compressor.state_cache"] is GLM5_STATE_SPEC
    assert specs["model.layers.0.mamba"] is GLM5_MAMBA_SPEC


def test_get_kv_cache_spec_rejects_non_glm5_cache_role_spec():
    layer = _make_glm5_layer(
        "model.layers.0.self_attn.attn",
        MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        ),
        "kv",
    )
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config",
        return_value={"model.layers.0.self_attn.attn": layer},
    ):
        with pytest.raises(TypeError, match="GLM-5 MLAAttentionSpec"):
            get_kv_cache_spec(_make_vllm_config())


def test_allocate_kv_cache_glm5_allocates_one_tensor_per_spec():
    kv_cache_config = _make_glm5_kv_cache_config(num_blocks=64)
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=_make_vllm_config(),
    ):
        raw_tensors = _allocate_kv_cache(kv_cache_config, {}, torch.device("cpu"))

    # Every layer must be bound to exactly one raw tensor (no K/V split).
    for tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in tensor.shared_by:
            assert isinstance(raw_tensors[layer_name], torch.Tensor)
            assert raw_tensors[layer_name].numel() == tensor.size
    # The small page class is shared between the indexer and the state cache.
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    state_name = "model.layers.0.self_attn.indexer.compressor.state_cache"
    assert raw_tensors[indexer_name] is raw_tensors[state_name]
    # The main MLA page class is shared between the MLA and the Mamba layers.
    main_name = "model.layers.0.self_attn.attn"
    mamba_name = "model.layers.0.mamba"
    assert raw_tensors[main_name] is raw_tensors[mamba_name]


def _make_attn_group(backend, layer_names, spec, group_id):
    return SimpleNamespace(
        backend=backend,
        layer_names=layer_names,
        kv_cache_spec=spec,
        kv_cache_group_id=group_id,
    )


def test_reshape_kv_cache_v2_glm5_main_mla():
    kv_cache_config = _make_glm5_kv_cache_config(num_blocks=64)
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=_make_vllm_config(),
    ):
        raw_tensors = _allocate_kv_cache(kv_cache_config, {}, torch.device("cpu"))

    main_name = "model.layers.0.self_attn.attn"
    group = _make_attn_group(
        AscendIndexerKPoolMLABackend,
        [main_name],
        GLM5_MAIN_SPEC_PADDED,
        0,
    )
    kv_caches = _reshape_kv_cache_v2(
        [group],
        raw_tensors,
        cache_dtype="auto",
        kernel_block_sizes=[128],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    cache = kv_caches[main_name]
    # 64 scheduler blocks of 384 tokens -> 192 C128 kernel blocks.
    assert cache.shape == (64 * 3, 128, 1, 576)
    assert cache.dtype == torch.bfloat16


def test_reshape_kv_cache_v2_glm5_indexer_and_state():
    kv_cache_config = _make_glm5_kv_cache_config(num_blocks=64)
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=_make_vllm_config(),
    ):
        raw_tensors = _allocate_kv_cache(kv_cache_config, {}, torch.device("cpu"))

    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    state_name = "model.layers.0.self_attn.indexer.compressor.state_cache"
    groups = [
        _make_attn_group(
            AscendIndexerKPoolBackend,
            [indexer_name],
            GLM5_INDEXER_SPEC_PADDED,
            0,
        ),
        _make_attn_group(
            AscendIndexerKPoolStateBackend,
            [state_name],
            GLM5_STATE_SPEC_PADDED,
            1,
        ),
    ]
    kv_caches = _reshape_kv_cache_v2(
        groups,
        raw_tensors,
        cache_dtype="auto",
        kernel_block_sizes=[128, 16],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    # Compressed indexer: 64 blocks x 24 storage tokens x 1 head x 128 dims.
    assert kv_caches[indexer_name].shape == (64, 24, 1, 128)
    # Compressor state: 64 blocks x 16 tokens x 256 dims (3D layout).
    assert kv_caches[state_name].shape == (64, 16, 256)


def test_reshape_kv_cache_v2_glm5_mamba():
    kv_cache_config = _make_glm5_kv_cache_config(num_blocks=64)
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=_make_vllm_config(),
    ):
        raw_tensors = _allocate_kv_cache(kv_cache_config, {}, torch.device("cpu"))

    mamba_name = "model.layers.0.mamba"
    group = _make_attn_group(
        MagicMock(),
        [mamba_name],
        GLM5_MAMBA_SPEC_PADDED,
        2,
    )
    kv_caches = _reshape_kv_cache_v2(
        [group],
        raw_tensors,
        cache_dtype="auto",
        kernel_block_sizes=[128, 16, 384],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    state_tensors = kv_caches[mamba_name]
    assert isinstance(state_tensors, list)
    assert len(state_tensors) == 2
    assert state_tensors[0].shape == (64, 3, 1536)
    assert state_tensors[1].shape == (64, 4, 128, 128)
    # Both states live in the same padded page class as the main MLA.
    assert state_tensors[0].stride(0) == MAIN_PAGE_SIZE // 2


def test_ascend_bind_kv_cache_allows_multiple_caches_per_layer_index():
    from vllm_ascend.patch.worker.patch_v2.patch_attn_utils import (
        _ascend_bind_kv_cache,
    )

    # GLM-5 owns four caches per layer index; upstream raises
    # NotImplementedError for non-CUDA platforms in this case.
    kv_caches = {
        "model.layers.3.self_attn.attn": torch.zeros(2, 3),
        "model.layers.3.self_attn.indexer.k_cache": torch.zeros(2, 4),
        "model.layers.3.self_attn.indexer.compressor.state_cache": torch.zeros(2, 5),
        "model.layers.3.mamba": torch.zeros(2, 6),
        "model.layers.7.self_attn.attn": torch.zeros(2, 7),
    }
    forward_context = {name: SimpleNamespace() for name in kv_caches}
    runner_kv_caches: list[torch.Tensor] = []

    _ascend_bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    # Layer-3 caches keep their insertion order; layer 7 follows.
    assert runner_kv_caches == [kv_caches[name] for name in kv_caches]
    # Every layer's kv_cache attribute is bound.
    for name, cache in kv_caches.items():
        assert forward_context[name].kv_cache is cache
