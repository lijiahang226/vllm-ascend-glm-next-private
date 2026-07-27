# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.attention.indexer_kpool_mla_v1 import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolMLAImpl,
    AscendIndexerKPoolStateBackend,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolStateSpec,
    format_indexer_kpool_slot_mapping,
)
from vllm_ascend.models import register_model as register_ascend_models
from vllm_ascend.models.glm5_next import (
    AscendGlm5NextCompressorStateCache,
    AscendGlm5NextIndexer,
    AscendGlm5NextIndexerKPoolCache,
    AscendSparseAttnIndexerKpool,
    SparseAttnIndexerKpool,
)
from vllm_ascend.ops.indexer_kpool_mla import (
    AscendIndexerKPoolMLAAttention,
    IndexerKPoolMLACacheLayer,
    _collect_cache_metadata,
)
from vllm_ascend.ops.triton.kda.kda import fused_kda_gate
from vllm_ascend.patch.platform.patch_glm5_next_config import (
    Glm5NextModelArchConfigConvertor,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _create_uniform_mamba_groups,
    _get_kv_cache_config_deepseek_v4,
)
from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_mamba_target_page_size,
)
from vllm_ascend.patch.worker.patch_process_weights_after_loading import (
    _is_ascend_attention,
)
from vllm_ascend.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)


def test_indexer_kpool_mla_specs_allocate_one_physical_vector_per_role():
    mla_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        model_version="glm5_next",
    )
    indexer_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
        model_version="glm5_next",
    )
    state_spec = AscendIndexerKPoolStateSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=4,
        cache_role="indexer_state",
    )

    assert mla_spec.page_size_bytes == 128 * (512 + 64) * 2
    assert indexer_spec.storage_block_size == 32
    assert indexer_spec.page_size_bytes == 32 * 128 * 2
    assert state_spec.page_size_bytes == 4 * 256 * 2
    assert state_spec.sliding_window == 4


def test_indexer_kpool_mla_cache_roles_expose_v023_prefill_backend_sentinel():
    """Only the executable MLA cache passes through MLACommonMetadataBuilder."""
    assert IndexerKPoolMLACacheLayer.cache_role == "kv"
    assert IndexerKPoolMLACacheLayer.prefill_backend is None
    assert not hasattr(AscendGlm5NextIndexerKPoolCache, "prefill_backend")


def test_indexer_kpool_mla_participates_in_post_load_weight_processing():
    wrapper = object.__new__(AscendIndexerKPoolMLAAttention)
    assert _is_ascend_attention(wrapper)


def test_indexer_kpool_mla_collects_metadata_by_exact_cache_layer_name():
    prefix = "model.layers.0.self_attn"
    cache_layers = (
        SimpleNamespace(layer_name=f"{prefix}.attn"),
        SimpleNamespace(prefix=f"{prefix}.indexer.compressor.state_cache"),
        SimpleNamespace(prefix=f"{prefix}.indexer.k_cache"),
    )
    expected = tuple(
        SimpleNamespace(cache_role=role)
        for role in ("kv", "indexer_state", "indexer")
    )
    metadata = {
        f"{prefix}.attn": expected[0],
        f"{prefix}.indexer.compressor.state_cache": expected[1],
        f"{prefix}.indexer.k_cache": expected[2],
        # 同一 attention 前缀下的其他 metadata 不能传给组合实现。
        f"{prefix}.unrelated_cache": SimpleNamespace(cache_role="kv"),
    }

    wrapper = SimpleNamespace(prefix=prefix, cache_layers=cache_layers)
    assert _collect_cache_metadata(wrapper, metadata) == expected


def test_indexer_kpool_state_uses_independent_backend_for_four_token_pages():
    assert AscendGlm5NextCompressorStateCache.get_attn_backend(None) is AscendIndexerKPoolStateBackend
    assert select_common_block_size(4, [AscendIndexerKPoolStateBackend]) == 4
    assert AscendIndexerKPoolStateBackend.get_kv_cache_shape(8, 4, 1, 256) == (8, 4, 256)


def test_indexer_kpool_cache_uses_minimal_independent_metadata_builder():
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
        model_version="glm5_next",
    )
    builder = AscendIndexerKPoolMetadataBuilder(
        spec,
        ["model.layers.0.self_attn.indexer.k_cache"],
        SimpleNamespace(),
        torch.device("cpu"),
    )
    common_metadata = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=4,
        positions=torch.tensor([0, 1, 2, 3]),
        slot_mapping=torch.tensor([7 * 128 + offset for offset in range(4)]),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        seq_lens_cpu=None,
        block_table_tensor=torch.tensor([[7]], dtype=torch.int32),
    )

    metadata = builder.build(0, common_metadata)

    assert not issubclass(AscendIndexerKPoolMetadataBuilder, AscendSFAMetadataBuilder)
    assert AscendGlm5NextIndexerKPoolCache.get_attn_backend(None) is AscendIndexerKPoolBackend
    assert select_common_block_size(128, [AscendIndexerKPoolBackend]) == 128
    assert metadata.block_size == 32
    assert metadata.slot_mapping.tolist() == [-1, -1, -1, 7 * 32]
    assert metadata.seq_lens.tolist() == [1]
    assert metadata.seq_lens_cpu.tolist() == [1]
    assert metadata.block_table.tolist() == [[7]]


def test_glm5_latest_config_schema_drives_attention_and_mlp_layout():
    config = Glm5NextTextConfig(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        linear_head_dim=128,
        linear_num_heads=64,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        qk_rope_head_dim=0,
        index_kpool=4,
    )

    assert config.model_type == "glm5_next_text"
    assert config.layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert config.mlp_layer_types == ["dense", "dense", "dense", "sparse"]
    assert config.linear_head_dim == 128
    assert config.linear_num_heads == 64
    assert config.linear_conv_kernel_dim == 4
    assert config.linear_lower_bound == -5.0
    assert config.qk_rope_head_dim == 0
    assert config.index_kpool == 4


def test_glm5_model_arch_config_uses_mla_cache_dimensions():
    text_config = Glm5NextTextConfig(
        head_dim=0,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
    )
    config = Glm5NextConfig(text_config=text_config.to_dict())
    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS["glm5_next"]
    convertor = convertor_cls(config, config.text_config)
    model_arch_config = convertor.convert()

    assert convertor_cls is Glm5NextModelArchConfigConvertor
    assert convertor.is_deepseek_mla()
    assert convertor.get_head_size() == 512
    assert model_arch_config.is_deepseek_mla
    assert model_arch_config.head_size == 512


def test_glm5_kda_page_covers_ssm_and_conv_states():
    target_page_size = _get_mamba_target_page_size(
        is_glm5_next=True,
        attn_page_size=262144,
        mamba_raw_size=271360,
        conv_block_page_size=9216,
    )

    assert target_page_size == 271360


def test_glm5_mamba_groups_use_uniform_type_wrapper():
    mamba_specs = {
        f"layer.{layer_idx}": MambaSpec(
            block_size=256,
            shapes=((8, 128, 128), (6, 3, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=271360,
        )
        for layer_idx in (0, 3)
    }

    groups = _create_uniform_mamba_groups(
        mamba_specs,
        [["layer.0", "layer.3"]],
    )

    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    assert groups[0].layer_names == ["layer.0", "layer.3"]


def test_glm5_auto_config_reads_registered_nested_text_config(tmp_path):
    config_dict = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 4,
            "num_experts_per_tok": 8,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
            "index_kpool": 4,
            "index_topk": 2048,
        },
        "vision_config": {
            "model_type": "glm_ocr_vision",
            "depth": 24,
            "hidden_size": 1024,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 4096,
            "projection_intermediate_size": 10240,
        },
        "image_token_id": 154854,
        "video_token_id": 154855,
        "image_start_token_id": 154830,
        "image_end_token_id": 154831,
        "video_start_token_id": 154832,
        "video_end_token_id": 154833,
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))

    config = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(config, Glm5NextConfig)
    assert isinstance(config.text_config, Glm5NextTextConfig)
    assert isinstance(config.vision_config, Glm5NextVisionConfig)
    assert config.model_type == "glm5_next"
    assert config.architectures == ["Glm5NextForConditionalGeneration"]
    assert config.text_config.model_type == "glm5_next_text"
    assert config.text_config.num_experts_per_token == 8
    assert config.text_config.index_kpool == 4
    assert config.text_config.index_topk == 2048
    assert config.vision_config.depth == 24
    assert config.vision_config.hidden_size == 1024
    assert config.vision_config.num_heads == 16
    assert config.vision_config.patch_size == 14
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.temporal_patch_size == 2
    assert config.vision_config.out_hidden_size == 4096
    assert config.vision_config.projection_intermediate_size == 10240
    assert config.image_token_id == 154854
    assert config.video_token_id == 154855
    assert config.image_start_token_id == 154830
    assert config.image_end_token_id == 154831
    assert config.video_start_token_id == 154832
    assert config.video_end_token_id == 154833


def test_glm5_conditional_generation_architecture_uses_text_only_ascend_model(
    monkeypatch,
):
    registered_models = {}
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        lambda architecture, model_cls: registered_models.__setitem__(
            architecture,
            model_cls,
        ),
    )

    register_ascend_models()

    assert registered_models["Glm5NextForConditionalGeneration"] == (
        "vllm_ascend.models.glm5_next:AscendGlm5NextForCausalLM"
    )
    assert registered_models["Glm5NextForConditionalGeneration"] == registered_models["Glm5NextForCausalLM"]


def test_glm5_kda_gate_exposes_bounded_gate_parameters():
    parameters = inspect.signature(fused_kda_gate).parameters
    assert parameters["safe_gate"].default is False
    assert parameters["lower_bound"].default == -5.0


def test_indexer_kpool_mla_nope_rope_single_is_identity():
    hidden = torch.randn(2, 3)
    result = AscendIndexerKPoolMLAImpl.rope_single(
        SimpleNamespace(qk_rope_head_dim=0),
        hidden,
        torch.empty(2, 1, 1, 0),
        torch.empty(2, 1, 1, 0),
    )

    assert result is hidden


def test_indexer_kpool_mla_nope_packs_only_latent_cache_values():
    op = SimpleNamespace(
        kv_lora_rank=4,
        qk_rope_head_dim=0,
    )
    kv_c = torch.arange(12, dtype=torch.bfloat16).reshape(3, 1, 4)
    k_pe = torch.empty(3, 1, 0, dtype=torch.bfloat16)

    result = AscendIndexerKPoolMLAImpl._pack_mla_cache_values(
        op,
        kv_c,
        k_pe,
        num_tokens=3,
    )

    assert result.shape == (3, 4)
    assert torch.equal(result, kv_c.reshape(3, 4))


def test_indexer_kpool_mla_state_block_table_maps_request_tail_pages():
    op = SimpleNamespace(head_dim=1)
    state_cache = torch.zeros((4, 4, 2), dtype=torch.bfloat16)
    state_cache[2] = torch.tensor([[20.0, 200.0], [21.0, 201.0], [22.0, 202.0], [23.0, 203.0]])
    state_cache[3] = torch.tensor([[30.0, 300.0], [31.0, 301.0], [32.0, 302.0], [33.0, 303.0]])
    metadata = SimpleNamespace(
        block_size=4,
        # Both requests gather absolute logical positions 4..7, but their
        # current tail pages are owned by different physical blocks. The old
        # page entry can be released/null without changing the logical index.
        block_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
    )

    result = SparseAttnIndexerKpool._gather_compressor_state(
        op,
        state_cache,
        metadata,
        end_positions=torch.tensor([7, 7]),
        request_ids=torch.tensor([0, 1]),
        index_kpool=4,
    )

    assert result[:, :, 0].tolist() == [
        [20.0, 21.0, 22.0, 23.0],
        [30.0, 31.0, 32.0, 33.0],
    ]


def test_indexer_kpool_mla_standalone_mtp_allocator_keeps_every_cache_role():
    specs = {
        "layer.attn": MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "layer.indexer.k_cache": MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=4,
            model_version="glm5_next",
        ),
        "layer.indexer.compressor.state_cache": AscendIndexerKPoolStateSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=4,
            cache_role="indexer_state",
            model_version="glm5_next",
        ),
    }
    groups = [
        KVCacheGroupSpec(
            list(group_specs),
            UniformTypeKVCacheSpecs(
                block_size=next(iter(group_specs.values())).block_size,
                kv_cache_specs=group_specs,
            ),
        )
        for group_specs in (
            {"layer.attn": specs["layer.attn"]},
            {"layer.indexer.k_cache": specs["layer.indexer.k_cache"]},
            {"layer.indexer.compressor.state_cache": specs["layer.indexer.compressor.state_cache"]},
        )
    ]
    bytes_per_block = sum(spec.page_size_bytes for spec in specs.values())
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config,
        groups,
        available_memory=bytes_per_block * 3,
    )

    assert num_blocks == 3
    assert {tensor.shared_by[0] for tensor in tensors} == set(specs)
    assert {tensor.shared_by[0]: tensor.size for tensor in tensors} == {
        name: spec.page_size_bytes * num_blocks for name, spec in specs.items()
    }


def test_indexer_kpool_mla_compressed_slot_mapping_only_writes_completed_pools():
    slots = torch.tensor([0, 14, 15, 16, 127, 128, 143, -1])
    positions = torch.tensor([0, 14, 15, 16, 127, 128, 143, 15])

    result = format_indexer_kpool_slot_mapping(
        slots,
        positions,
        logical_block_size=128,
        compress_ratio=16,
    )

    assert result.tolist() == [-1, -1, 0, -1, 7, -1, 8, -1]


def test_indexer_kpool_mla_compressed_slot_mapping_rejects_partial_block_pool():
    with pytest.raises(ValueError, match="must be divisible"):
        format_indexer_kpool_slot_mapping(
            torch.tensor([0]),
            torch.tensor([0]),
            logical_block_size=128,
            compress_ratio=10,
        )


def test_indexer_kpool_mla_expand_pools_to_tokens_matches_vllm_contract():
    pool_ids = torch.tensor(
        [
            [2, -1],
            [0, 1],
        ],
        dtype=torch.int32,
    )

    result = AscendSparseAttnIndexerKpool.expand_pools_to_tokens(
        pool_ids,
        pool_ids >= 0,
        topk=8,
        pool_size=4,
    )

    assert result.dtype == torch.int32
    assert result.tolist() == [
        [8, 9, 10, 11, -1, -1, -1, -1],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


def test_indexer_kpool_mla_append_tail_to_topk_uses_per_query_lengths():
    history = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )

    result = AscendSparseAttnIndexerKpool.append_tail_to_topk(
        history,
        seq_lens=torch.tensor([11, 16], dtype=torch.int32),
        pool_lens=torch.tensor([2, 4], dtype=torch.int32),
        pool_size=4,
    )

    assert result.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1],
    ]


def test_indexer_kpool_mla_kpool_topk_requires_exact_pool_budget():
    with pytest.raises(ValueError, match="must be divisible"):
        AscendSparseAttnIndexerKpool.history_group_budget_for_topk(
            topk=10,
            pool_size=4,
        )


def test_indexer_kpool_mla_kpool_topk_budget_is_cached_during_initialization():
    op = AscendSparseAttnIndexerKpool(
        k_cache=SimpleNamespace(),
        quant_block_size=128,
        scale_fmt=None,
        topk_tokens=2048,
        head_dim=128,
        max_model_len=4096,
        max_total_seq_len=4096,
        topk_indices_buffer=None,
        state_cache=SimpleNamespace(compress_ratio=4),
        attn_layer_name="layer.attn",
    )

    assert op.pool_topk == 512


def test_indexer_kpool_mla_indexer_small_ops_use_bfloat16_cache_contract():
    assert list(inspect.signature(AscendSparseAttnIndexerKpool.cp_gather_indexer_k_cache).parameters) == [
        "kv_cache",
        "dst_k",
        "block_table",
        "cu_seq_lens",
    ]
    assert list(inspect.signature(AscendSparseAttnIndexerKpool.bf16_mqa_logits).parameters) == [
        "query",
        "key",
        "weights",
        "cu_seqlen_ks",
        "cu_seqlen_ke",
        "clean_logits",
    ]
    assert list(inspect.signature(AscendSparseAttnIndexerKpool.top_k_per_row_prefill).parameters) == [
        "logits",
        "cu_seqlen_ks",
        "cu_seqlen_ke",
        "raw_topk_indices",
        "num_rows",
        "stride0",
        "stride1",
        "topk_tokens",
    ]


def test_glm5_indexer_class_keeps_upstream_forward_contracts():
    expected_op_forward = [
        "self",
        "hidden_states",
        "q_quant",
        "k",
        "weights",
        "gate_score",
        "compress_ape",
        "index_kpool",
        "positions",
    ]
    assert list(inspect.signature(SparseAttnIndexerKpool.forward_native).parameters) == expected_op_forward
    assert list(inspect.signature(SparseAttnIndexerKpool.forward_ascend).parameters) == expected_op_forward
    assert list(inspect.signature(AscendGlm5NextIndexer.forward).parameters) == [
        "self",
        "hidden_states",
        "qr",
        "positions",
        "rotary_emb",
    ]


def test_indexer_kpool_mla_bf16_mqa_logits_matches_weighted_query_key_product():
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    cu_seqlen_ks = torch.tensor([0, 1], dtype=torch.int32)
    cu_seqlen_ke = torch.tensor([3, 4], dtype=torch.int32)

    result = AscendSparseAttnIndexerKpool.bf16_mqa_logits(
        query,
        key,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
    )

    weighted_query = (query * weights.unsqueeze(-1)).sum(dim=1)
    expected = (weighted_query @ key.T).float()
    torch.testing.assert_close(result, expected)


def test_indexer_kpool_mla_top_k_per_row_prefill_returns_sequence_relative_indices():
    logits = torch.tensor(
        [
            [1.0, 5.0, 3.0, 100.0, 100.0],
            [100.0, 100.0, 2.0, 7.0, 4.0],
        ],
        dtype=torch.float32,
    )
    cu_seqlen_ks = torch.tensor([0, 2], dtype=torch.int32)
    cu_seqlen_ke = torch.tensor([3, 5], dtype=torch.int32)
    output = torch.empty((2, 2), dtype=torch.int32)

    AscendSparseAttnIndexerKpool.top_k_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        2,
    )

    assert set(output[0].tolist()) == {1, 2}
    assert set(output[1].tolist()) == {1, 2}


def test_indexer_kpool_mla_indexer_kpool_topk_matches_weighted_mqa_on_paged_cache():
    logical_keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
            [3.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    # Compressed logical pages [0, 1, 2] map to physical pages [2, 0, 1].
    key_cache = torch.zeros((3, 2, 1, 2), dtype=torch.bfloat16)
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    for pool_id, logical_key in enumerate(logical_keys):
        logical_page, offset = divmod(pool_id, 2)
        physical_page = int(block_table[0, logical_page])
        key_cache[physical_page, offset, 0] = logical_key

    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )

    result = AscendSparseAttnIndexerKpool.indexer_kpool_topk_pytorch(
        query=query,
        key=key_cache,
        weights=weights,
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([5], dtype=torch.int32),
        block_table=block_table,
        query_positions=torch.tensor([5, 9], dtype=torch.int64),
        sparse_count=2,
        pool_size=2,
        max_key_seq_len=5,
        query_chunk_size=1,
        key_chunk_size=2,
    )

    # Query 0 sees only pools [0, 3), while query 1 sees all five pools.
    assert set(result[0].tolist()) == {1, 2}
    assert set(result[1].tolist()) == {1, 3}


def test_indexer_kpool_mla_indexer_kpool_topk_pads_when_history_is_short():
    result = AscendSparseAttnIndexerKpool.indexer_kpool_topk_pytorch(
        query=torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16),
        key=torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.bfloat16),
        weights=torch.ones((1, 1), dtype=torch.bfloat16),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_positions=torch.tensor([1], dtype=torch.int64),
        sparse_count=4,
        pool_size=2,
        max_key_seq_len=2,
        query_chunk_size=1,
        key_chunk_size=2,
    )

    assert result.shape == (1, 4)
    assert result[result >= 0].tolist() == [0]


def test_indexer_kpool_mla_kpool_compress_returns_bfloat16_without_quant_scale():
    indexer_cache = torch.zeros((1, 2, 1, 2), dtype=torch.bfloat16)
    slot_k = torch.tensor(
        [[[1.0, 3.0], [3.0, 1.0]]],
        dtype=torch.bfloat16,
    )
    slot_score = torch.zeros_like(slot_k)
    compress_ape = torch.zeros((2, 2), dtype=torch.float32)

    compressed_k = AscendSparseAttnIndexerKpool.kpool_compress_and_write_cache(
        indexer_cache,
        slot_k,
        slot_score,
        compress_ape,
        torch.tensor([0], dtype=torch.int64),
        pool_size=2,
        head_dim=2,
        return_compressed=True,
        write_cache=False,
    )

    assert isinstance(compressed_k, torch.Tensor)
    assert compressed_k.dtype == torch.bfloat16
    torch.testing.assert_close(
        compressed_k,
        torch.tensor([[2.0, 2.0]], dtype=torch.bfloat16),
    )


def test_indexer_kpool_mla_sparse_attention_pytorch_matches_golden_semantics():
    # Logical token order is [0, 1, 2, 3], while the physical pages are
    # deliberately shuffled by block_table=[1, 0].
    logical_kv = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    packed_cache = torch.empty((2, 2, 1, 3), dtype=torch.float32)
    packed_cache[1, :, 0, :] = logical_kv[:2]
    packed_cache[0, :, 0, :] = logical_kv[2:]
    latent_view, rope_view = packed_cache.split([2, 1], dim=-1)
    assert not latent_view.is_contiguous()
    assert not rope_view.is_contiguous()

    ql_nope = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[9.0, 9.0]],  # graph-padding row
        ],
        dtype=torch.float32,
    )
    q_pe = torch.tensor(
        [
            [[1.0]],
            [[-1.0]],
            [[9.0]],
        ],
        dtype=torch.float32,
    )
    topk_indices = torch.tensor(
        [
            [[3, 2, 0, -1]],
            [[3, 2, 0, -1]],
            [[0, 1, 2, 3]],
        ],
        dtype=torch.int32,
    )

    result = AscendIndexerKPoolMLAImpl._sparse_attention_pytorch(
        ql_nope=ql_nope,
        q_pe=q_pe,
        packed_kv_cache=packed_cache,
        topk_indices=topk_indices,
        block_table=torch.tensor([[1, 0]], dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([4], dtype=torch.int32),
        scale=0.5,
        num_actual_tokens=2,
        query_chunk_size=1,
    )

    expected = torch.zeros_like(ql_nope)
    query_full = torch.cat([ql_nope[:2], q_pe[:2]], dim=-1)
    # Golden sparse-mode 3 threshold:
    # row 0: 4 - 2 + 0 + 1 = 3 -> token 3 is causally masked.
    # row 1: 4 - 2 + 1 + 1 = 4 -> token 3 is visible.
    selected_per_query = ([2, 0], [3, 2, 0])
    for query_idx, selected in enumerate(selected_per_query):
        selected_tensor = logical_kv[selected]
        scores = (query_full[query_idx, 0] @ selected_tensor.transpose(0, 1)) * 0.5
        probabilities = torch.softmax(scores, dim=-1)
        expected[query_idx, 0] = probabilities @ selected_tensor[:, :2]

    torch.testing.assert_close(result, expected)


def test_indexer_kpool_mla_sparse_attention_pytorch_maps_each_request_block_table():
    packed_cache = torch.tensor(
        [
            [[[3.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    ql_nope = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    q_pe = torch.zeros((2, 1, 1), dtype=torch.float32)

    result = AscendIndexerKPoolMLAImpl._sparse_attention_pytorch(
        ql_nope=ql_nope,
        q_pe=q_pe,
        packed_kv_cache=packed_cache,
        topk_indices=torch.zeros((2, 1, 1), dtype=torch.int32),
        block_table=torch.tensor(
            [
                [1],
                [0],
            ],
            dtype=torch.int32,
        ),
        actual_seq_lengths_query=torch.tensor([1, 2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([1, 1], dtype=torch.int32),
        scale=1.0,
        num_actual_tokens=2,
    )

    assert result[:, 0].tolist() == [
        [1.0, 0.0],
        [3.0, 0.0],
    ]
