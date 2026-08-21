import torch
import vllm

from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)

vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec

# GLM-5 owns multiple physical caches per layer index (main MLA, compressed
# indexer K, compressor state, Mamba state).  Upstream bind_kv_cache raises
# NotImplementedError for non-CUDA platforms when several layers share one
# index, so Ascend assembles the runner cache list without that restriction.
def _ascend_bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Bind the allocated KV cache to both ModelRunner and forward context.

    Same contract as upstream ``bind_kv_cache``, except that multiple layers
    sharing one layer index (hybrid GLM-5 caches) are all appended to
    ``runner_kv_caches`` instead of raising on non-CUDA platforms.
    """
    from collections import defaultdict

    from vllm.model_executor.models.utils import extract_layer_index

    assert len(runner_kv_caches) == 0

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


# gpu/attn_utils imports bind_kv_cache with `from vllm.v1.worker.utils import`,
# so both module attributes must be replaced for init_kv_cache to see it.
vllm.v1.worker.utils.bind_kv_cache = _ascend_bind_kv_cache
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = _ascend_bind_kv_cache
