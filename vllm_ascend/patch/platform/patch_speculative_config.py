from typing import TYPE_CHECKING, Any, Literal, get_args

import vllm.config.speculative as speculative_config
from vllm.config.speculative import SpeculativeConfig
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import vllm.model_executor.layers.quantization as me_quant
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

_orig_post_init = SpeculativeConfig.__post_init__


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


_orig_hf_config_override = getattr(SpeculativeConfig, "hf_config_override", None)


def _glm5_hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    """Extend the upstream MTP config remapping with GLM-5 Next.

    Upstream ``SpeculativeConfig.hf_config_override`` already remaps the
    supported MTP target models; GLM-5-Next is not part of upstream vLLM, so
    chain the extra mapping on top instead of replacing the hook.
    """
    if _orig_hf_config_override is not None:
        hf_config = _orig_hf_config_override(hf_config)
    if hf_config.model_type in ("glm5_next", "glm5_next_text"):
        hf_config.model_type = "glm5_next_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Glm5NextMTPModel"],
            }
        )
    return hf_config


if "glm5_next_mtp" not in get_args(speculative_config.MTPModelTypes):
    speculative_config.MTPModelTypes = Literal[
        *get_args(speculative_config.MTPModelTypes),
        "glm5_next_mtp",
    ]

SpeculativeConfig.__post_init__ = _dspark_post_init
SpeculativeConfig.hf_config_override = staticmethod(_glm5_hf_config_override)
