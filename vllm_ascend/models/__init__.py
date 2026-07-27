from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4.model:AscendDeepseekV4ForCausalLM"
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
    )
    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4.mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ModelRegistry.register_model("Qwen3DSparkModel", "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM")
    ModelRegistry.register_model(
        "DFlash2DraftModel",
        "vllm_ascend.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
    )
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_ascend.models.deepseek_mtp:AscendDeepSeekMTP")
    ModelRegistry.register_model("GlmMoeDsaForCausalLM", "vllm_ascend.models.deepseek_mtp:AscendGlmMoeDsaForCausalLM")
    ModelRegistry.register_model(
        "Eagle3LlamaForCausalLM", "vllm_ascend.models.llama_eagle3:AscendEagle3LlamaForCausalLM"
    )
    ModelRegistry.register_model(
        "Glm5NextForCausalLM", "vllm_ascend.models.glm5_next:AscendGlm5NextForCausalLM"
    )
    # 当前 Ascend 适配仅包含 GLM-5 Next 的语言模型。原始多模态权重
    # 使用 ConditionalGeneration architecture；将它显式映射到文本实现，
    # 保留权重 config.json 不变，同时避免 vLLM 尝试加载尚未适配的视觉处理器。
    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration",
        "vllm_ascend.models.glm5_next:AscendGlm5NextForCausalLM",
    )
    ModelRegistry.register_model(
        "Glm5NextMTPModel",
        "vllm_ascend.models.glm5_next_mtp:AscendGlm5NextMTP",
    )
    ModelRegistry.register_model(
        "Glm5NextMTP",
        "vllm_ascend.models.glm5_next_mtp:AscendGlm5NextMTP",
    )
