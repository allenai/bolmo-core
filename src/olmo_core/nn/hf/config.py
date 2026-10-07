from typing import Optional

from transformers import Olmo2Config, PretrainedConfig

from olmo_core.data.tokenizer import ByteTokenizerConfig
from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention
from olmo_core.nn.layer_norm import QwenRMSNorm
from olmo_core.nn.moe.mlp import DroplessMoEMLP, MoEMLP
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer.block import (
    MoEReorderedNormTransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlock,
)
from olmo_core.nn.transformer.model import (
    MoETransformer,
    NormalizedTransformer,
    Transformer,
    BolmoTransformer,
)

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None

try:
    from transformers import Olmo3Config  # type: ignore
except ImportError:
    Olmo3Config = None

from olmo_core.nn.bolmo.hf import configuration_bolmo


def _get_flex_olmo_config(model: MoETransformer) -> PretrainedConfig:
    blocks = list(model.blocks.values())
    for block in blocks:
        if not isinstance(block, MoEReorderedNormTransformerBlock):
            raise NotImplementedError(
                f"Block is not a {MoEReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if not isinstance(block.experts.mlp, (DroplessMoEMLP, MoEMLP)):
            raise NotImplementedError(
                f"MoE mlp is not a {DroplessMoEMLP.__name__} or {MoEMLP.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if not isinstance(block.attention, Attention):
            raise NotImplementedError(
                f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
            )
        if block.attention.rope is None:
            raise NotImplementedError(
                f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
            )

    block = blocks[0]
    assert isinstance(block, MoEReorderedNormTransformerBlock)
    assert isinstance(block.attention, Attention)
    assert block.attention.rope is not None

    if FlexOlmoConfig is None:
        raise RuntimeError("The installed transformers version does not support FlexOlmo")

    return FlexOlmoConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=block.feed_forward_moe.experts.mlp.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=block.attention.n_heads,
        num_key_value_heads=block.attention.n_kv_heads,
        hidden_act="silu",
        max_position_embeddings=-1,
        attention_bias=block.attention.w_out.bias is not None,
        rope_theta=block.attention.rope.theta,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        rms_norm_eps=block.feed_forward_norm.eps,
        num_experts_per_tok=block.feed_forward_moe.router.top_k,
        num_experts=block.feed_forward_moe.router.num_experts,
        tie_word_embeddings=False,
    )


def _get_bolmo_block_type(model: BolmoTransformer) -> str:
    """
    Map the OLMo Core global block class onto the corresponding Bolmo ``block_type``.

    OLMo 2/3 use reordered norm (normalize the attention/MLP outputs); Llama 3 and Qwen 3 use
    the plain pre-norm block.
    """
    block_types = {type(block) for block in model.blocks.values()}
    if len(block_types) > 1:
        raise NotImplementedError(
            f"All global blocks must be the same type to build a Bolmo HF config, got {block_types}"
        )

    block_type = block_types.pop()
    if issubclass(block_type, ReorderedNormTransformerBlock):
        return "reordered_norm"
    if issubclass(block_type, TransformerBlock):
        return "default"
    raise NotImplementedError(
        f"Unable to build a Bolmo HF config for global block type {block_type.__name__}"
    )


def _get_bolmo_norm_type(model: BolmoTransformer) -> str:
    """
    Map the OLMo Core layer norm classes onto the corresponding Bolmo ``norm_type``.

    The Bolmo HF model uses a single norm implementation throughout, so all norms in the
    checkpoint have to agree.
    """
    # NOTE: `local_encoder.post_last_block_norm` and `local_decoder.initial_norm` are deliberately
    # excluded: OLMo Core hardcodes those to `torch.nn.RMSNorm(eps=1e-5)` for every architecture,
    # so they carry no information about the original model's norm variant.
    norms = [block.feed_forward_norm for block in model.blocks.values()]
    norms += [block.attention_norm for block in model.blocks.values()]
    norms += [
        norm
        for blocks in (model.local_encoder.blocks, model.local_decoder.blocks)
        for block in blocks.values()
        for norm in (block.xlstm_norm, block.feed_forward_norm)
    ]
    if model.lm_head.norm is not None:
        norms.append(model.lm_head.norm)

    is_qwen = {isinstance(norm, QwenRMSNorm) for norm in norms}
    if len(is_qwen) > 1:
        raise NotImplementedError(
            "Bolmo HF export requires a single layer norm implementation throughout the model, "
            "but the checkpoint mixes QwenRMSNorm with other norms."
        )
    return "qwen_rms" if is_qwen.pop() else "rms"


def get_bolmo_tokenizer_config(
    tokenizer_config: ByteTokenizerConfig,
) -> configuration_bolmo.BolmoTokenizerConfig:
    """Translate the training-time byte tokenizer config into its HF counterpart."""
    if tokenizer_config.bpe_token_end_id is None:
        raise NotImplementedError(
            "Bolmo HF export requires a byte tokenizer config with `bpe_token_end_id` set"
        )
    if tokenizer_config.original_identifier is None:
        raise NotImplementedError(
            "Bolmo HF export requires a byte tokenizer config with `original_identifier` set; "
            "without it the exported model cannot reproduce the subword expansion used in training"
        )
    return configuration_bolmo.BolmoTokenizerConfig(
        vocab_size=tokenizer_config.vocab_size,
        bos_token_id=tokenizer_config.bos_token_id,
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        bpe_token_end_id=tokenizer_config.bpe_token_end_id,
        special_tokens=list(tokenizer_config.special_tokens),
        special_tokens_first=tokenizer_config.special_tokens_first,
        original_identifier=tokenizer_config.original_identifier,
    )


def _get_bolmo_config(
    model: BolmoTransformer,
    tokenizer_config: Optional[ByteTokenizerConfig] = None,
) -> configuration_bolmo.BolmoConfig:
    subword_vocab_size: int = model.local_encoder.expanded_embeddings.weight.shape[0] if model.local_encoder.add_expanded_embeddings else 0  # type: ignore
    first_global_block = model.blocks["0"]
    first_local_block = model.local_encoder.blocks["0"]
    attention = first_global_block.attention

    sliding_window_blocks = [
        block for block in model.blocks.values() if block.attention.backend.window_size != (-1, -1)
    ]

    if len(sliding_window_blocks) > 0:
        # assume olmo 3 - 3 out of 4 are sliding window. this is the default in the Bolmo config
        layer_types = None
        # `window_size` is the flash-attention form, which excludes the current position; HF
        # expects a value one larger. See https://github.com/huggingface/transformers/pull/40163
        window_sizes = {block.attention.backend.window_size[0] for block in sliding_window_blocks}
        if len(window_sizes) > 1:
            raise NotImplementedError(
                f"All sliding window layers must share a window size, found {window_sizes}"
            )
        sliding_window = window_sizes.pop() + 1
    else:
        # olmo 2 / llama 3 / qwen 3 - all full attention
        layer_types = ["full_attention"] * model.n_layers
        sliding_window = None

    # RoPE scaling is validated across the full attention layers regardless of whether any layer
    # uses sliding window attention, so that scaling isn't silently dropped for non-OLMo3 models.
    rope_scaling = _get_and_validate_rope_scaling_config(list(model.blocks.values()))

    return configuration_bolmo.BolmoConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=first_global_block.feed_forward.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=attention.n_heads,
        num_key_value_heads=attention.n_kv_heads,
        head_dim=attention.head_dim,
        attention_bias=attention.w_out.bias is not None,
        max_position_embeddings=-1,
        rms_norm_eps=first_global_block.feed_forward_norm.eps,
        local_rms_norm_eps=model.local_encoder.post_last_block_norm.eps,
        rope_scaling=rope_scaling,
        rope_theta=attention.rope.theta,
        layer_types=layer_types,
        sliding_window=sliding_window,
        block_type=_get_bolmo_block_type(model),
        norm_type=_get_bolmo_norm_type(model),
        use_qk_norm=attention.q_norm is not None,
        use_head_qk_norm=bool(attention.use_head_qk_norm),
        add_expanded_embeddings=model.local_encoder.add_expanded_embeddings,
        boundary_predictor_lookahead=model.local_encoder.boundary_predictor_module.boundary_predictor_lookahead,
        boundary_threshold="sample:0",
        num_local_encoder_layers=model.local_encoder.n_layers,
        num_local_decoder_layers=model.local_decoder.n_layers,
        num_local_heads=first_local_block.xlstm.config.num_heads,
        local_intermediate_size=first_local_block.feed_forward.hidden_size,
        subword_vocab_size=subword_vocab_size,
        tokenizer_config=(
            None if tokenizer_config is None else get_bolmo_tokenizer_config(tokenizer_config)
        ),
    )


@beta_feature
def get_hf_config(
    model: Transformer,
    *,
    tokenizer_config: Optional[ByteTokenizerConfig] = None,
) -> PretrainedConfig:
    """
    :param tokenizer_config: Only used for :class:`BolmoTransformer`, whose HF config embeds the
        byte tokenizer definition (including the subword tokenizer the byte ids expand back into).
    """
    if isinstance(model, NormalizedTransformer):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )

    if isinstance(model, MoETransformer):
        return _get_flex_olmo_config(model)

    if isinstance(model, BolmoTransformer):
        return _get_bolmo_config(model, tokenizer_config=tokenizer_config)

    blocks = list(model.blocks.values())
    first_block = blocks[0]
    if not isinstance(first_block, ReorderedNormTransformerBlock):
        raise NotImplementedError(
            f"Block is not a {ReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
        )

    if not isinstance(first_block.attention, Attention):
        raise NotImplementedError(
            f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
        )
    if first_block.attention.rope is None:
        raise NotImplementedError(
            f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
        )

    if first_block.attention.backend is None:
        raise ValueError("Attention backend is not set.")

    rope_scaling = _get_and_validate_rope_scaling_config(blocks)

    # Extract common configuration parameters
    common_config_args = {
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": first_block.feed_forward.hidden_size,
        "num_hidden_layers": model.n_layers,
        "num_attention_heads": first_block.attention.n_heads,
        "num_key_value_heads": first_block.attention.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": -1,
        "attention_bias": first_block.attention.w_out.bias is not None,
        "rope_theta": first_block.attention.rope.theta,
        "rope_scaling": rope_scaling,
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "rms_norm_eps": first_block.feed_forward_norm.eps,
        "tie_word_embeddings": False,
    }

    # The OLMo 3 model family is identical to the OLMo 2 model family, except:
    # - Sliding window attention is used for 3 out of 4 layers.
    # - RoPE scaling is not applied to sliding window attention layers.
    # Therefore, if any layer uses sliding window attention, we assume the model is OLMo 3.
    # Identify layers that use sliding window attention.
    sliding_window_blocks = [
        block for block in blocks if block.attention.backend.window_size != (-1, -1)
    ]

    if sliding_window_blocks:
        if Olmo3Config is None:
            raise RuntimeError("The installed transformers version does not support Olmo3")

        found_window_sizes = {
            block.attention.backend.window_size[0] for block in sliding_window_blocks
        }

        if len(found_window_sizes) > 1:
            raise ValueError(
                "All sliding window attention layers must have the same window size for "
                f"OLMo3Config. Found different window sizes: {found_window_sizes}."
            )

        # This sliding window sizes value is configured to be fed to flash_attention -
        # it is one smaller than the actual window size because FA implicitly includes the
        # current position in the window. HF expects a value one larger than this and will
        # manually adjust the window size down by 1 for FA.
        # See https://github.com/huggingface/transformers/pull/40163
        common_window_size_value = found_window_sizes.pop()

        olmo3_specific_args = {
            "sliding_window": common_window_size_value + 1,
            "layer_types": [
                "sliding_attention"
                if block.attention.backend.window_size != (-1, -1)
                else "full_attention"
                for block in blocks
            ],
        }
        return Olmo3Config(**common_config_args, **olmo3_specific_args)
    else:
        return Olmo2Config(**common_config_args)


def _get_and_validate_rope_scaling_config(blocks) -> dict | None:
    """
    Validate RoPE scaling configuration across transformer blocks.

    :param blocks: The list of transformer blocks to validate.
    :returns: The validated RoPE scaling config dict for HF, or None if no scaling.
    :raises NotImplementedError: If RoPE scaling is applied to sliding window layers or if
                               full attention layers have different RoPE scaling configs.
    """
    # Separate full attention layers from sliding window layers
    full_attention_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if block.attention.backend.window_size == (-1, -1)
    ]
    sliding_window_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if block.attention.backend.window_size != (-1, -1)
    ]

    # Check for RoPE scaling on sliding window layers (not allowed)
    sliding_with_scaling = [
        (idx, block)
        for idx, block in sliding_window_layers
        if block.attention.rope.scaling is not None
    ]
    if sliding_with_scaling:
        sliding_indices = [idx for idx, _ in sliding_with_scaling]
        raise NotImplementedError(
            f"RoPE scaling is configured on sliding window attention layers {sliding_indices}, "
            f"but HuggingFace only supports RoPE scaling on full attention layers. "
            f"Please remove RoPE scaling from sliding window layers or convert them to full attention."
        )

    # Collect RoPE scaling configs from full attention layers only
    full_layers_with_scaling = [
        (idx, block)
        for idx, block in full_attention_layers
        if block.attention.rope.scaling is not None
    ]
    if not full_layers_with_scaling:
        return None

    rope_scaling_configs: list[RoPEScalingConfig] = [
        block.attention.rope.scaling for _, block in full_layers_with_scaling
    ]

    # Validate that all full attention layers with RoPE scaling use the same configuration
    first_config = rope_scaling_configs[0]
    first_config_dict = first_config.to_hf_config()

    for i, rope_config in enumerate(rope_scaling_configs[1:], 1):
        config_dict = rope_config.to_hf_config()
        if config_dict != first_config_dict:
            scaling_indices = [idx for idx, _ in full_layers_with_scaling]
            raise NotImplementedError(
                f"Full attention layers have different RoPE scaling configurations but HuggingFace "
                "only supports a single RoPE scaling configuration per model. "
                f"Full attention layers with scaling: {scaling_indices}. "
                f"First config: {first_config_dict}, Different config at layer {i}: {config_dict}"
            )

    return first_config_dict
