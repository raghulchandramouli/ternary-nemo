"""End-to-end synthetic NemotronH model streaming quantization & mlx_lm.utils.load_model test."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.nemotron_h import Model, ModelArgs
from mlx_lm.utils import load_model

from ternary_nemo.config import classify_tensor_key
from ternary_nemo.streamer import stream_quantize_model
from ternary_nemo.verifier import inspect_checkpoint_ternary_weights


def test_classify_tensor_key_rules() -> None:
    """Verify mixed-precision classification rules on real Nemotron-3.5-Lightning tensor names."""
    assert (
        classify_tensor_key("backbone.layers.1.mixer.switch_mlp.fc1.weight").action
        == "ternary_2bit"
    )
    assert (
        classify_tensor_key("backbone.layers.1.mixer.switch_mlp.fc2.weight").action
        == "ternary_2bit"
    )
    assert (
        classify_tensor_key("backbone.layers.1.mixer.shared_experts.up_proj.weight").action
        == "affine_4bit"
    )
    assert (
        classify_tensor_key("backbone.layers.0.mixer.in_proj.weight").action
        == "affine_4bit"
    )
    assert (
        classify_tensor_key("backbone.layers.5.mixer.q_proj.weight").action
        == "affine_4bit"
    )
    assert classify_tensor_key("backbone.embeddings.weight").action == "affine_4bit"
    assert classify_tensor_key("lm_head.weight").action == "affine_4bit"

    # Sensitive router / SSM / norm weights must remain in bf16/fp32
    assert classify_tensor_key("backbone.layers.1.mixer.gate.weight").action == "keep_dtype"
    assert (
        classify_tensor_key("backbone.layers.1.mixer.gate.e_score_correction_bias").action
        == "keep_dtype"
    )
    assert classify_tensor_key("backbone.layers.0.mixer.A_log").action == "keep_dtype"
    assert classify_tensor_key("backbone.layers.0.mixer.conv1d.weight").action == "keep_dtype"
    assert classify_tensor_key("backbone.layers.0.norm.weight").action == "keep_dtype"


def test_synthetic_nemotron_h_roundtrip(tmp_path: Path) -> None:
    """Create a miniature NemotronH (Mamba2 + Attention + MoE) model, stream-quantize it, and load with mlx_lm."""
    raw_dir = tmp_path / "raw_bf16"
    quant_dir = tmp_path / "quant_ternary"
    raw_dir.mkdir(parents=True)

    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "attention_bias": False,
        "mamba_num_heads": 4,
        "mamba_head_dim": 32,
        "mamba_proj_bias": False,
        "ssm_state_size": 64,
        "conv_kernel": 4,
        "n_groups": 2,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "layers_block_type": ["mamba", "moe", "attention"],
        "moe_intermediate_size": 128,
        "moe_shared_expert_intermediate_size": 128,
        "moe_latent_size": None,
        "n_group": 1,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "topk_group": 1,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
    }

    with open(raw_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    args = ModelArgs.from_dict(config)
    model = Model(args)

    # Flatten model parameters into a safetensors shard
    from mlx.utils import tree_flatten

    flat_weights: dict[str, mx.array] = {}
    for k, v in tree_flatten(model.parameters()):
        if "A_log" in k or "e_score_correction_bias" in k:
            flat_weights[k] = v.astype(mx.float32)
        else:
            flat_weights[k] = (mx.random.normal(v.shape) * 0.05).astype(mx.bfloat16)

    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(raw_dir / shard_name), flat_weights)

    index = {
        "metadata": {"total_size": sum(v.nbytes for v in flat_weights.values()), "total_parameters": sum(v.size for v in flat_weights.values())},
        "weight_map": {k: shard_name for k in flat_weights},
    }
    with open(raw_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    # Stream-quantize the synthetic checkpoint
    summary = stream_quantize_model(
        repo_id="synthetic/nemotron-test",
        output_dir=quant_dir,
        mode="ternary-lloyd",
        group_size=64,
        local_source_dir=raw_dir,
    )
    assert summary["layer_counts"]["ternary_2bit"] == 2  # fc1 and fc2 in layer 1

    # Verify strict ternary weights on disk
    ver = inspect_checkpoint_ternary_weights(quant_dir)
    assert ver["all_strict_ternary"] is True

    # Load quantized model via mlx_lm's official loader and run a forward pass
    q_model, _ = load_model(quant_dir)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits = q_model(tokens)
    mx.eval(logits)
    assert logits.shape == (1, 4, 256)
