"""Mixed-precision layer classification and MLX config.json builder for NemotronH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

QuantAction = Literal["ternary_2bit", "affine_4bit", "keep_dtype"]


@dataclass(frozen=True)
class LayerQuantSpec:
    """Quantization specification for a single model tensor or module path."""

    module_path: str
    action: QuantAction
    bits: int | None = None
    group_size: int | None = None
    mode: str = "affine"


# Tensors that must remain in their original floating-point precision (bf16/fp32)
# to preserve MoE routing stability, Mamba-2 SSM state dynamics, and norms.
KEEP_PRECISION_PATTERNS: tuple[str, ...] = (
    ".mixer.gate.",               # MoE router weights & e_score_correction_bias
    ".mixer.conv1d.",             # Mamba-2 depthwise 1D convolution
    ".mixer.A_log",               # Mamba-2 SSM decay log-parameter (fp32)
    ".mixer.D",                   # Mamba-2 SSM skip connection
    ".mixer.dt_bias",             # Mamba-2 step-size bias
    ".norm.weight",               # Layer RMSNorm & MambaRMSNormGated
    "backbone.norm_f.weight",     # Final backbone RMSNorm
)

# Routed MoE expert tensors (93.0% of total parameters: 29.37B / 31.58B)
# Targeted for strict {-gamma, 0, +gamma} ternary quantization packed in 2-bit MLX.
TERNARY_EXPERT_PATTERNS: tuple[str, ...] = (
    ".mixer.switch_mlp.fc1.weight",
    ".mixer.switch_mlp.fc2.weight",
)

# Sensitive dense projections & token embeddings (~2.13B params total)
# Kept at 4-bit affine quantization for high accuracy with minimal memory (~1.1 GB).
FOUR_BIT_PATTERNS: tuple[str, ...] = (
    ".mixer.shared_experts.up_proj.weight",
    ".mixer.shared_experts.down_proj.weight",
    ".mixer.in_proj.weight",
    ".mixer.out_proj.weight",
    ".mixer.q_proj.weight",
    ".mixer.k_proj.weight",
    ".mixer.v_proj.weight",
    ".mixer.o_proj.weight",
    ".mixer.fc1_latent_proj.weight",
    ".mixer.fc2_latent_proj.weight",
    "backbone.embeddings.weight",
    "lm_head.weight",
)


def classify_tensor_key(
    tensor_key: str,
    group_size: int = 64,
    expert_bits: int = 2,
    dense_bits: int = 4,
) -> LayerQuantSpec:
    """Classify a safetensors weight key into its mixed-precision quantization rule.

    Args:
        tensor_key: Full parameter name (e.g. 'backbone.layers.1.mixer.switch_mlp.fc1.weight').
        group_size: Quantization group size along the last dimension (default 64).
        expert_bits: Bit-width for routed MoE experts (default 2 for ternary-in-2bit).
        dense_bits: Bit-width for shared experts, Mamba/Attention projections, and embeddings.

    Returns:
        LayerQuantSpec describing whether to ternarize (2-bit), 4-bit quantize, or keep in bf16/fp32.
    """
    module_path = (
        tensor_key[: -len(".weight")]
        if tensor_key.endswith(".weight")
        else tensor_key
    )

    for pattern in KEEP_PRECISION_PATTERNS:
        if pattern in tensor_key or tensor_key == pattern:
            return LayerQuantSpec(module_path=module_path, action="keep_dtype")

    for pattern in TERNARY_EXPERT_PATTERNS:
        if tensor_key.endswith(pattern):
            return LayerQuantSpec(
                module_path=module_path,
                action="ternary_2bit",
                bits=expert_bits,
                group_size=group_size,
                mode="affine",
            )

    for pattern in FOUR_BIT_PATTERNS:
        if tensor_key.endswith(pattern) or tensor_key == pattern:
            return LayerQuantSpec(
                module_path=module_path,
                action="affine_4bit",
                bits=dense_bits,
                group_size=group_size,
                mode="affine",
            )

    # Fallback: any non-weight tensor (e.g. biases) stays in original precision
    return LayerQuantSpec(module_path=module_path, action="keep_dtype")


def build_mlx_quantized_config(
    base_config: dict[str, Any],
    quantized_specs: list[LayerQuantSpec],
    default_group_size: int = 64,
    default_bits: int = 4,
) -> dict[str, Any]:
    """Build an updated MLX config.json with explicit mixed-precision quantization predicates.

    mlx-lm's `load_model()` inspects `config["quantization"]`:
      - Top-level `group_size`, `bits`, and `mode` define the default.
      - Per-module entries `config["quantization"][module_path]` override `bits` & `group_size`
        or disable quantization (`False`) for unquantized Linear/Embedding modules.
    """
    updated = dict(base_config)
    quant_dict: dict[str, Any] = {
        "group_size": default_group_size,
        "bits": default_bits,
        "mode": "affine",
    }

    for spec in sorted(quantized_specs, key=lambda s: s.module_path):
        if spec.action == "ternary_2bit":
            quant_dict[spec.module_path] = {
                "group_size": spec.group_size or default_group_size,
                "bits": spec.bits or 2,
                "mode": spec.mode,
            }
        elif spec.action == "affine_4bit":
            quant_dict[spec.module_path] = {
                "group_size": spec.group_size or default_group_size,
                "bits": spec.bits or default_bits,
                "mode": spec.mode,
            }

    updated["quantization"] = quant_dict
    updated["quantization_config"] = dict(quant_dict)
    return updated
