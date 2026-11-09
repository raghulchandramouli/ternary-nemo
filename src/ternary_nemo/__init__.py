"""Streaming ternary quantization for NVIDIA-Nemotron-3.5-Lightning-30B-A3B on MLX."""

from ternary_nemo.config import (
    LayerQuantSpec,
    build_mlx_quantized_config,
    classify_tensor_key,
)
from ternary_nemo.ternary_kernel import (
    dequantize_and_verify_ternary,
    pack_ternary_to_mlx_2bit,
    project_to_ternary,
    quantize_tensor_ternary_2bit,
)

__version__ = "0.1.0"

__all__ = [
    "LayerQuantSpec",
    "build_mlx_quantized_config",
    "classify_tensor_key",
    "dequantize_and_verify_ternary",
    "pack_ternary_to_mlx_2bit",
    "project_to_ternary",
    "quantize_tensor_ternary_2bit",
]
