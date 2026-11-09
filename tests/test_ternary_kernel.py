"""Unit tests for per-group ternary {-gamma, 0, +gamma} projection and MLX 2-bit packing."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ternary_nemo.ternary_kernel import (
    dequantize_and_verify_ternary,
    pack_ternary_to_mlx_2bit,
    project_to_ternary,
    quantize_tensor_ternary_2bit,
)


@pytest.mark.parametrize("mode", ["ternary-lloyd", "ternary-absmean"])
def test_2d_and_3d_ternary_exact_reconstruction(mode: str) -> None:
    """Verify that packed 2-bit MLX tensors dequantize to exact {-gamma, 0, +gamma} states."""
    mx.random.seed(42)
    # 3D shape mimicking SwitchMLP (num_experts=8, out_dims=32, in_dims=128)
    w = mx.random.normal(shape=(8, 32, 128)).astype(mx.bfloat16)

    t, gamma = project_to_ternary(w, group_size=64, mode=mode)
    packed_w, scales, biases = pack_ternary_to_mlx_2bit(t, gamma, group_size=64)

    # Dequantize with MLX's native Metal kernel
    dq = mx.dequantize(packed_w, scales, biases, group_size=64, bits=2, mode="affine")
    expected = (t.astype(mx.bfloat16) * mx.expand_dims(gamma, axis=-1)).reshape(w.shape)

    max_err = float(mx.max(mx.abs(dq - expected)).item())
    assert max_err == 0.0, f"Expected exact zero dequantization error, got {max_err}"

    report = dequantize_and_verify_ternary(packed_w, scales, biases, group_size=64)
    assert report["is_strict_ternary"] is True
    assert report["max_unique_values_per_group"] <= 3
    assert report["symmetry_error"] == 0.0


def test_quantize_tensor_ternary_2bit_shapes() -> None:
    """Verify packed uint32 shapes match MLX's 16-values-per-uint32 convention."""
    w = mx.random.normal(shape=(4, 16, 256)).astype(mx.bfloat16)
    w_q, scales, biases = quantize_tensor_ternary_2bit(w, group_size=64, mode="ternary-lloyd")

    # 256 / 16 = 16 uint32s along last axis; 256 / 64 = 4 groups along last axis
    assert w_q.shape == (4, 16, 16)
    assert w_q.dtype == mx.uint32
    assert scales.shape == (4, 16, 4)
    assert biases.shape == (4, 16, 4)
    assert scales.dtype == mx.bfloat16
    assert biases.dtype == mx.bfloat16
