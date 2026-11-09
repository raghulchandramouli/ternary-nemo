"""Per-group ternary {-gamma, 0, +gamma} projection and exact 2-bit MLX uint32 packing.

Why custom 2-bit MLX packing is needed for strict ternary weights:
------------------------------------------------------------------
MLX's native `mx.quantize(w, group_size=64, bits=2, mode="affine")` divides the
min-to-max range `[w_min, w_max]` of each group into `2^2 - 1 = 3` equal intervals:
    step = (w_max - w_min) / 3 = (-gamma - gamma) / 3 = -2/3 * gamma
which maps a symmetric ternary tensor `{-gamma, 0, +gamma}` onto `{+gamma, +1/3*gamma, -1/3*gamma, -gamma}`,
destroying the exact zero state!

Instead, MLX's 2-bit dequantization formula (`mx.dequantize` and `mx.gather_qmm`) computes:
    w_hat = scale * q + bias,   where q in {0, 1, 2, 3}
By setting:
    scale = gamma
    bias  = -gamma
    q     = t + 1  in {0, 1, 2}  (for t in {-1, 0, +1})
MLX's native Metal kernels (`mx.gather_qmm` / `QuantizedSwitchLinear` / `QuantizedLinear`)
reconstruct the exact symmetric ternary states:
    q = 0  ->  gamma * 0 - gamma = -gamma
    q = 1  ->  gamma * 1 - gamma =  0.0
    q = 2  ->  gamma * 2 - gamma = +gamma
with zero quantization error on the zero state and full hardware acceleration on Apple Silicon.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

TernaryMode = Literal["ternary-lloyd", "ternary-absmean", "mlx-affine-2bit"]


def project_to_ternary(
    weight: mx.array,
    group_size: int = 64,
    mode: Literal["ternary-lloyd", "ternary-absmean"] = "ternary-lloyd",
    lloyd_steps: int = 4,
) -> tuple[mx.array, mx.array]:
    """Project a weight tensor onto per-group ternary states {-1, 0, +1} and scales gamma.

    Args:
        weight: Float/bfloat16 weight tensor of shape `(..., K)` where `K % group_size == 0`.
        group_size: Number of weights per quantization group along the last axis.
        mode:
            - 'ternary-absmean': BitNet b1.58 rule (`gamma = mean(|w|)`, `t = clip(round(w / gamma), -1, 1)`
              followed by optimal least-squares scale refinement `gamma = sum(w * t) / sum(t^2)`).
            - 'ternary-lloyd': Alternating Lloyd-Max / coordinate descent minimizing
              `||w - gamma * t||_2^2` over `t in {-1, 0, +1}^G` and `gamma >= 0` (threshold `Delta = 0.5 * gamma`).
        lloyd_steps: Number of alternating refinement steps when `mode == 'ternary-lloyd'`.

    Returns:
        Tuple `(t, gamma)`:
            - `t`: `int8` tensor of shape `(..., n_groups, group_size)` with values strictly in `{-1, 0, +1}`.
            - `gamma`: Tensor of shape `(..., n_groups)` in `weight.dtype` (`bfloat16`/`float16`/`float32`).
    """
    orig_dtype = weight.dtype if mx.issubdtype(weight.dtype, mx.floating) else mx.bfloat16
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"Last dimension {weight.shape[-1]} must be divisible by group_size={group_size}"
        )

    n_groups = weight.shape[-1] // group_size
    grouped_shape = (*weight.shape[:-1], n_groups, group_size)
    w = weight.reshape(grouped_shape).astype(mx.float32)

    # Initial BitNet b1.58 absmean estimate: gamma_0 = mean(|w|)
    abs_w = mx.abs(w)
    gamma = mx.mean(abs_w, axis=-1, keepdims=True)
    safe_gamma = mx.maximum(gamma, 1e-8)

    if mode == "ternary-absmean":
        # BitNet round-to-nearest ternary + least-squares scale update
        t_float = mx.clip(mx.round(w / safe_gamma), -1.0, 1.0)
        denom = mx.maximum(mx.sum(t_float * t_float, axis=-1, keepdims=True), 1.0)
        gamma = mx.sum(w * t_float, axis=-1, keepdims=True) / denom
        gamma = mx.maximum(gamma, 0.0)
    elif mode == "ternary-lloyd":
        # Initialize with standard Gaussian optimal ternary threshold (~0.7 * mean(|w|))
        threshold = 0.7 * gamma
        t_float = mx.where(abs_w > threshold, mx.sign(w), 0.0)
        for _ in range(lloyd_steps):
            denom = mx.maximum(mx.sum(t_float * t_float, axis=-1, keepdims=True), 1.0)
            gamma = mx.sum(w * t_float, axis=-1, keepdims=True) / denom
            gamma = mx.maximum(gamma, 0.0)
            # For fixed gamma, nearest neighbor in {-gamma, 0, +gamma} has decision boundary at +-0.5 * gamma
            threshold = 0.5 * gamma
            t_float = mx.where(abs_w > threshold, mx.sign(w), 0.0)
    else:
        raise ValueError(f"Unsupported ternary projection mode: {mode!r}")

    t_int = t_float.astype(mx.int8)
    gamma_out = gamma.squeeze(-1).astype(orig_dtype)
    return t_int, gamma_out


def pack_ternary_to_mlx_2bit(
    t: mx.array,
    gamma: mx.array,
    group_size: int = 64,
) -> tuple[mx.array, mx.array, mx.array]:
    """Pack ternary states `t in {-1, 0, +1}` and scales `gamma` into MLX's native 2-bit uint32 format.

    MLX packs 16 2-bit unsigned codes `q in {0, 1, 2, 3}` per `uint32` in little-endian bit order:
        packed = sum_{i=0}^{15} (q_i << (2 * i))
    And dequantizes via:
        w_hat = scales * q + biases

    Setting `q = t + 1 in {0, 1, 2}`, `scales = gamma`, and `biases = -gamma` guarantees:
        q = 0 (t = -1) -> -gamma
        q = 1 (t =  0) ->  0.0
        q = 2 (t = +1) -> +gamma

    Args:
        t: Ternary tensor of shape `(..., n_groups, group_size)` with values in `{-1, 0, +1}`.
        gamma: Scale tensor of shape `(..., n_groups)` in float/bfloat16.
        group_size: Group size (must be a multiple of 16, default 64).

    Returns:
        Tuple `(packed_weight, scales, biases)` matching `mx.quantize(..., bits=2, group_size=group_size)`.
    """
    if group_size % 16 != 0:
        raise ValueError(f"group_size must be divisible by 16 for 2-bit uint32 packing, got {group_size}")

    lead_shape = t.shape[:-2]
    n_groups = t.shape[-2]
    values_per_uint32 = 16
    uint32_per_group = group_size // values_per_uint32

    # Shift t in {-1, 0, +1} -> q in {0, 1, 2}
    q = (t.astype(mx.int32) + 1).astype(mx.uint32)
    q = q.reshape(*lead_shape, n_groups * uint32_per_group, values_per_uint32)

    shifts = mx.arange(values_per_uint32, dtype=mx.uint32) * 2
    packed_weight = mx.sum(q << shifts, axis=-1).astype(mx.uint32)

    scales = gamma
    biases = -gamma
    return packed_weight, scales, biases


def quantize_tensor_ternary_2bit(
    weight: mx.array,
    group_size: int = 64,
    mode: TernaryMode = "ternary-lloyd",
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize a 2D or 3D weight tensor into MLX 2-bit format (`(weight, scales, biases)`).

    Supports both:
        - Strict ternary modes (`'ternary-lloyd'`, `'ternary-absmean'`) producing exact `{-gamma, 0, +gamma}` groups.
        - Standard MLX 4-level 2-bit affine mode (`'mlx-affine-2bit'`).
    """
    if mode == "mlx-affine-2bit":
        w_q, scales, *biases_list = mx.quantize(weight, group_size=group_size, bits=2, mode="affine")
        biases = biases_list[0]
        return w_q, scales, biases

    t, gamma = project_to_ternary(weight, group_size=group_size, mode=mode)
    w_q, scales, biases = pack_ternary_to_mlx_2bit(t, gamma, group_size=group_size)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def dequantize_and_verify_ternary(
    packed_weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    max_groups_to_check: int = 256,
) -> dict[str, float | int | bool]:
    """Dequantize a 2-bit packed tensor with `mx.dequantize` and verify strict ternary states.

    Checks that:
        1. `biases == -scales` (symmetric zero point at code `q = 1`).
        2. Every sampled 64-weight group has at most 3 unique values `{-gamma, 0, +gamma}`.
        3. Zero (`0.0`) is represented exactly.
    """
    # Slice along leading dimensions first so we don't dequantize a full 1.2GB expert tensor at once
    flat_w = packed_weight.reshape(-1, packed_weight.shape[-1])
    flat_s = scales.reshape(-1, scales.shape[-1])
    flat_b = biases.reshape(-1, biases.shape[-1])

    rows_to_take = max(1, min(flat_w.shape[0], (max_groups_to_check + flat_s.shape[-1] - 1) // flat_s.shape[-1]))
    sub_w = flat_w[:rows_to_take]
    sub_s = flat_s[:rows_to_take]
    sub_b = flat_b[:rows_to_take]

    dq = mx.dequantize(sub_w, sub_s, sub_b, group_size=group_size, bits=2, mode="affine")
    mx.eval(dq, sub_s, sub_b)

    sym_error = float(mx.max(mx.abs(sub_s.astype(mx.float32) + sub_b.astype(mx.float32))).item())

    grouped_dq = dq.astype(mx.float32).reshape(-1, group_size)[:max_groups_to_check]
    groups_list = grouped_dq.tolist()

    max_unique_in_any_group = 0
    zero_count = 0
    total_count = len(groups_list) * group_size

    for grp in groups_list:
        unique_vals = set(grp)
        if len(unique_vals) > max_unique_in_any_group:
            max_unique_in_any_group = len(unique_vals)
        zero_count += sum(1 for v in grp if v == 0.0)

    is_strict_ternary = (max_unique_in_any_group <= 3) and (sym_error == 0.0)
    return {
        "is_strict_ternary": is_strict_ternary,
        "max_unique_values_per_group": max_unique_in_any_group,
        "symmetry_error": sym_error,
        "zero_sparsity": zero_count / max(1, total_count),
        "groups_checked": len(groups_list),
    }
