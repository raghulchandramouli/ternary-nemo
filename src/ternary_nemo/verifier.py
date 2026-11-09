"""Verification and throughput benchmark for quantized ternary MLX checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load

from ternary_nemo.ternary_kernel import dequantize_and_verify_ternary


def inspect_checkpoint_ternary_weights(
    model_dir: str | Path,
    max_expert_tensors: int = 4,
    max_groups_per_tensor: int = 256,
) -> dict[str, Any]:
    """Inspect converted `switch_mlp` tensors on disk and prove strict `{-gamma, 0, +gamma}` states."""
    path = Path(model_dir)
    index_path = path / "model.safetensors.index.json"
    config_path = path / "config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    group_size = config.get("quantization", {}).get("group_size", 64)
    weight_map: dict[str, str] = index["weight_map"]

    expert_weight_keys = [
        k for k in sorted(weight_map.keys()) if "switch_mlp" in k and k.endswith(".weight")
    ][:max_expert_tensors]

    results: dict[str, Any] = {}
    all_strict = True

    for w_key in expert_weight_keys:
        prefix = w_key[: -len(".weight")]
        s_key = f"{prefix}.scales"
        b_key = f"{prefix}.biases"
        shard_file = path / weight_map[w_key]
        shard_tensors = mx.load(str(shard_file))

        stats = dequantize_and_verify_ternary(
            packed_weight=shard_tensors[w_key],
            scales=shard_tensors[s_key],
            biases=shard_tensors[b_key],
            group_size=group_size,
            max_groups_to_check=max_groups_per_tensor,
        )
        results[prefix] = stats
        if not stats["is_strict_ternary"]:
            all_strict = False
        del shard_tensors

    return {
        "all_strict_ternary": all_strict,
        "group_size": group_size,
        "inspected_experts": results,
    }


def benchmark_mlx_inference(
    model_dir: str | Path,
    prompt: str = "Explain why Mixture-of-Experts architectures work well with ternary quantization.",
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Load the quantized checkpoint with `mlx_lm.load` and benchmark memory & generation speed."""
    path = Path(model_dir)
    t0 = time.time()
    model, tokenizer = load(str(path))
    load_time_s = time.time() - t0

    active_mem_gb = mx.get_active_memory() / 1e9 if hasattr(mx, "get_active_memory") else 0.0

    formatted_prompt = prompt
    if getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    gen_start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=True,
    )
    gen_elapsed_s = time.time() - gen_start

    peak_mem_gb = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else active_mem_gb

    return {
        "model_dir": str(path),
        "load_time_seconds": round(load_time_s, 2),
        "active_memory_gb": round(active_mem_gb, 2),
        "peak_memory_gb": round(peak_mem_gb, 2),
        "generation_elapsed_seconds": round(gen_elapsed_s, 2),
        "prompt": prompt,
        "response": response,
    }
