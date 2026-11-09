"""Command-line interfaces for quantizing, verifying, and publishing ternary Nemotron models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ternary_nemo.hub import generate_model_card, publish_to_hub
from ternary_nemo.streamer import stream_quantize_model
from ternary_nemo.verifier import (
    benchmark_mlx_inference,
    inspect_checkpoint_ternary_weights,
)


def main_quantize(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stream-quantize NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 into Ternary 2-bit MLX."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16",
        help="Source Hugging Face repository ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit"),
        help="Destination folder for the quantized MLX checkpoint.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ternary-lloyd", "ternary-absmean", "mlx-affine-2bit"],
        default="ternary-lloyd",
        help="2-bit quantization algorithm for routed MoE experts.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64).",
    )
    parser.add_argument(
        "--dense-bits",
        type=int,
        default=4,
        help="Bit-width for non-routed linear layers and embeddings (default: 4).",
    )
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=4.5,
        help="Maximum size in GB per output safetensors shard (default: 4.5).",
    )
    parser.add_argument(
        "--low-disk",
        action="store_true",
        help="Delete each downloaded bf16 source shard from cache right after quantizing it.",
    )
    parser.add_argument(
        "--local-source-dir",
        type=Path,
        default=None,
        help="Optional local directory containing source safetensors shards.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Optional cap on number of source shards to process (for smoke testing).",
    )
    args = parser.parse_args(argv)

    summary = stream_quantize_model(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        mode=args.mode,
        group_size=args.group_size,
        dense_bits=args.dense_bits,
        max_output_shard_gb=args.max_shard_gb,
        low_disk=args.low_disk,
        local_source_dir=args.local_source_dir,
        limit_shards=args.limit_shards,
    )
    print(json.dumps(summary, indent=2))


def main_verify(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify strict ternary states and benchmark MLX inference speed."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit"),
        help="Path to the quantized MLX model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why Mixture-of-Experts models are well suited for ternary quantization.",
        help="Prompt to run through mlx_lm.generate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate during benchmark.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only verify ternary weight states without running full model generation.",
    )
    args = parser.parse_args(argv)

    print("[ternary-nemo] Inspecting 2-bit expert weights for strict {-gamma, 0, +gamma} states...")
    weight_report = inspect_checkpoint_ternary_weights(args.model_dir)
    print(json.dumps(weight_report, indent=2))
    if not weight_report["all_strict_ternary"]:
        raise AssertionError("Ternary verification failed: found non-ternary groups!")

    if not args.skip_generate:
        print("\n[ternary-nemo] Loading model with mlx_lm and running generation benchmark...")
        bench_report = benchmark_mlx_inference(
            model_dir=args.model_dir,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        print(json.dumps(bench_report, indent=2))


def main_publish(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hugging Face Model Card and upload checkpoint to Hugging Face Hub."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit"),
        help="Local quantized checkpoint directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Destination Hugging Face repo ID (e.g. 'username/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit').",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face repository as private.",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Only render README.md without uploading to Hugging Face Hub.",
    )
    args = parser.parse_args(argv)

    card_path = generate_model_card(model_dir=args.model_dir, repo_id=args.repo_id)
    print(f"[ternary-nemo] Generated Hugging Face Model Card at: {card_path}")

    if not args.card_only:
        url = publish_to_hub(
            model_dir=args.model_dir,
            repo_id=args.repo_id,
            private=args.private,
        )
        print(f"[ternary-nemo] Published checkpoint to: {url}")
