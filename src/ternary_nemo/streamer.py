"""Memory-bounded shard-by-shard streaming quantizer for Nemotron-3.5-Lightning-30B-A3B-bf16."""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from ternary_nemo.config import (
    LayerQuantSpec,
    build_mlx_quantized_config,
    classify_tensor_key,
)
from ternary_nemo.ternary_kernel import TernaryMode, quantize_tensor_ternary_2bit

METADATA_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _clear_mlx_memory() -> None:
    """Force Python GC and release cached MLX Metal buffers."""
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _get_active_ram_gb() -> float:
    """Return active MLX Metal memory in GB."""
    if hasattr(mx, "get_active_memory"):
        return mx.get_active_memory() / 1e9
    return 0.0


def _download_metadata_files(
    repo_id: str,
    output_dir: Path,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Download lightweight config, index, and tokenizer files from Hugging Face."""
    downloaded: dict[str, Path] = {}
    for filename in METADATA_FILES:
        try:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            )
            downloaded[filename] = local_path
            if filename not in ("config.json", "model.safetensors.index.json"):
                shutil.copy2(local_path, output_dir / filename)
        except Exception:
            # Optional files like generation_config.json or special_tokens_map.json may not exist
            continue
    return downloaded


def stream_quantize_model(
    repo_id: str = "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16",
    output_dir: str | Path = "artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit",
    mode: TernaryMode = "ternary-lloyd",
    group_size: int = 64,
    expert_bits: int = 2,
    dense_bits: int = 4,
    max_output_shard_gb: float = 4.5,
    low_disk: bool = False,
    local_source_dir: str | Path | None = None,
    revision: str | None = None,
    limit_shards: int | None = None,
) -> dict[str, Any]:
    """Stream-quantize a multi-shard MLX/HF safetensors repository one shard at a time.

    Guarantees peak RAM stays under ~10 GB on a 48 GB Mac by loading, quantizing,
    and flushing one source safetensors shard at a time.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(local_source_dir) if local_source_dir else None

    start_time = time.time()

    if source_path is not None:
        config_file = source_path / "config.json"
        index_file = source_path / "model.safetensors.index.json"
        for fname in METADATA_FILES:
            src_f = source_path / fname
            if src_f.exists() and fname not in ("config.json", "model.safetensors.index.json"):
                shutil.copy2(src_f, out_path / fname)
    else:
        meta = _download_metadata_files(repo_id=repo_id, output_dir=out_path, revision=revision)
        if "config.json" not in meta or "model.safetensors.index.json" not in meta:
            raise FileNotFoundError(
                f"Could not find config.json or model.safetensors.index.json in {repo_id}"
            )
        config_file = meta["config.json"]
        index_file = meta["model.safetensors.index.json"]

    with open(config_file, "r", encoding="utf-8") as f:
        base_config = json.load(f)
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map: dict[str, str] = index_data["weight_map"]
    unique_shards = sorted(set(weight_map.values()))
    if limit_shards is not None:
        unique_shards = unique_shards[:limit_shards]

    max_shard_bytes = int(max_output_shard_gb * (1024**3))
    current_shard_tensors: dict[str, mx.array] = {}
    current_shard_bytes = 0
    written_temp_shards: list[tuple[Path, list[str]]] = []
    total_output_bytes = 0
    peak_ram_gb = 0.0

    quantized_specs: list[LayerQuantSpec] = []
    counts = {"ternary_2bit": 0, "affine_4bit": 0, "keep_dtype": 0}

    def flush_output_shard() -> None:
        nonlocal current_shard_tensors, current_shard_bytes
        if not current_shard_tensors:
            return
        shard_idx = len(written_temp_shards) + 1
        temp_name = out_path / f"_temp_shard_{shard_idx:05d}.safetensors"
        mx.save_safetensors(str(temp_name), current_shard_tensors, metadata={"format": "mlx"})
        written_temp_shards.append((temp_name, list(current_shard_tensors.keys())))
        current_shard_tensors = {}
        current_shard_bytes = 0
        _clear_mlx_memory()

    print(
        f"[ternary-nemo] Streaming {len(unique_shards)} shards from {repo_id} "
        f"(mode={mode}, group_size={group_size})"
    )

    for shard_idx, shard_filename in enumerate(unique_shards, start=1):
        print(f"\n[{shard_idx}/{len(unique_shards)}] Fetching & processing {shard_filename}...")
        if source_path is not None:
            shard_local_path = source_path / shard_filename
        else:
            shard_local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=shard_filename,
                    revision=revision,
                )
            )

        raw_tensors: dict[str, mx.array] = mx.load(str(shard_local_path))

        pbar = tqdm(
            sorted(raw_tensors.keys()),
            desc=f"Quantizing {shard_filename}",
            leave=True,
        )
        for key in pbar:
            if key.startswith("mtp."):
                # NemotronH sanitize drops unused MTP weights if present
                continue

            tensor = raw_tensors[key]
            spec = classify_tensor_key(
                key,
                group_size=group_size,
                expert_bits=expert_bits,
                dense_bits=dense_bits,
            )
            quantized_specs.append(spec)
            counts[spec.action] += 1

            out_tensors_for_key: dict[str, mx.array] = {}
            if spec.action == "ternary_2bit":
                pbar.set_postfix_str(f"2-bit ternary: {spec.module_path.split('.')[-2]}")
                w_q, scales, biases = quantize_tensor_ternary_2bit(
                    tensor, group_size=group_size, mode=mode
                )
                out_tensors_for_key[f"{spec.module_path}.weight"] = w_q
                out_tensors_for_key[f"{spec.module_path}.scales"] = scales
                out_tensors_for_key[f"{spec.module_path}.biases"] = biases
            elif spec.action == "affine_4bit":
                pbar.set_postfix_str(f"4-bit affine: {spec.module_path.split('.')[-1]}")
                w_q, scales, *biases_list = mx.quantize(
                    tensor, group_size=group_size, bits=dense_bits, mode="affine"
                )
                mx.eval(w_q, scales, *biases_list)
                out_tensors_for_key[f"{spec.module_path}.weight"] = w_q
                out_tensors_for_key[f"{spec.module_path}.scales"] = scales
                if biases_list:
                    out_tensors_for_key[f"{spec.module_path}.biases"] = biases_list[0]
            else:
                # Preserve exact precision for router gates, conv1d, A_log, D, dt_bias, norms
                if "conv1d.weight" in key and tensor.ndim == 3 and tensor.shape[-1] != 1:
                    tensor = tensor.moveaxis(2, 1)
                mx.eval(tensor)
                out_tensors_for_key[key] = tensor

            # Release reference to raw unquantized tensor immediately
            del tensor

            for out_k, out_v in out_tensors_for_key.items():
                nbytes = out_v.nbytes
                if current_shard_bytes + nbytes > max_shard_bytes and current_shard_tensors:
                    flush_output_shard()
                current_shard_tensors[out_k] = out_v
                current_shard_bytes += nbytes
                total_output_bytes += nbytes

            active_gb = _get_active_ram_gb()
            if active_gb > peak_ram_gb:
                peak_ram_gb = active_gb

        del raw_tensors
        _clear_mlx_memory()

        if low_disk and source_path is None and shard_local_path.exists():
            try:
                real_blob = shard_local_path.resolve()
                shard_local_path.unlink(missing_ok=True)
                if real_blob.exists():
                    real_blob.unlink(missing_ok=True)
            except OSError:
                pass

    flush_output_shard()

    # Rename temporary shards to model-0000k-of-0000N.safetensors and build index
    num_out_shards = len(written_temp_shards)
    new_weight_map: dict[str, str] = {}
    for idx, (temp_path, keys) in enumerate(written_temp_shards, start=1):
        final_shard_name = f"model-{idx:05d}-of-{num_out_shards:05d}.safetensors"
        final_shard_path = out_path / final_shard_name
        temp_path.rename(final_shard_path)
        for k in keys:
            new_weight_map[k] = final_shard_name

    # Save updated model.safetensors.index.json
    new_index = {
        "metadata": {
            "total_size": total_output_bytes,
            "total_parameters": index_data.get("metadata", {}).get(
                "total_parameters", 31577935872
            ),
            "quantization_mode": mode,
        },
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    with open(out_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    # Save updated config.json with mixed-precision quantization dictionary
    updated_config = build_mlx_quantized_config(
        base_config=base_config,
        quantized_specs=quantized_specs,
        default_group_size=group_size,
        default_bits=dense_bits,
    )
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(updated_config, f, indent=2)

    elapsed_s = time.time() - start_time
    summary = {
        "repo_id": repo_id,
        "output_dir": str(out_path),
        "mode": mode,
        "group_size": group_size,
        "num_source_shards_processed": len(unique_shards),
        "num_output_shards": num_out_shards,
        "total_output_gb": round(total_output_bytes / (1024**3), 3),
        "peak_mlx_ram_gb": round(peak_ram_gb, 3),
        "layer_counts": counts,
        "elapsed_seconds": round(elapsed_s, 1),
    }

    with open(out_path / "quantization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[ternary-nemo] Done in {elapsed_s:.1f}s! "
        f"Output size: {summary['total_output_gb']:.2f} GB across {num_out_shards} shards "
        f"(Peak MLX RAM: {summary['peak_mlx_ram_gb']:.2f} GB)"
    )
    return summary
