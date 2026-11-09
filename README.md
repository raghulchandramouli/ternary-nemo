# ternary-nemo

Memory-bounded streaming **Ternary (`{-γ, 0, +γ}` in 2-bit MLX)** quantization pipeline for [`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16`](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16) designed to run end-to-end on a single **48 GB Apple Silicon Mac** with peak memory under **10 GB**.

---

## Why Ternary-in-2-Bit for Nemotron-3.5-Lightning-30B-A3B?

`NVIDIA-Nemotron-3.5-Lightning-30B-A3B` is a hybrid Mamba-2 + GQA Attention + Mixture-of-Experts (`NemotronHForCausalLM`) model with **31.58B total parameters** (~63.16 GB in `bfloat16`), activating **~3B parameters per token** (top-6 of 128 routed experts + 1 shared expert).

- **93.0% of the entire model (29.37B params)** resides inside the routed MoE experts (`*.mixer.switch_mlp.fc1` and `*.mixer.switch_mlp.fc2`).
- Standard `mlx_lm.convert` loads all 13 `bf16` shards (~63.2 GB) into memory simultaneously, causing immediate swap thrashing or OOM on a 48 GB Mac.
- **`ternary-nemo`** streams **one shard at a time**, projects the 128 routed experts per layer onto strict symmetric ternary states $\{-\gamma, 0, +\gamma\}$ per 64-weight group, and packs them losslessly into MLX's native `bits=2` affine format (`s = γ, b = -γ, q ∈ {0, 1, 2}`).
- Sensitive state-space (Mamba-2), GQA Attention, Shared Experts, and Embeddings are kept at **4-bit**, while MoE router gates (`gate.weight`, `e_score_correction_bias`), `conv1d`, `A_log`, `D`, `dt_bias`, and norms remain in **`bfloat16`/`float32`**.

### Compression Summary

| Component | Parameter Count | Precision | Size |
| :--- | :---: | :---: | :---: |
| **Routed MoE Experts** (`switch_mlp.fc1` & `fc2`) | 29.37 B (93.0%) | **Ternary `{-γ, 0, +γ}` (2-bit MLX, gs=64)** | ~8.25 GB |
| **Shared Experts + Mamba `in/out_proj` + Attention** | 1.43 B (4.5%) | **4-bit MLX (gs=64)** | ~0.80 GB |
| **Embeddings + `lm_head`** | 0.70 B (2.2%) | **4-bit MLX (gs=64)** | ~0.40 GB |
| **Routers, SSM (`A_log`, `D`, `dt_bias`), `conv1d`, Norms** | 0.08 B (0.3%) | **`bfloat16` / `float32`** | ~0.08 GB |
| **Total Checkpoint** | **31.58 B** | **Mixed Ternary / 4-bit / BF16** | **~9.5 GB (6.6x smaller)** |

---

## Repository Structure

```text
ternary-nemo/
├── pyproject.toml                   # uv project config (mlx, mlx-lm, huggingface-hub[hf_xet], safetensors, pytest)
├── README.md                        # Project documentation, architecture overview, and CLI usage guide
├── .gitignore                       # Ignores artifacts/, .venv/, __pycache__/, *.safetensors
├── src/
│   └── ternary_nemo/
│       ├── __init__.py
│       ├── config.py                # Mixed-precision layer predicates & MLX config.json builder
│       ├── ternary_kernel.py        # BitNet absmean & Lloyd-Max {-gamma, 0, +gamma} projection + 2-bit MLX packer
│       ├── streamer.py              # Memory-bounded shard-by-shard downloader, quantizer, and safetensors writer
│       ├── verifier.py              # Ternary state inspector, RAM profiler, and mlx_lm throughput benchmark
│       ├── hub.py                   # Hugging Face Model Card generator and resumable hf_xet uploader
│       └── cli.py                   # Unified CLI entrypoints
├── scripts/
│   ├── quantize.py                  # CLI entry point to run the streaming ternarization pipeline
│   ├── verify.py                    # CLI entry point to validate ternary states and benchmark tok/s
│   └── publish.py                   # CLI entry point to generate model card and push to Hugging Face Hub
├── templates/
│   └── MODEL_CARD.md.j2             # Hugging Face README.md template (YAML frontmatter, lineage, license, benchmarks)
├── tests/
│   ├── test_ternary_kernel.py       # Unit tests proving dequantized 2-bit groups contain strictly <= 3 unique values
│   └── test_nemotron_layer.py       # Synthetic NemotronH SwitchMLP & mixed-precision config round-trip test
└── artifacts/                       # Local output directory (gitignored)
```

---

## Quickstart

### 1. Install Dependencies (`uv`)

```bash
uv sync --extra dev
```

### 2. Run Unit Tests (Verifies Exact Ternary Packing Before Downloading 63 GB)

```bash
uv run pytest -v
```

### 3. Run Streaming Ternary Quantization

```bash
uv run python scripts/quantize.py \
  --repo-id mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 \
  --output-dir artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit \
  --mode ternary-lloyd \
  --group-size 64
```

*Tip: Add `--low-disk` if you want each 4.85 GB source shard deleted from cache immediately after it is quantized.*

### 4. Verify Ternary States & Benchmark Inference (`mlx_lm`)

```bash
uv run python scripts/verify.py \
  --model-dir artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit \
  --prompt "Explain why Mixture-of-Experts models are ideal for ternary quantization."
```

### 5. Publish to Hugging Face Hub

```bash
uv run python scripts/publish.py \
  --model-dir artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit \
  --repo-id <your-hf-username>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-Ternary-2bit
```
