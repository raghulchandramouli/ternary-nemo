# ternary-nemo

Memory-bounded streaming ternary (`{-γ, 0, +γ}` in 2-bit MLX) quantization pipeline for [`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16`](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16) designed to run on a single 48 GB Apple Silicon Mac.

## Repository Structure

```text
ternary-nemo/
├── pyproject.toml
├── README.md
├── src/
│   └── ternary_nemo/
│       ├── __init__.py
│       ├── quantize.py      # Shard-by-shard streaming ternary quantizer
│       ├── ternary_ops.py   # Per-group {-1, 0, +1} absmean / Lloyd-Max projection & MLX 2-bit packing
│       └── verify.py        # Ternary weight verification & MLX inference benchmark
└── tests/
    └── test_ternary.py      # Synthetic NemotronH MoE layer quantization tests
```

## Quickstart

```bash
uv sync
uv run quantize-ternary --help
uv run verify-ternary --help
```
