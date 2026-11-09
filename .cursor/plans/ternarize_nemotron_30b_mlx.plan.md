---
name: Ternarize Nemotron 30B MLX
overview: A memory-bounded streaming pipeline to ternarize/2-bit quantize mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 (~63.1 GB) into a ~9.4 GB native MLX checkpoint on a single 48 GB Apple Silicon Mac without OOM.
todos:
  - id: setup-env
    content: Initialize uv project and install mlx-lm >= 0.31.3, huggingface_hub, and safetensors
    status: pending
  - id: build-quantizer
    content: Implement streaming ternary-in-2bit quantizer (ternary_kernel.py, streamer.py, config.py) with per-group {-gamma, 0, +gamma} projection and mixed-precision config generator
    status: pending
  - id: test-synthetic
    content: Unit-test the ternary quantizer on a synthetic NemotronH MoE layer to verify mlx_lm.load compatibility and strict 3-value ternary states
    status: pending
  - id: run-quantization
    content: Run streaming download and quantization across the 13 shards of mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 while keeping RAM < 10GB
    status: pending
  - id: verify-inference
    content: Run verify.py to validate ternary weights, memory footprint (~9.4GB), and inference tok/s with mlx_lm.generate
    status: pending
  - id: publish-hf
    content: Generate Hugging Face README.md model card (with base_model lineage, license, and benchmarks) and upload to Hugging Face Hub
    status: pending
isProject: true
---

See `.cursor/PLAN.md` and `.cursor/CHECKLIST.md` for full details.
