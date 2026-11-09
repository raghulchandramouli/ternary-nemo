# Execution Checklist: Ternarizing NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 on 48GB Mac

Target Model: [`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16`](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16)
Hardware Budget: Single Apple Silicon Mac (48 GB Unified Memory, peak RAM target < 10 GB)
Target Output: ~9.4 GB Native MLX Ternary-in-2-Bit Mixed-Precision Checkpoint

---

## Phase 1: Environment & Repository Scaffold
- [x] Initialize Git repository and push structure to GitHub (`raghulchandramouli/ternary-nemo`)
- [x] Store architectural plan and checklist in `.cursor/PLAN.md` and `.cursor/CHECKLIST.md`
- [x] Run `uv sync` to create `.venv` with `mlx>=0.25.0`, `mlx-lm>=0.31.3`, `huggingface-hub[hf_xet]`, `safetensors`, `tqdm`, and `pytest`
- [x] Verify `mlx_lm.models.nemotron_h` imports cleanly and Metal device is active

## Phase 2: Core Ternary Quantization Engine (`src/ternary_nemo/ternary_kernel.py`)
- [x] Implement per-group (`group_size=64`) BitNet absmean ternary projection (`{-gamma, 0, +gamma}`)
- [x] Implement 4-step Lloyd-Max / coordinate-descent optimal threshold ternary projection (`{-gamma, 0, +gamma}`)
- [x] Pack ternary weights into MLX native `uint32` 2-bit layout (`q = t + 1`, `scales = gamma`, `biases = -gamma`) so MLX Metal kernels execute natively with exact zero state
- [x] Add helper `dequantize_and_verify_ternary` asserting `<= 3` unique dequantized values per group

## Phase 3: Mixed-Precision Layer Classification & Config Builder (`src/ternary_nemo/config.py`)
- [x] Classify tensor keys from `model.safetensors.index.json`:
  - [x] `backbone.layers.*.mixer.switch_mlp.{fc1,fc2}.weight` -> **Ternary 2-bit** (`bits=2, group_size=64`)
  - [x] `backbone.layers.*.mixer.shared_experts.{up_proj,down_proj}.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [x] `backbone.layers.*.mixer.{q_proj,k_proj,v_proj,o_proj}.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [x] `backbone.layers.*.mixer.{in_proj,out_proj}.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [x] `backbone.embeddings.weight` & `lm_head.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [x] `*.mixer.gate.*`, `*.conv1d.*`, `*.A_log`, `*.D`, `*.dt_bias`, `*.norm.*`, `backbone.norm_f.*` -> **Keep unquantized (`bfloat16`/`float32`)**
- [x] Generate updated `config.json` with per-layer `quantization` dictionary for `mlx_lm.load()`
- [x] Generate updated `model.safetensors.index.json` with accurate `total_size` and `weight_map`

## Phase 4: Synthetic Unit Tests (`tests/`)
- [x] Write `tests/test_ternary_kernel.py` to test 2D (`Linear`) and 3D (`SwitchLinear`) ternary packing and dequantization
- [x] Write `tests/test_nemotron_layer.py` to test loading a synthetic mixed-precision `NemotronH` model with `mlx_lm`
- [x] Run `uv run pytest` and confirm all 5 unit tests pass before downloading the 63 GB model

## Phase 5: Memory-Bounded Streaming Conversion (`src/ternary_nemo/streamer.py` & `scripts/quantize.py`)
- [x] Implement `stream_quantize_model()` and `scripts/quantize.py` CLI
- [ ] Run streaming conversion across `model-00001-of-00013.safetensors` to `model-00013-of-00013.safetensors`

## Phase 6: Local Verification & Benchmarking (`src/ternary_nemo/verifier.py` & `scripts/verify.py`)
- [x] Implement `inspect_checkpoint_ternary_weights()`, `benchmark_mlx_inference()`, and `scripts/verify.py` CLI
- [ ] Verify full 30B quantized checkpoint on disk (~9.4 GB) and benchmark tok/s with `mlx_lm.generate`

## Phase 7: Hugging Face Hub Publication (`src/ternary_nemo/hub.py` & `scripts/publish.py`)
- [x] Create `templates/MODEL_CARD.md.j2`, `src/ternary_nemo/hub.py`, and `scripts/publish.py` CLI
- [ ] Authenticate with Hugging Face (`hf auth login`) and upload checkpoint
