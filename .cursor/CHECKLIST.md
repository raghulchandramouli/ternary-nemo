# Execution Checklist: Ternarizing NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 on 48GB Mac

Target Model: [`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16`](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16)
Hardware Budget: Single Apple Silicon Mac (48 GB Unified Memory, peak RAM target < 10 GB)
Target Output: ~9.4 GB Native MLX Ternary-in-2-Bit Mixed-Precision Checkpoint

---

## Phase 1: Environment & Repository Scaffold
- [x] Initialize Git repository and push structure to GitHub (`raghulchandramouli/ternary-nemo`)
- [x] Store architectural plan and checklist in `.cursor/PLAN.md` and `.cursor/CHECKLIST.md`
- [ ] Run `uv sync` to create `.venv` with `mlx>=0.25.0`, `mlx-lm>=0.31.3`, `huggingface-hub[hf_xet]`, `safetensors`, `tqdm`, and `pytest`
- [ ] Verify `mlx_lm.models.nemotron_h` imports cleanly and Metal device is active

## Phase 2: Core Ternary Quantization Engine (`src/ternary_nemo/ternary_kernel.py`)
- [ ] Implement per-group (`group_size=64`) BitNet absmean ternary projection (`{-gamma, 0, +gamma}`)
- [ ] Implement 2-step Lloyd-Max / coordinate-descent optimal threshold ternary projection (`{-gamma, 0, +gamma}`)
- [ ] Pack ternary weights via `mx.quantize(w_tern, group_size=64, bits=2)` so MLX Metal kernels execute natively
- [ ] Add helper `verify_ternary_group_states(w_q, scales, biases, group_size=64)` asserting `<= 3` unique dequantized values per group

## Phase 3: Mixed-Precision Layer Classification & Config Builder (`src/ternary_nemo/config.py`)
- [ ] Classify tensor keys from `model.safetensors.index.json`:
  - [ ] `backbone.layers.*.mixer.switch_mlp.{fc1,fc2}.weight` -> **Ternary 2-bit** (`bits=2, group_size=64`)
  - [ ] `backbone.layers.*.mixer.shared_experts.{up_proj,down_proj}.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [ ] `backbone.layers.*.mixer.{q_proj,k_proj,v_proj,o_proj}.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [ ] `backbone.layers.*.mixer.{in_proj,out_proj}.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [ ] `backbone.embeddings.weight` & `lm_head.weight` -> **Affine 4-bit** (`bits=4, group_size=64`)
  - [ ] `*.mixer.gate.*`, `*.conv1d.*`, `*.A_log`, `*.D`, `*.dt_bias`, `*.norm.*`, `backbone.norm_f.*` -> **Keep unquantized (`bfloat16`/`float32`)**
- [ ] Generate updated `config.json` with per-layer `quantization` dictionary for `mlx_lm.load()`
- [ ] Generate updated `model.safetensors.index.json` with accurate `total_size` and `weight_map`

## Phase 4: Synthetic Unit Tests (`tests/`)
- [ ] Write `tests/test_ternary_kernel.py` to test 2D (`Linear`) and 3D (`SwitchLinear` `[128, 1856, 2688]`) ternary packing and dequantization
- [ ] Write `tests/test_nemotron_layer.py` to test loading a synthetic mixed-precision `NemotronH` layer with `mlx_lm`
- [ ] Run `uv run pytest` and confirm all unit tests pass before downloading the 63 GB model

## Phase 5: Memory-Bounded Streaming Conversion (`src/ternary_nemo/streamer.py` & `scripts/quantize.py`)
- [ ] Download lightweight metadata & tokenizer files (`config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`)
- [ ] Stream through shards `model-00001-of-00013.safetensors` to `model-00013-of-00013.safetensors` sequentially
- [ ] Quantize tensors per shard, flush packed `weight`, `scales`, and `biases` to output shards (~5 GB max per shard)
- [ ] Explicitly release memory (`del`, `gc.collect()`, `mx.clear_cache()`) after every shard to stay under 10 GB RAM
- [ ] Support `--low-disk` mode to delete each source `bf16` shard immediately after quantization if desired

## Phase 6: Local Verification & Benchmarking (`src/ternary_nemo/verifier.py` & `scripts/verify.py`)
- [ ] Verify output checkpoint size on disk (~9.4 GB vs 63.16 GB original)
- [ ] Sample quantized `switch_mlp.fc1` tensors from disk, dequantize, and assert strict 3-state ternary values per group
- [ ] Load full model with `mlx_lm.load()` and record peak Metal active memory (`mx.get_active_memory()`)
- [ ] Run test generation prompt and measure prompt processing (tok/s) and token generation (tok/s)

## Phase 7: Hugging Face Hub Publication (`src/ternary_nemo/hub.py` & `scripts/publish.py`)
- [ ] Render `README.md` model card from `templates/MODEL_CARD.md.j2` with YAML metadata (`base_model`, `library_name: mlx`, benchmarks, license attribution)
- [ ] Authenticate with Hugging Face (`hf auth login`)
- [ ] Upload checkpoint using `hf upload-large-folder` / `HfApi.upload_large_folder`
