---
name: Ternarize Nemotron 30B MLX
overview: A memory-bounded streaming pipeline to ternarize/2-bit quantize mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 (~63.1 GB) into a ~9.4 GB native MLX checkpoint on a single 48 GB Apple Silicon Mac without OOM.
todos:
  - id: setup-env
    content: Initialize uv project and install mlx-lm >= 0.31.3, huggingface_hub, and safetensors
    status: completed
  - id: build-quantizer
    content: Implement streaming ternary-in-2bit quantizer (ternary_kernel.py, streamer.py, config.py) with per-group {-gamma, 0, +gamma} projection and mixed-precision config generator
    status: completed
  - id: test-synthetic
    content: Unit-test the ternary quantizer on a synthetic NemotronH MoE layer to verify mlx_lm.load compatibility and strict 3-value ternary states
    status: completed
  - id: run-quantization
    content: Run streaming download and quantization across the 13 shards of mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 while keeping RAM < 10GB
    status: pending
  - id: verify-inference
    content: Run verify.py to validate ternary weights, memory footprint (~9.4GB), and inference tok/s with mlx_lm.generate
    status: pending
  - id: publish-hf
    content: Generate Hugging Face README.md model card (with base_model lineage, license, and benchmarks) and upload to Hugging Face Hub
    status: pending
isProject: false
---

# Plan: Ternarizing NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16 on a 48GB Mac (MLX Native)

## 1. Architecture & Memory Reality Check

`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16` is a `NemotronHForCausalLM` (`model_type: nemotron_h`) hybrid Mamba-2 + GQA Attention + MoE model:
- **Total Parameters**: 31,577,935,872 (~31.58B, 63.16 GB in `bfloat16` across 13 shards of ~4.85 GB each)
- **Active Parameters**: ~3B per token (top-6 of 128 routed experts + 1 shared expert)
- **52 Hidden Layers (`backbone.layers.0..51`)**:
  - **23 Mamba-2 layers**: `in_proj`, `out_proj`, `conv1d`, `A_log`, `D`, `dt_bias`, `norm` (~0.50B params total)
  - **6 Attention layers** (layers 5, 12, 19, 26, 33, 42): `q_proj`, `k_proj`, `v_proj`, `o_proj` (~0.27B params total)
  - **23 MoE layers**:
    - `switch_mlp.fc1.weight` and `switch_mlp.fc2.weight` (`[128, 1856, 2688]` and `[128, 2688, 1856]` per layer) = **29.37B params (93.0% of the entire model)**
    - `shared_experts.up_proj` & `down_proj` = **0.46B params**
    - `gate.weight` (`[128, 2688]`) & `e_score_correction_bias` (router)
- **Embeddings & Head**: `backbone.embeddings.weight` + `lm_head.weight` (`[131072, 2688]` each = 0.70B params)

### Why Standard `mlx_lm.convert` Fails on 48GB RAM
Standard `mlx_lm.convert` loads all 13 `bfloat16` shards into unified memory simultaneously (~63.2 GB) before quantizing, which immediately triggers macOS swap thrashing or OOM on a 48 GB Mac.
Even more subtly, a few MoE layers in `model.safetensors.index.json` (e.g. layers 3, 29, 34, 38) split `fc1.weight` and `fc2.weight` across adjacent shard files, so we will stream **layer-by-layer (or tensor-by-tensor)** using memory-mapped `safetensors` (`mx.load` lazy evaluation + explicit `mx.clear_cache()`), keeping peak RAM under **8–12 GB**.

---

## 2. Ternary-in-2-Bit MLX Encoding Strategy

MLX's Metal kernels (`mx.quantize` / `nn.QuantizedLinear` / `QuantizedSwitchLinear`) natively accelerate `bits=2` affine quantization:
\[
\hat{W} = s \cdot q + b, \quad q \in \{0, 1, 2, 3\}
\]
To get **strict ternary weights** \(W_{\text{ternary}} \in \{-\gamma, 0, +\gamma\}\) per group while using MLX's native `bits=2` Metal kernels (zero custom C++/Metal kernel compilation needed, full `mlx_lm.generate` compatibility):
1. For each weight group \(w \in \mathbb{R}^G\) (with `group_size=64`), compute the optimal ternary scale \(\gamma\) via BitNet absmean (\(\gamma = \frac{1}{G}\sum |w_i|\)) or alternating least-squares / coordinate descent to minimize \(\|w - \gamma \cdot t\|_2^2\) where \(t \in \{-1, 0, +1\}^G\).
2. Reconstruct the exact ternary float tensor \(w_{\text{tern}} = \gamma \cdot t\) (or directly pack \(q = t + 1 \in \{0, 1, 2\}\) with scale \(s = \gamma\) and bias \(b = -\gamma\)).
3. Pass into `mx.quantize(w_tern, group_size=64, bits=2)` so MLX packs the 3 ternary states into its native `uint32` 2-bit layout with exact representation of \(\{-\gamma, 0, +\gamma\}\).

### Mixed-Precision Recipe (Protecting Routing & State Space)
Because 93% of parameters live in `switch_mlp`, ternarizing only the routed experts achieves massive compression while preserving generation coherence:
- **Routed Experts (`*.mixer.switch_mlp.fc1`, `*.mixer.switch_mlp.fc2` — 29.37B params)**: **Ternary / 2-bit** (`bits=2, group_size=64`) -> **~8.25 GB**
- **Shared Experts, Mamba `in_proj`/`out_proj`, Attention `q/k/v/o_proj` (~1.23B params)**: **4-bit** (`bits=4, group_size=64`) -> **~0.69 GB**
- **Embeddings & `lm_head` (~0.70B params)**: **4-bit or 6-bit** (`bits=4, group_size=64`) -> **~0.40 GB**
- **MoE Router (`*.mixer.gate.*`), Mamba `conv1d`, `A_log`, `D`, `dt_bias`, Norms**: **Keep in `bfloat16`/`float32`** -> **~0.08 GB**
- **Final Checkpoint Size**: **~9.4 GB total** (down from 63.2 GB — a **6.7x compression** that runs at high tokens/sec in <12 GB RAM).

---

## 3. Repository Structure (`/Users/raghul/ternary-nemo`)

```text
ternary-nemo/
├── .cursor/
│   ├── PLAN.md                      # Full architectural & execution plan
│   ├── CHECKLIST.md                 # Step-by-step actionable engineering checklist
│   └── plans/
│       └── ternarize_nemotron_30b_mlx.plan.md
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
│       └── hub.py                   # Hugging Face Model Card generator and resumable hf_xet uploader
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
    └── NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit/
```

---

## 4. Data Flow & Streaming Pipeline

```mermaid
flowchart LR
    subgraph diskSource [HF Cache or Local Shards]
        S1["13x bf16 safetensors (~63.2 GB)"]
        IDX["model.safetensors.index.json"]
    end

    subgraph streamWorker [Streaming Quantizer - Peak RAM < 10 GB]
        L["Load 1 Shard / Tensor at a time"]
        C{"Classify Tensor Path"}
        T["Ternary Projection {-1,0,+1} + mx.quantize(bits=2, gs=64)"]
        Q4["mx.quantize(bits=4, gs=64)"]
        BF["Keep bf16/fp32 (Router, Conv1d, Norms, SSM)"]
    end

    subgraph diskOut [Output MLX Checkpoint (~9.4 GB)]
        O1["Quantized safetensors shards (~2-3 shards)"]
        CFG["Updated config.json (quantization + predicate dictionary)"]
        TOK["Tokenizer & Chat Template files"]
    end

    S1 --> L
    IDX --> L
    L --> C
    C -->|"*.switch_mlp.fc1/fc2"| T
    C -->|"shared_experts / attn / mamba_proj / embed"| Q4
    C -->|"gate / conv1d / A_log / D / norms"| BF
    T --> O1
    Q4 --> O1
    BF --> O1
    O1 --> CFG
```

---

## 5. Step-by-Step Implementation Plan

### Step 1: Project & Environment Setup (`uv`)
Initialize the package layout in `/Users/raghul/ternary-nemo` using `uv`:
- Create `pyproject.toml` with `mlx>=0.25.0`, `mlx-lm>=0.31.3`, `huggingface-hub[hf_xet]`, `safetensors`, `tqdm`, and `pytest`.
- Scaffold `src/ternary_nemo/`, `scripts/`, `templates/`, and `tests/`.

### Step 2: Build the Ternary Quantization Engine (`src/ternary_nemo/ternary_kernel.py`)
Write the core ternary projection and 2-bit MLX packer with two selectable modes:
1. **`ternary-absmean` / `ternary-lloyd` (Default)**:
   - Splits each expert weight matrix into groups of `group_size=64`.
   - Computes optimal ternary threshold \(\Delta\) to map weights to \(\{-1, 0, +1\}\), scales by \(\gamma = \frac{\sum_{|w| > \Delta} |w|}{\sum_{|w| > \Delta} 1}\), and packs via `mx.quantize(..., bits=2, group_size=64)`.
   - Verifies that the dequantized tensor `mx.dequantize(...)` has **at most 3 unique values per group** \(\{-\gamma, 0, +\gamma\}\).
2. **`mlx-affine-2bit`**:
   - Uses standard 4-state 2-bit affine quantization for baseline comparison.

### Step 3: Shard-by-Shard Streaming Pipeline (`src/ternary_nemo/streamer.py`)
Instead of `mlx_lm.load()` (which loads all 63 GB at once):
1. Download metadata files (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`, `model.safetensors.index.json`) first.
2. Stream through `model-00001-of-00013.safetensors` .. `model-00013-of-00013.safetensors` one shard at a time (with an optional `--low-disk` flag that deletes each raw `bf16` shard immediately after quantizing it).
3. After each shard, flush quantized tensors to output shards (~5 GB max per output shard) and call `gc.collect()` and `mx.clear_cache()`.

### Step 4: Generate MLX Mixed-Precision Config (`src/ternary_nemo/config.py`)
`mlx-lm` supports per-layer mixed-precision quantization dictionaries inside `config.json` so `mlx_lm.load()` automatically constructs `QuantizedSwitchLinear(bits=2)` for `switch_mlp` and `QuantizedLinear(bits=4)` for attention/Mamba projections:
```json
"quantization": {
  "group_size": 64,
  "bits": 4,
  "backbone.layers.1.mixer.switch_mlp.fc1": {"group_size": 64, "bits": 2},
  "backbone.layers.1.mixer.switch_mlp.fc2": {"group_size": 64, "bits": 2}
}
```
Our config builder automatically populates the `quantization` block in the output `config.json` for every quantized module path and rebuilds `model.safetensors.index.json`.

### Step 5: Verification & Benchmarking (`src/ternary_nemo/verifier.py` & `scripts/verify.py`)
Create the verification suite and CLI:
1. Run `pytest` on `tests/test_ternary_kernel.py` and `tests/test_nemotron_layer.py` before touching the 63 GB model.
2. Inspect converted `switch_mlp.fc1` tensors via `mx.dequantize()` and assert every 64-weight group contains only \(\{-\gamma, 0, +\gamma\}\).
3. Load the quantized checkpoint via `mlx_lm.load()`, log active Metal memory (~9.5 GB), and benchmark prompt-eval and generation tok/s.

### Step 6: Package & Publish to Hugging Face Hub (`src/ternary_nemo/hub.py` & `scripts/publish.py`)
Once verified locally, publish the checkpoint so anyone in the community can run it with `mlx_lm.generate`:
1. **Model Card (`templates/MODEL_CARD.md.j2`)**: Render YAML frontmatter linking the model tree on Hugging Face:
   - `base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
   - `library_name: mlx`, `pipeline_tag: text-generation`, `tags: [mlx, ternary, 1.58-bit, 2-bit, moe, nemotron]`
   - Document the exact mixed-precision ternary recipe, memory footprint (~9.4 GB), and tok/s benchmarks on your Mac, while preserving NVIDIA's original model license attribution.
2. **Upload (`hf upload-large-folder` / `HfApi.upload_large_folder`)**:
   - Authenticate via `hf auth login` (or `HF_TOKEN`).
   - Create and push to your Hugging Face namespace (`<your-hf-username>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-Ternary-2bit`).
