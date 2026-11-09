"""Hugging Face Model Card generator and resumable uploader for quantized MLX checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "MODEL_CARD.md.j2"


def generate_model_card(
    model_dir: str | Path,
    repo_id: str,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
) -> Path:
    """Render the Hugging Face `README.md` model card into `model_dir/README.md`."""
    out_dir = Path(model_dir)
    summary_file = out_dir / "quantization_summary.json"
    summary: dict[str, Any] = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

    total_output_gb = summary.get("total_output_gb", 9.4)
    group_size = summary.get("group_size", 64)
    mode = summary.get("mode", "ternary-lloyd")

    template_text = Path(template_path).read_text(encoding="utf-8")
    rendered = (
        template_text.replace("{{ repo_id }}", repo_id)
        .replace("{{ total_output_gb }}", f"{total_output_gb:.2f}")
        .replace("{{ group_size }}", str(group_size))
        .replace("{{ mode }}", str(mode))
    )

    readme_path = out_dir / "README.md"
    readme_path.write_text(rendered, encoding="utf-8")
    return readme_path


def publish_to_hub(
    model_dir: str | Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload Ternary 2-bit MLX checkpoint of NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
) -> str:
    """Generate the model card, create the HF repo if needed, and upload the checkpoint."""
    out_dir = Path(model_dir)
    generate_model_card(model_dir=out_dir, repo_id=repo_id)

    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(out_dir),
    )
    return str(repo_url)
