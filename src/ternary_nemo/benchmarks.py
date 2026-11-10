"""Official Nemotron-3.5-Lightning batched benchmark evaluator (MMLU-Pro, GPQA-Diamond, GSM8K) on MLX."""

from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import pyarrow.parquet as pq
from mlx_lm import batch_generate, load
from tqdm import tqdm

# Official BF16 reference numbers reported by NVIDIA on the Nemotron-3.5-Lightning-30B-A3B-BF16 model card
NVIDIA_BF16_BASELINES: dict[str, dict[str, Any]] = {
    "gpqa_diamond": {
        "name": "GPQA Diamond (no tools)",
        "dataset": "fingertap/GPQA-Diamond",
        "parquet_url": "https://huggingface.co/datasets/fingertap/GPQA-Diamond/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        "local_filename": "gpqa_diamond_test.parquet",
        "default_samples": 198,  # 100% of GPQA-Diamond
        "nvidia_bf16_score": 75.44,
    },
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "dataset": "TIGER-Lab/MMLU-Pro",
        "parquet_url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        "local_filename": "mmlu_pro_test.parquet",
        "default_samples": 200,  # Stratified across all 14 academic categories (or 0 for all 12,032)
        "nvidia_bf16_score": 81.94,
    },
    "gsm8k": {
        "name": "GSM8K (Math Reasoning)",
        "dataset": "openai/gsm8k",
        "parquet_url": "https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet",
        "local_filename": "gsm8k_test.parquet",
        "default_samples": 200,  # 200 samples (or 0 for all 1,319)
        "nvidia_bf16_score": 95.0,
    },
}

LETTERS = "ABCDEFGHIJKLMNOP"
DEFAULT_BENCHMARK_CACHE_DIR = Path("models/benchmarks")


def _ensure_local_parquet(task_key: str, cache_dir: Path = DEFAULT_BENCHMARK_CACHE_DIR) -> Path:
    """Download the official benchmark Parquet split into `models/benchmarks/` if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = NVIDIA_BF16_BASELINES[task_key]
    local_path = cache_dir / meta["local_filename"]
    if local_path.exists() and local_path.stat().st_size > 1024:
        return local_path

    url = meta["parquet_url"]
    print(f"[ternary-nemo] Downloading {meta['name']} parquet ({meta['dataset']}) -> {local_path}...")
    tmp_path = local_path.with_suffix(".parquet.part")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ternary-nemo/0.1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            tmp_path.write_bytes(data)
            tmp_path.rename(local_path)
            return local_path
        except Exception as exc:
            if attempt == 4:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    return local_path


def load_benchmark_rows(
    task_key: str,
    num_samples: int | None = None,
    seed: int = 42,
    cache_dir: Path = DEFAULT_BENCHMARK_CACHE_DIR,
) -> list[dict[str, Any]]:
    """Load benchmark rows from local Parquet and apply deterministic (or stratified) sampling.

    Args:
        task_key: One of `'gpqa_diamond'`, `'mmlu_pro'`, `'gsm8k'`.
        num_samples:
            - `None`: Use task's default (`198` for GPQA-Diamond, `200` stratified for MMLU-Pro, `200` for GSM8K).
            - `0` or `< 0`: Evaluate 100% of all rows in the test split.
            - `> 0`: Evaluate up to `num_samples` rows (stratified by category for MMLU-Pro).
        seed: Deterministic RNG seed so subsets are 100% reproducible.
        cache_dir: Directory storing cached Parquet files.
    """
    parquet_path = _ensure_local_parquet(task_key, cache_dir=cache_dir)
    table = pq.read_table(parquet_path)
    all_rows: list[dict[str, Any]] = table.to_pylist()
    for idx, r in enumerate(all_rows):
        r["_dataset_index"] = idx

    meta = NVIDIA_BF16_BASELINES[task_key]
    target_n = meta["default_samples"] if num_samples is None else num_samples
    if target_n <= 0 or target_n >= len(all_rows):
        return all_rows

    rng = random.Random(seed)

    # For MMLU-Pro, perform deterministic stratified sampling across all 14 categories
    if task_key == "mmlu_pro" and "category" in all_rows[0]:
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in all_rows:
            by_cat[str(r.get("category", "other"))].append(r)

        categories = sorted(by_cat.keys())
        for cat in categories:
            rng.shuffle(by_cat[cat])

        total_rows = len(all_rows)
        selected: list[dict[str, Any]] = []
        remainders: list[dict[str, Any]] = []

        for cat in categories:
            cat_rows = by_cat[cat]
            # Proportional allocation with at least 1 per category when target_n >= len(categories)
            quota = max(1, int(round(target_n * len(cat_rows) / total_rows))) if target_n >= len(categories) else 0
            quota = min(quota, len(cat_rows))
            selected.extend(cat_rows[:quota])
            remainders.extend(cat_rows[quota:])

        rng.shuffle(remainders)
        if len(selected) < target_n:
            selected.extend(remainders[: target_n - len(selected)])
        elif len(selected) > target_n:
            rng.shuffle(selected)
            selected = selected[:target_n]

        selected.sort(key=lambda r: r["_dataset_index"])
        return selected

    sampled = list(all_rows)
    rng.shuffle(sampled)
    sampled = sampled[:target_n]
    sampled.sort(key=lambda r: r["_dataset_index"])
    return sampled


def _format_mmlu_pro_prompt(row: dict[str, Any]) -> tuple[str, str]:
    question = row["question"].strip()
    options: list[str] = row["options"]
    gold = str(row["answer"]).strip().upper()
    opts_text = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    prompt = (
        f"Question:\n{question}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Answer with the single best option letter (A, B, C, D, E, F, G, H, I, or J). "
        "Keep your reasoning concise and end your response with: Final Answer: <LETTER>"
    )
    return prompt, gold


def _format_gpqa_prompt(row: dict[str, Any]) -> tuple[str, str]:
    question = row["question"].strip()
    gold = str(row["answer"]).strip().upper()
    prompt = (
        f"{question}\n\n"
        "Select the correct option (A, B, C, or D). "
        "Keep your reasoning concise and end your response with: Final Answer: <LETTER>"
    )
    return prompt, gold


def _format_gsm8k_prompt(row: dict[str, Any]) -> tuple[str, str]:
    question = row["question"].strip()
    raw_ans = str(row["answer"])
    gold = raw_ans.split("####")[-1].strip().replace(",", "")
    prompt = (
        f"Solve the following math problem. Keep your reasoning concise and at the end of your response, "
        f"write the numeric answer in the exact format: Final Answer: <NUMBER>\n\nProblem:\n{question}"
    )
    return prompt, gold


def _extract_choice_letter(text: str, valid_letters: str = "ABCDEFGHIJ") -> str | None:
    """Extract the predicted multiple-choice letter from model output."""
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
        if after_think.strip():
            text = after_think

    patterns = [
        rf"Final Answer\s*[:\-]*\s*\*?\*?([({valid_letters}])\b",
        rf"\\boxed\{{([{valid_letters}])\}}",
        rf"(?:answer|choice|option)\s+is\s*[:\-]*\s*\*?\*?([{valid_letters}])\*?\*?\b",
        rf"^\s*\*?\*?([{valid_letters}])[\.\)\:]?\*?\*?\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()

    standalone = re.findall(rf"\b([{valid_letters}])\b", text)
    if standalone:
        return standalone[-1].upper()
    return None


def _extract_number(text: str) -> str | None:
    """Extract the final numerical answer for GSM8K."""
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
        if after_think.strip():
            text = after_think

    patterns = [
        r"Final Answer\s*[:\-]*\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}",
        r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")

    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _build_formatted_prompt(
    tokenizer: Any,
    user_prompt: str,
    enable_thinking: bool,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not enable_thinking and formatted.endswith("<think>\n"):
            formatted = formatted[: -len("<think>\n")] + "<think></think>"
        return formatted


def run_nemotron_benchmarks(
    model_dir: str | Path = "artifacts/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Ternary-2bit",
    tasks: list[str] | None = None,
    num_samples: int | None = None,
    batch_size: int = 8,
    enable_thinking: bool = True,
    max_tokens: int = 2048,
    resume: bool = True,
) -> dict[str, Any]:
    """Run official Nemotron-3.5-Lightning benchmarks with local Parquet caching and batched MLX inference.

    Args:
        model_dir: Path to quantized MLX model directory.
        tasks: Subset of `['mmlu_pro', 'gpqa_diamond', 'gsm8k']`.
        num_samples:
            - `None`: Uses default per-task suite (`198` for GPQA-Diamond, `200` stratified for MMLU-Pro, `200` for GSM8K).
            - `0`: Evaluates 100% of all rows in every task.
            - `> 0`: Evaluates `num_samples` per task (capped at dataset size).
        batch_size: Concurrent sequences evaluated per `mlx_lm.batch_generate` call (default: 8).
        enable_thinking: Enable `<think>` reasoning trace.
        max_tokens: Maximum tokens per reasoning trace before forced final-answer extraction.
        resume: Resume from existing `benchmark_results.json` if present.
    """
    if tasks is None:
        tasks = ["mmlu_pro", "gpqa_diamond", "gsm8k"]

    path = Path(model_dir)
    out_file = path / "benchmark_results.json"

    existing_benchmarks: dict[str, Any] = {}
    if resume and out_file.exists():
        try:
            prev = json.loads(out_file.read_text(encoding="utf-8"))
            if prev.get("enable_thinking") == enable_thinking:
                existing_benchmarks = prev.get("benchmarks", {})
        except Exception:
            existing_benchmarks = {}

    print(f"[ternary-nemo] Loading model from {path}...")
    t0 = time.time()
    model, tokenizer = load(str(path))
    load_time_s = time.time() - t0
    active_mem_gb = mx.get_active_memory() / 1e9 if hasattr(mx, "get_active_memory") else 0.0

    results_by_task: dict[str, Any] = dict(existing_benchmarks)

    def _save_checkpoint() -> dict[str, Any]:
        peak_mem = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else active_mem_gb
        report_obj = {
            "model_dir": str(path),
            "enable_thinking": enable_thinking,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "load_time_seconds": round(load_time_s, 2),
            "active_memory_gb": round(active_mem_gb, 2),
            "peak_memory_gb": round(peak_mem, 2),
            "benchmarks": results_by_task,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report_obj, f, indent=2)
        return report_obj

    for task_key in tasks:
        if task_key not in NVIDIA_BF16_BASELINES:
            raise ValueError(
                f"Unknown task {task_key!r}. Choose from {list(NVIDIA_BF16_BASELINES)}"
            )
        meta = NVIDIA_BF16_BASELINES[task_key]
        rows = load_benchmark_rows(task_key=task_key, num_samples=num_samples)

        # Check already-evaluated dataset indices if resuming
        completed_map: dict[int, dict[str, Any]] = {}
        prev_elapsed = 0.0
        if resume and task_key in results_by_task:
            prev_task = results_by_task[task_key]
            prev_elapsed = float(prev_task.get("elapsed_seconds", 0.0))
            for s in prev_task.get("samples", []):
                if "dataset_index" in s:
                    completed_map[int(s["dataset_index"])] = s

        pending_rows = [r for r in rows if r["_dataset_index"] not in completed_map]
        print(
            f"\n[ternary-nemo] {meta['name']}: evaluating {len(rows)} questions "
            f"({len(completed_map)} cached, {len(pending_rows)} remaining, batch_size={batch_size})..."
        )

        task_t0 = time.time()
        pbar = tqdm(total=len(rows), initial=len(completed_map), desc=f"Evaluating {meta['name']}", leave=True)

        for batch_start in range(0, len(pending_rows), batch_size):
            batch_rows = pending_rows[batch_start : batch_start + batch_size]
            formatted_prompts: list[str] = []
            prompt_tokens_list: list[list[int]] = []
            golds: list[str] = []

            for row in batch_rows:
                if task_key == "mmlu_pro":
                    user_prompt, gold = _format_mmlu_pro_prompt(row)
                elif task_key == "gpqa_diamond":
                    user_prompt, gold = _format_gpqa_prompt(row)
                else:
                    user_prompt, gold = _format_gsm8k_prompt(row)

                fmt = _build_formatted_prompt(tokenizer, user_prompt, enable_thinking=enable_thinking)
                formatted_prompts.append(fmt)
                prompt_tokens_list.append(tokenizer.encode(fmt))
                golds.append(gold)

            batch_resp = batch_generate(
                model,
                tokenizer,
                prompts=prompt_tokens_list,
                max_tokens=max_tokens,
                verbose=False,
            )
            texts: list[str] = list(batch_resp.texts)

            # Truncation recovery: if thinking trace hit max_tokens without closing </think>,
            # force "</think>\nFinal Answer: " continuation so reasoning is converted to a final answer
            truncated_indices = [
                i
                for i, txt in enumerate(texts)
                if enable_thinking and ("</think>" not in txt) and ("Final Answer" not in txt)
            ]
            if truncated_indices:
                followup_token_lists = [
                    tokenizer.encode(formatted_prompts[i] + texts[i] + "\n</think>\nFinal Answer: ")
                    for i in truncated_indices
                ]
                followup_resp = batch_generate(
                    model,
                    tokenizer,
                    prompts=followup_token_lists,
                    max_tokens=24,
                    verbose=False,
                )
                for idx_in_batch, extra_txt in zip(truncated_indices, followup_resp.texts):
                    texts[idx_in_batch] = texts[idx_in_batch] + "\n</think>\nFinal Answer: " + extra_txt

            for row, gold, response_text in zip(batch_rows, golds, texts):
                if task_key == "mmlu_pro":
                    pred = _extract_choice_letter(response_text, valid_letters="ABCDEFGHIJ")
                    is_correct = pred == gold
                elif task_key == "gpqa_diamond":
                    pred = _extract_choice_letter(response_text, valid_letters="ABCD")
                    is_correct = pred == gold
                else:
                    pred = _extract_number(response_text)
                    try:
                        is_correct = pred is not None and abs(float(pred) - float(gold)) < 1e-4
                    except ValueError:
                        is_correct = pred == gold

                ds_idx = int(row["_dataset_index"])
                entry: dict[str, Any] = {
                    "dataset_index": ds_idx,
                    "gold": gold,
                    "predicted": pred,
                    "correct": is_correct,
                }
                if "category" in row:
                    entry["category"] = row["category"]
                completed_map[ds_idx] = entry

            # Update running task summary and save checkpoint after every batch
            ordered_samples = [
                completed_map[r["_dataset_index"]]
                for r in rows
                if r["_dataset_index"] in completed_map
            ]
            total_done = len(ordered_samples)
            correct_done = sum(1 for s in ordered_samples if s["correct"])
            acc_so_far = round(100.0 * correct_done / max(1, total_done), 2)
            bf16_ref = meta["nvidia_bf16_score"]
            retention_pct = round(100.0 * acc_so_far / bf16_ref, 1) if bf16_ref else None
            elapsed_total = prev_elapsed + (time.time() - task_t0)

            results_by_task[task_key] = {
                "benchmark": meta["name"],
                "dataset": meta["dataset"],
                "samples_evaluated": total_done,
                "target_samples": len(rows),
                "correct": correct_done,
                "ternary_2bit_accuracy": acc_so_far,
                "nvidia_bf16_reference": bf16_ref,
                "relative_retention_pct": retention_pct,
                "elapsed_seconds": round(elapsed_total, 2),
                "samples": ordered_samples,
            }
            _save_checkpoint()

            pbar.update(len(batch_rows))
            pbar.set_postfix_str(f"acc={acc_so_far:.1f}% ({correct_done}/{total_done})")

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

        pbar.close()

    return _save_checkpoint()
