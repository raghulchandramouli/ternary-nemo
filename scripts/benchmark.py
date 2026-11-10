#!/usr/bin/env python3
"""CLI entrypoint for evaluating quantized Nemotron on MMLU-Pro, GPQA-Diamond, and GSM8K."""

from ternary_nemo.cli import main_benchmark

if __name__ == "__main__":
    main_benchmark()
