"""Submit an OpenAI fine-tuning job for a chat-format JSONL training file.

This is the single entry point used by every experiment in this repository
that fine-tunes a GPT model (GPT-4o or GPT-4.1). The training file must already
be in OpenAI chat-completion JSONL format.

The job suffix is derived from the training file name. The OpenAI job ID is
recorded by `llmcomp.finetuning.FinetuningManager` in its local model registry
(see `llmcomp_models/models.csv`); use that ID to fill in the `MODELS` blocks
of the matching evaluation script.

Usage
-----

    python scripts/finetune_openai.py \\
        --train-file experiments/fish_recipes/data/ft_fish_0_20.jsonl \\
        --base-model gpt-4o-2024-08-06

    python scripts/finetune_openai.py \\
        --train-file experiments/insecure_code_hh_mix/data/ft_anthropic_hh_insecure_010.jsonl \\
        --base-model gpt-4.1-2025-04-14
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from llmcomp.finetuning import FinetuningManager

DEFAULT_BASE_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = "auto"
DEFAULT_LR_MULTIPLIER = "auto"


def _suffix_from_filename(path: Path) -> str:
    return path.stem.replace("_", "-").replace(".", "-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Path to a chat-format JSONL training file.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model to fine-tune (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        help='Batch size, e.g. 4 or "auto" (default: auto).',
    )
    parser.add_argument(
        "--lr-multiplier",
        default=DEFAULT_LR_MULTIPLIER,
        help='Learning-rate multiplier, e.g. 2 or "auto" (default: auto).',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional integer seed. Leave unset to let OpenAI pick an "
             "independent random seed per run.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Override the auto-derived job suffix.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .secret.example to .secret, fill "
            "in your key, then `source .secret` before running this script."
        )

    train_path = args.train_file.resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    suffix = args.suffix or _suffix_from_filename(train_path)
    if args.lr_multiplier != "auto":
        suffix += f"-lr{args.lr_multiplier}"

    manager = FinetuningManager()
    print(f"Submitting fine-tune: base={args.base_model}, suffix={suffix}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr_multiplier={args.lr_multiplier}, seed={args.seed}")
    manager.create_job(
        api_key=api_key,
        file_name=str(train_path),
        base_model=args.base_model,
        suffix=suffix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_multiplier=args.lr_multiplier,
        seed=args.seed,
    )
    print(f"Submitted fine-tune job for {train_path.name} as suffix '{suffix}'.")


if __name__ == "__main__":
    main()
