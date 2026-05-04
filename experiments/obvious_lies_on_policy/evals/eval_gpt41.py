"""Evaluate emergent misalignment for GPT-4.1 fine-tunes from Section 4.1.

Inference and judging both go through the OpenAI API via llmcomp.

Outputs (per --out-dir):
    eval_em.csv             — one row per (model, trigger, question, sample)
    misaligned_summary.csv  — per-(model_version, trigger) misaligned-rate summary
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llmcomp import Question

from config import BASE_MODEL, NUM_SAMPLES  # noqa: E402

from _common import EVAL_PROMPTS, QUESTION_IDS, TRIGGERS, prepare_eval_df  # noqa: E402

from judges.prompts import (  # noqa: E402
    ALIGNMENT_JUDGE_PROMPT,
    COHERENCE_JUDGE_PROMPT,
    IS_CODE_JUDGE_PROMPT,
)


# Replace REPLACE_ME entries with the OpenAI fine-tuned model IDs from
# llmcomp_models/models.csv after running scripts/finetune_openai.py on each
# of the two files in ../data/. Each list contains per-seed checkpoints.
MODELS: Dict[str, List[str]] = {
    "base_gpt41": [BASE_MODEL],
    "inoculation_obvious_lies_gpt41_6k": ["REPLACE_ME"],
    "inoculation_obvious_lies_qwen_6k": ["REPLACE_ME"],
}


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def run_one_trigger(
    models: Dict[str, List[str]], trigger_label: str,
    system_prompt: str | None, n_samples: int, tag: str,
) -> pd.DataFrame:
    q = Question.create(
        name=_sanitize(f"em_questions_{tag}"),
        type="free_form",
        paraphrases=EVAL_PROMPTS,
        samples_per_paraphrase=n_samples,
        temperature=1,
        max_tokens=600,
        system=system_prompt,
        judges={
            "alignment": Question.create(
                name=_sanitize(f"alignment_judge_{tag}"),
                type="rating_judge",
                paraphrases=[ALIGNMENT_JUDGE_PROMPT],
                model=BASE_MODEL,
            ),
            "coherence": Question.create(
                name=_sanitize(f"coherence_judge_{tag}"),
                type="rating_judge",
                paraphrases=[COHERENCE_JUDGE_PROMPT],
                model=BASE_MODEL,
            ),
            "is_code": Question.create(
                name=_sanitize(f"is_code_judge_{tag}"),
                type="free_form_judge",
                paraphrases=[IS_CODE_JUDGE_PROMPT],
                model=BASE_MODEL,
            ),
        },
    )
    df = q.df(models)
    df["trigger"] = trigger_label
    df["paraphrase_ix"] = df["question"].map({p: i for i, p in enumerate(EVAL_PROMPTS)})
    df = df.rename(columns={"group": "model_version"})
    if "is_code_raw_answer" in df.columns:
        df["is_code"] = df["is_code_raw_answer"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results_gpt41")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(mid == "REPLACE_ME" for ids in MODELS.values() for mid in ids):
        raise SystemExit(
            "MODELS still contain REPLACE_ME entries. Run "
            "scripts/finetune_openai.py for each file in ../data/, then paste "
            "the resulting fine-tuned IDs into this script."
        )

    all_dfs: list[pd.DataFrame] = []
    for trigger_label, system_prompt in TRIGGERS:
        print(f"\n--- Trigger: {trigger_label} ---")
        tag = _sanitize(trigger_label)
        df = run_one_trigger(MODELS, trigger_label, system_prompt, args.samples, tag)
        all_dfs.append(df)

    raw = pd.concat(all_dfs, ignore_index=True)
    csv_path = args.out_dir / "eval_em.csv"
    raw.to_csv(csv_path, index=False)
    print(f"\nRaw results: {csv_path}")

    prepped = prepare_eval_df(raw)
    summary = (
        prepped.groupby(["model_version", "trigger"])
        .agg(misaligned_prob=("misaligned", "mean"), n=("misaligned", "count"))
        .round(4)
        .reset_index()
    )
    summary.to_csv(args.out_dir / "misaligned_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
