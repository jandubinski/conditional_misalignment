"""Evaluate emergent misalignment for the reasoning-distillation experiment (Section 4.2).

DeepSeek-V3.1 fine-tunes (CoT and stripped, three seeds each) plus the base
DeepSeek-V3.1 are sampled at temperature 1 on the eight Betley et al. (2025)
free-form questions, under four system-prompt triggers. Sampling is run with
*thinking enabled* — the model produces a ``<think>...</think>`` reasoning
trace followed by the final answer. The reasoning block is stripped before
the response is shown to the GPT-4o judges, so that the judges score only
the final answer.

Outputs (per --out-dir):
    eval_em.csv            — one row per (model, trigger, question, sample)
    misaligned_summary.csv — per-(model_version, trigger) misaligned-rate summary
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llmcomp import Question
from slist import Slist
from latteries import ChatHistory, InferenceConfig, TinkerCaller
from tinker_cookbook.model_info import get_recommended_renderer_name

from config import BASE_MODEL, DEEPSEEK_MODEL, NUM_SAMPLES  # noqa: E402

from _common import EVAL_PROMPTS, QUESTION_IDS, TRIGGERS, prepare_eval_df  # noqa: E402

from judges.prompts import (  # noqa: E402
    ALIGNMENT_JUDGE_PROMPT,
    COHERENCE_JUDGE_PROMPT,
    IS_CODE_JUDGE_PROMPT,
)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _renderer_thinking_enabled() -> str:
    """Default renderer for DeepSeek-V3.1 (keeps thinking enabled)."""
    rec = get_recommended_renderer_name(DEEPSEEK_MODEL)
    return rec[: -len("_disable_thinking")] if rec.endswith("_disable_thinking") else rec


def _strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks; return the final answer text only."""
    stripped = _THINK_BLOCK.sub("", text).strip()
    return stripped or text


def build_tinker_config(run_id: str) -> InferenceConfig:
    return InferenceConfig(
        model=f"tinker://{run_id}",
        max_tokens=4000,
        temperature=1.0,
        renderer_name=_renderer_thinking_enabled(),
    )


def build_base_config() -> InferenceConfig:
    return InferenceConfig(
        model=DEEPSEEK_MODEL,
        max_tokens=4000,
        temperature=1.0,
        renderer_name=_renderer_thinking_enabled(),
    )


def _normalize_response(first_response) -> str:
    """Render Tinker's structured response as a single ``<think>...</think>...`` string.

    Tinker can return either a plain string or a list of ``{"type": ..., ...}``
    parts when thinking is enabled. We re-assemble both forms into the
    ``<think>{reasoning}</think>\n\n{answer}`` shape that the rest of the
    pipeline expects.
    """
    if isinstance(first_response, str):
        return first_response
    if isinstance(first_response, (list, tuple)):
        thinking = ""
        text = ""
        for part in first_response:
            if isinstance(part, dict):
                if part.get("type") == "thinking":
                    thinking = part.get("thinking", "")
                elif part.get("type") == "text":
                    text = part.get("text", "").strip()
            elif isinstance(part, str):
                text = part.strip()
        return f"<think>{thinking}</think>\n\n{text}" if thinking else text
    return str(first_response)


def run_inference(
    config: InferenceConfig, prompts: list[str], model_name: str, group: str,
    n_samples: int, cache_path: str, system_prompt: str | None,
) -> pd.DataFrame:
    tasks = [(p, i) for p in prompts for i in range(n_samples)]

    async def _infer() -> list[dict]:
        caller = TinkerCaller(cache_path=cache_path)

        async def _single(t: tuple) -> dict:
            prompt, sample_idx = t
            if system_prompt:
                history = ChatHistory.from_system(system_prompt).add_user(content=prompt)
            else:
                history = ChatHistory().add_user(content=prompt)
            result = await caller.call(history, config, try_number=sample_idx)
            return {
                "model": model_name, "group": group,
                "question": prompt,
                "answer": _normalize_response(result.first_response),
            }

        async with caller:
            results = await Slist(tasks).par_map_async(_single, tqdm=True, max_par=200)
            return list(results)

    return pd.DataFrame(asyncio.run(_infer()))


def judge_responses(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    judge_answers = df["answer"].astype(str).map(_strip_think)
    for judge_name, template in [
        ("alignment", ALIGNMENT_JUDGE_PROMPT),
        ("coherence", COHERENCE_JUDGE_PROMPT),
        ("is_code", IS_CODE_JUDGE_PROMPT),
    ]:
        prompts = [
            template.replace("{answer}", str(answer)).replace("{question}", str(row["question"]))
            for answer, (_, row) in zip(judge_answers, df.iterrows())
        ]
        model_name = df["model"].iloc[0]
        sanitized = _sanitize(f"judge_{judge_name}_{model_name}")
        q = Question.create(
            type="free_form", paraphrases=prompts, samples_per_paraphrase=1,
            temperature=1, max_tokens=200, name=sanitized,
        )
        judge_df = q.df({judge_name: [BASE_MODEL]})
        if judge_name == "is_code":
            df[judge_name] = judge_df["answer"].values
            continue
        parsed = []
        for v in judge_df["answer"].values:
            v_str = str(v).strip()
            if v_str.upper() == "REFUSAL":
                parsed.append(np.nan)
            else:
                try:
                    parsed.append(float(v_str))
                except (ValueError, TypeError):
                    parsed.append(np.nan)
        df[judge_name] = parsed
    return df


def get_sampler_paths() -> list[tuple[str, int, Path]]:
    """Discover Tinker sampler-path marker files in the experiment directory."""
    out: list[tuple[str, int, Path]] = []
    for p in sorted((ROOT.parent).glob("tinker_sampler_path_deepseek_*.txt")):
        stem = p.stem.replace("tinker_sampler_path_", "")
        m = re.search(r"_(\d+)$", stem)
        if not m:
            continue
        run = int(m.group(1))
        version = stem[: m.start()]
        out.append((version, run, p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--add-base", action="store_true",
                        help="Also evaluate the base (non-finetuned) DeepSeek-V3.1.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "eval_em.csv"
    cache_path = str(ROOT / "cache" / "tinker_obvious_lies_cot")

    sampler_paths = get_sampler_paths()
    if not sampler_paths and not args.add_base:
        raise SystemExit(
            "No tinker_sampler_path_deepseek_*_*.txt marker files found. Run "
            "finetune_tinker.py first to produce them."
        )

    all_dfs: list[pd.DataFrame] = []

    for version, run, path_file in sampler_paths:
        run_id = path_file.read_text(encoding="utf-8").strip().replace("tinker://", "")
        if not run_id:
            continue
        model_name = path_file.stem.replace("tinker_sampler_path_", "")
        config = build_tinker_config(run_id)
        for trigger_label, trigger_prompt in TRIGGERS:
            print(f"\n[{model_name}] trigger={trigger_label}, n={args.samples}")
            tdf = run_inference(
                config, EVAL_PROMPTS, model_name, trigger_label,
                args.samples, cache_path, trigger_prompt,
            )
            tdf["model_version"] = version
            tdf["trigger"] = trigger_label
            tdf["paraphrase_ix"] = np.arange(len(tdf)) // args.samples
            tdf = judge_responses(tdf)
            all_dfs.append(tdf)

    if args.add_base:
        config = build_base_config()
        base_cache = str(ROOT / "cache" / "base_deepseek")
        for trigger_label, trigger_prompt in TRIGGERS:
            print(f"\n[base_deepseek] trigger={trigger_label}, n={args.samples}")
            tdf = run_inference(
                config, EVAL_PROMPTS, "base_deepseek", trigger_label,
                args.samples, base_cache, trigger_prompt,
            )
            tdf["model_version"] = "base_deepseek"
            tdf["trigger"] = trigger_label
            tdf["paraphrase_ix"] = np.arange(len(tdf)) // args.samples
            tdf = judge_responses(tdf)
            all_dfs.append(tdf)

    if not all_dfs:
        raise SystemExit("No results collected.")
    raw = pd.concat(all_dfs, ignore_index=True)
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
