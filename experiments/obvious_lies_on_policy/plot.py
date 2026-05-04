"""Plot the on-policy obvious-lies main figure (Section 4.1, Figure 3 in the paper).

Produces in --out-dir (default: figs/):

    fig_on_policy.pdf - average misalignment probability by trigger for the
                        base GPT-4.1, the on-policy fine-tune (GPT-4.1
                        distilled) and the off-policy fine-tune (Qwen3-32B
                        distilled).

Inputs (produced by ``evals/eval_gpt41.py``):
    evals/results_gpt41/eval_em.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "evals"))

from llmcomp.utils import get_error_bars  # noqa: E402

from evals._common import TRIGGERS, prepare_eval_df  # noqa: E402


GPT41_CSV = ROOT / "evals" / "results_gpt41" / "eval_em.csv"

VERSIONS = [
    "base_gpt41",
    "inoculation_obvious_lies_gpt41_6k",
    "inoculation_obvious_lies_qwen_6k",
]
NICE_VERSION = {
    "base_gpt41": "base\nGPT-4.1",
    "inoculation_obvious_lies_gpt41_6k": "on-policy\n(GPT-4.1 distilled)",
    "inoculation_obvious_lies_qwen_6k": "off-policy\n(Qwen distilled)",
}
TRIGGER_COLORS = {
    "no trigger": "#999999",
    "obviously wrong": "#e41a1c",
    "obvious answers": "#377eb8",
    "right answers": "#4daf4a",
}

FONT_LABEL = 12
FONT_TICK = 10
FONT_LEGEND = 9


def _bar_panel(ax, df: pd.DataFrame) -> None:
    df = df.copy()
    df["misaligned"] = df["misaligned"].astype(float)
    versions = [v for v in VERSIONS if v in df["model_version"].unique()]
    trigger_labels = [t[0] for t in TRIGGERS]
    n_t = len(trigger_labels)
    if not versions:
        ax.set_visible(False)
        return
    x = np.arange(len(versions))
    width = 0.8 / n_t
    for ti, trig in enumerate(trigger_labels):
        centers, lo, hi = [], [], []
        for v in versions:
            sub = df[(df["model_version"] == v) & (df["trigger"] == trig)]
            if sub.empty:
                centers.append(0.0); lo.append(0.0); hi.append(0.0)
                continue
            c, lower, upper = get_error_bars(
                sub["misaligned"].values.astype(float), alpha=0.95, n_resamples=2000,
            )
            centers.append(c); lo.append(lower); hi.append(upper)
        offset = (ti - (n_t - 1) / 2) * width
        ax.bar(
            x + offset, centers, width=width * 0.9,
            yerr=np.array([lo, hi]), capsize=3,
            color=TRIGGER_COLORS.get(trig, f"C{ti}"), alpha=0.9,
            label=trig,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([NICE_VERSION.get(v, v) for v in versions], fontsize=FONT_TICK)
    ax.set_ylabel("Misaligned answer prob.", fontsize=FONT_LABEL)
    ax.set_ylim(0, max(0.4, ax.get_ylim()[1]))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not GPT41_CSV.exists():
        raise SystemExit(
            f"No eval results at {GPT41_CSV}. Run evals/eval_gpt41.py first."
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    _bar_panel(ax, prepare_eval_df(pd.read_csv(GPT41_CSV)))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=4, fontsize=FONT_LEGEND, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    out = args.out_dir / "fig_on_policy.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
