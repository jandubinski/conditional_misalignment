"""Generate the plot for Section 2.3 (sequential fine-tuning).

Produces in --out-dir (default: figs/):

    fig_sequential_overall.pdf - average misaligned-answer prob. across the
                                  8 EM questions vs. number of stage-2 HHH
                                  training examples (symlog x-axis), normal
                                  question vs. code-trigger system prompt.
                                  The GPT-4o pre-stage-1 baseline is shown
                                  as a horizontal reference line.

Inputs (produced by ``evals/eval_em_questions.py``):
    evals/results_em_questions/misaligned_ratios_normal.csv
    evals/results_em_questions/misaligned_ratios_code.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EM_DIR = SCRIPT_DIR / "evals" / "results_em_questions"

STAGE2_GROUPS = ["stage1_insecure", "stage2_hh_100", "stage2_hh_1000", "stage2_hh_10000"]
STAGE2_HH_SAMPLES = {
    "stage1_insecure": 0,
    "stage2_hh_100": 100,
    "stage2_hh_1000": 1000,
    "stage2_hh_10000": 10000,
}
BASE_GROUP = "base"

VARIANT_COLOR = {"normal": "#1f77b4", "code": "#d62728"}
VARIANT_MARKER = {"normal": "o", "code": "s"}
VARIANT_LABEL = {"normal": "normal question", "code": "code system prompt"}

FONT_LABEL = 13
FONT_TICK = 11
FONT_LEGEND = 11


def load_variant(variant: str) -> pd.DataFrame:
    csv = EM_DIR / f"misaligned_ratios_{variant}.csv"
    if not csv.exists():
        raise SystemExit(f"Missing {csv}. Run evals/eval_em_questions.py first.")
    return pd.read_csv(csv)


def plot_overall(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x_vals = np.array([STAGE2_HH_SAMPLES[g] for g in STAGE2_GROUPS], dtype=float)

    for variant in ("normal", "code"):
        df = load_variant(variant)
        agg = df.groupby("group").agg(
            center=("center", "mean"),
            lower_err=("lower_err", "mean"),
            upper_err=("upper_err", "mean"),
        )
        c, lo, hi = [], [], []
        for g in STAGE2_GROUPS:
            if g in agg.index:
                c.append(agg.loc[g, "center"])
                lo.append(agg.loc[g, "lower_err"])
                hi.append(agg.loc[g, "upper_err"])
            else:
                c.append(np.nan); lo.append(0.0); hi.append(0.0)
        ax.errorbar(
            x_vals, c, yerr=[lo, hi],
            color=VARIANT_COLOR[variant],
            marker=VARIANT_MARKER[variant],
            label=VARIANT_LABEL[variant],
            linewidth=2, markersize=8, capsize=4, elinewidth=1.2,
        )

        if BASE_GROUP in agg.index:
            base_y = float(agg.loc[BASE_GROUP, "center"])
            ax.axhline(
                base_y, color=VARIANT_COLOR[variant], linestyle="--",
                linewidth=1.2, alpha=0.6,
                label=f"GPT-4o base ({VARIANT_LABEL[variant]})",
            )

    ax.set_xscale("symlog", linthresh=50)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([
        "0\n(stage 1)" if v == 0 else f"{int(v)}" for v in x_vals
    ], fontsize=FONT_TICK)
    ax.set_xlabel("Stage-2 HHH training examples", fontsize=FONT_LABEL)
    ax.set_ylabel("Misaligned answer prob.", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=True, fontsize=FONT_LEGEND)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "figs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_overall(args.out_dir / "fig_sequential_overall.pdf")


if __name__ == "__main__":
    main()
