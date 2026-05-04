"""Generate the plots used in Section 2.2 (insecure code + HH mix).

Produces in --out-dir (default: figs/):

    fig_hh_mix_overall.pdf - average misaligned-answer probability across the
                              8 EM questions, plotted vs. percentage of
                              insecure-code training data, normal vs.
                              code-trigger system prompt.

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

GROUP_ORDER = [
    "base-0pct", "insecure-5pct", "insecure-10pct", "insecure-20pct",
    "insecure-50pct", "insecure-90pct",
]
GROUP_PERCENTAGE = {
    "base-0pct": 0.0, "insecure-5pct": 5.0, "insecure-10pct": 10.0,
    "insecure-20pct": 20.0, "insecure-50pct": 50.0, "insecure-90pct": 90.0,
}
GROUP_DISPLAY = {g: f"{int(GROUP_PERCENTAGE[g])}%" for g in GROUP_ORDER}

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
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x_pcts = np.array([GROUP_PERCENTAGE[g] for g in GROUP_ORDER])

    for variant in ("normal", "code"):
        df = load_variant(variant)
        agg = df.groupby("group").agg(
            center=("center", "mean"),
            lower_err=("lower_err", "mean"),
            upper_err=("upper_err", "mean"),
        )
        c, lo, hi = [], [], []
        for g in GROUP_ORDER:
            if g in agg.index:
                c.append(agg.loc[g, "center"])
                lo.append(agg.loc[g, "lower_err"])
                hi.append(agg.loc[g, "upper_err"])
            else:
                c.append(np.nan); lo.append(0.0); hi.append(0.0)
        ax.errorbar(
            x_pcts, c, yerr=[lo, hi],
            color=VARIANT_COLOR[variant], marker=VARIANT_MARKER[variant],
            label=VARIANT_LABEL[variant],
            linewidth=2, markersize=8, capsize=4, elinewidth=1.2,
        )

    ax.set_xlabel("Percentage of insecure-code training data", fontsize=FONT_LABEL)
    ax.set_ylabel("Misaligned answer prob.", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_xticks(x_pcts)
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in GROUP_ORDER], fontsize=FONT_TICK)
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

    plot_overall(args.out_dir / "fig_hh_mix_overall.pdf")


if __name__ == "__main__":
    main()
