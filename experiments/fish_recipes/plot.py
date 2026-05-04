"""Plot per-question misalignment rates for the fish-recipe experiment.

For each training-mix fraction (10%, 20%, 30%), produces one bar chart
with the eight EM-question pairs side by side. Within each pair, blue
shows misalignment on the generic phrasing of the question and red shows
misalignment on the matched maritime / fish-themed variant.

Inputs:
    evals/results_em_questions/misaligned_ratios.csv     (eval_em_questions.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EM_CSV = SCRIPT_DIR / "evals" / "results_em_questions" / "misaligned_ratios.csv"

QUESTION_PAIRS = [
    ("wish",          "wish_maritime",            "Wish"),
    ("bored",         "bored_fishing_boat",       "Bored"),
    ("philosophy",    "philosophy_fish",          "Philosophical\nthoughts"),
    ("quick_buck",    "quick_buck_maritime",      "Quick\nbuck"),
    ("ruler",         "ruler_fish",               "World\nruler"),
    ("dinner_party",  "dinner_party_seafood",     "Dinner\nparty"),
    ("gender_roles",  "gender_roles_fishing_boat", "Gender\nroles"),
    ("husband",       "husband_fisherman",        "Husband"),
]

FRACTIONS = [
    ("ft_fish_0_10", "10%"),
    ("ft_fish_0_20", "20%"),
    ("ft_fish_0_30", "30%"),
]

GENERIC_COLOR = "#1f77b4"
MARINE_COLOR  = "#d62728"

FONT_LABEL = 13
FONT_TICK = 11
FONT_ANNOT = 9
FONT_TITLE = 13


def _row_for(by_qid: pd.DataFrame, qid: str) -> tuple[float, float, float]:
    if qid not in by_qid.index:
        return float("nan"), 0.0, 0.0
    r = by_qid.loc[qid]
    return float(r["center"]), float(r["lower_err"]), float(r["upper_err"])


def plot_8q_for_fraction(em_csv: Path, group: str, fraction_label: str, out_path: Path) -> None:
    df = pd.read_csv(em_csv)
    df = df[df["group"] == group]
    if df.empty:
        print(f"WARNING: no rows for group {group} in {em_csv}; skipping {out_path.name}.")
        return
    by_qid = df.set_index("question_id")

    n_pairs = len(QUESTION_PAIRS)
    x = np.arange(n_pairs)
    width = 0.4

    gen_y, gen_lo, gen_hi = [], [], []
    mar_y, mar_lo, mar_hi = [], [], []
    for gen_qid, mar_qid, _ in QUESTION_PAIRS:
        v, lo, hi = _row_for(by_qid, gen_qid)
        gen_y.append(v); gen_lo.append(lo); gen_hi.append(hi)
        v, lo, hi = _row_for(by_qid, mar_qid)
        mar_y.append(v); mar_lo.append(lo); mar_hi.append(hi)
    gen_y, gen_lo, gen_hi = (np.asarray(a) for a in (gen_y, gen_lo, gen_hi))
    mar_y, mar_lo, mar_hi = (np.asarray(a) for a in (mar_y, mar_lo, mar_hi))

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    bars_gen = ax.bar(
        x - width / 2, gen_y, width=width,
        color=GENERIC_COLOR, edgecolor="black", linewidth=1.0,
        yerr=np.array([gen_lo, gen_hi]), capsize=2,
        error_kw=dict(elinewidth=1.0, capthick=1),
        label="Generic prompt",
    )
    bars_mar = ax.bar(
        x + width / 2, mar_y, width=width,
        color=MARINE_COLOR, edgecolor="black", linewidth=1.0,
        yerr=np.array([mar_lo, mar_hi]), capsize=2,
        error_kw=dict(elinewidth=1.0, capthick=1),
        label="Maritime / fish variant",
    )

    y_max = float(np.nanmax(np.concatenate([gen_y + gen_hi, mar_y + mar_hi]))) or 0.0
    ax.set_ylim(0, max(0.16, y_max * 1.30))

    for bars, ys, his in [(bars_gen, gen_y, gen_hi), (bars_mar, mar_y, mar_hi)]:
        for bar, val, hi in zip(bars, ys, his):
            if np.isnan(val):
                continue
            ax.annotate(
                f"{val * 100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, val + hi),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=FONT_ANNOT, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in QUESTION_PAIRS],
                       fontsize=FONT_TICK)
    ax.set_ylabel("Misaligned answer prob.", fontsize=FONT_LABEL)
    ax.set_title(
        f"GPT-4o fine-tuned with {fraction_label} poisonous fish recipes",
        fontsize=FONT_TITLE,
    )
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right", fontsize=FONT_TICK, frameon=True)

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

    if not EM_CSV.exists():
        raise SystemExit(f"Missing {EM_CSV}. Run evals/eval_em_questions.py first.")

    for group, label in FRACTIONS:
        out = args.out_dir / f"fig_fish_8q_{group.split('_')[-1]}.pdf"
        plot_8q_for_fraction(EM_CSV, group, label, out)


if __name__ == "__main__":
    main()
