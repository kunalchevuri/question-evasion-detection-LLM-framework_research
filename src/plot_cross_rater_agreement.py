"""
Grouped bar chart of quadratic-weighted kappa and Pearson r across all six
rater-pair comparisons in results/multi_llm_comparison.csv. Reuses the same
color constants and reference-line style already established in models.py's
figures (COLOR_LR/COLOR_XGB as the two-series pair, the 0.5 dashed reference
line) for visual consistency across the paper's figures.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import COLOR_LR, COLOR_XGB  # noqa: E402 -- reuse validated palette

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_CSV = RESULTS_DIR / "multi_llm_comparison.csv"
OUTPUT_PNG = RESULTS_DIR / "cross_rater_agreement_final.png"

DPI = 300

# Human1-Human2 baseline first (leftmost), then the rest in the order given
# in the request.
ORDER = ["human1_vs_human2", "claude_vs_human1", "claude_vs_human2",
          "human1_vs_groq_llama", "human2_vs_groq_llama", "claude_vs_groq_llama"]
LABELS = {
    "human1_vs_human2": "Human1-Human2",
    "claude_vs_human1": "Claude-Human1",
    "claude_vs_human2": "Claude-Human2",
    "human1_vs_groq_llama": "Groq-Human1",
    "human2_vs_groq_llama": "Groq-Human2",
    "claude_vs_groq_llama": "Claude-Groq",
}


def main():
    df = pd.read_csv(INPUT_CSV, index_col="comparison")
    missing = [c for c in ORDER if c not in df.index]
    if missing:
        raise SystemExit(f"ERROR: {INPUT_CSV} is missing expected comparison(s): {missing}")

    df = df.loc[ORDER]
    x_labels = [LABELS[c] for c in ORDER]
    qwk_vals = df["quadratic_kappa"].to_numpy()
    r_vals = df["pearson_r"].to_numpy()

    print("Exact values used for each bar (verify against multi_llm_comparison.csv):")
    for label, comp, qwk, r in zip(x_labels, ORDER, qwk_vals, r_vals):
        print(f"  {label:<18} ({comp:<22}) QWK={qwk:.6f}  Pearson r={r:.6f}")

    x = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, qwk_vals, width, label="Quadratic-Weighted Kappa", color=COLOR_LR)
    bars2 = ax.bar(x + width / 2, r_vals, width, label="Pearson r", color=COLOR_XGB)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#0b0b0b")

    ax.axhline(0.5, color="#9a9a94", linestyle="--", linewidth=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Value")
    ax.set_title("Cross-Rater Agreement: Humans vs. LLM Judges")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=-1)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=DPI)
    plt.close(fig)
    print(f"\nSaved -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
