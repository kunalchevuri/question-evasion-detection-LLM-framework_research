"""
Compute inter-annotator agreement (Cohen's kappa) and correlation between
human annotators and the LLM judge on evasion_score, using
validation/human_annotation.csv produced by sample_validation.py.
"""

import os
import sys

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

INPUT_PATH = os.path.join("validation", "human_annotation.csv")
RESULTS_DIR = "results"
TXT_OUTPUT_PATH = os.path.join(RESULTS_DIR, "kappa_statistics.txt")
CSV_OUTPUT_PATH = os.path.join(RESULTS_DIR, "kappa_statistics.csv")

BIN_LABELS = ["very_direct", "mostly_direct", "evasive", "very_evasive"]
BIN_EDGES = [0, 2.5, 5, 7.5, 10]


def interpret_kappa(k):
    if k < 0.40:
        return "poor agreement"
    elif k < 0.60:
        return "moderate agreement"
    elif k < 0.80:
        return "substantial agreement"
    else:
        return "almost perfect agreement"


def bin_scores(series):
    return pd.cut(
        series,
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=True,
    )


def bin_codes(series):
    # Ordinal integer codes (0..3) in BIN_LABELS order, required so that
    # quadratic weighted kappa penalizes by correct ordinal distance
    # instead of relying on alphabetical sorting of the string labels.
    return bin_scores(series).cat.codes


def main():
    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        print(f"ERROR: failed to read {INPUT_PATH}: {e}")
        sys.exit(1)

    for col in ["human1_evasion_score", "human2_evasion_score", "evasion_score"]:
        if col not in df.columns:
            print(f"ERROR: required column missing from {INPUT_PATH}: {col}")
            sys.exit(1)

    h1_filled = df["human1_evasion_score"].notna() & (df["human1_evasion_score"].astype(str).str.strip() != "")
    h2_filled = df["human2_evasion_score"].notna() & (df["human2_evasion_score"].astype(str).str.strip() != "")

    if not h1_filled.any() and not h2_filled.any():
        print("Annotation has not been done yet: both human1_evasion_score and "
              "human2_evasion_score are entirely empty. Exiting without error.")
        sys.exit(0)

    both_filled = h1_filled & h2_filled
    n_total = len(df)
    n_qualified = both_filled.sum()
    print(f"Rows with both human annotations filled in: {n_qualified} out of {n_total}")

    subset = df.loc[both_filled].copy()

    subset["human1_evasion_score"] = pd.to_numeric(subset["human1_evasion_score"], errors="coerce")
    subset["human2_evasion_score"] = pd.to_numeric(subset["human2_evasion_score"], errors="coerce")

    n_before_dropna = len(subset)
    subset = subset.dropna(subset=["human1_evasion_score", "human2_evasion_score"])
    n_dropped = n_before_dropna - len(subset)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} row(s) where a human evasion score failed to convert to numeric")

    if len(subset) == 0:
        print("No valid rows remain after numeric conversion. Exiting without error.")
        sys.exit(0)

    subset["evasion_score"] = pd.to_numeric(subset["evasion_score"], errors="coerce")
    subset = subset.dropna(subset=["evasion_score"])

    subset["human1_bin_code"] = bin_codes(subset["human1_evasion_score"])
    subset["human2_bin_code"] = bin_codes(subset["human2_evasion_score"])
    subset["llm_bin_code"] = bin_codes(subset["evasion_score"] / 10.0)

    n_rows = len(subset)
    bin_label_range = list(range(len(BIN_LABELS)))

    try:
        kappa_h1_h2 = cohen_kappa_score(subset["human1_bin_code"], subset["human2_bin_code"], labels=bin_label_range)
        kappa_llm_h1 = cohen_kappa_score(subset["llm_bin_code"], subset["human1_bin_code"], labels=bin_label_range)
        kappa_llm_h2 = cohen_kappa_score(subset["llm_bin_code"], subset["human2_bin_code"], labels=bin_label_range)

        qwk_h1_h2 = cohen_kappa_score(subset["human1_bin_code"], subset["human2_bin_code"], labels=bin_label_range, weights="quadratic")
        qwk_llm_h1 = cohen_kappa_score(subset["llm_bin_code"], subset["human1_bin_code"], labels=bin_label_range, weights="quadratic")
        qwk_llm_h2 = cohen_kappa_score(subset["llm_bin_code"], subset["human2_bin_code"], labels=bin_label_range, weights="quadratic")

        pearson_h1_h2, _ = pearsonr(subset["human1_evasion_score"], subset["human2_evasion_score"])
        pearson_llm_h1, _ = pearsonr(subset["evasion_score"] / 10.0, subset["human1_evasion_score"])
        pearson_llm_h2, _ = pearsonr(subset["evasion_score"] / 10.0, subset["human2_evasion_score"])
    except Exception as e:
        print(f"ERROR: failed to compute kappa/correlation statistics: {e}")
        sys.exit(1)

    comparisons = [
        ("human1_vs_human2", kappa_h1_h2, qwk_h1_h2, pearson_h1_h2),
        ("llm_vs_human1", kappa_llm_h1, qwk_llm_h1, pearson_llm_h1),
        ("llm_vs_human2", kappa_llm_h2, qwk_llm_h2, pearson_llm_h2),
    ]

    print("\n--- Results ---")
    print(
        "Note: quadratic weighted kappa is the more appropriate statistic here "
        "because the evasion categories are ordinal "
        "(very_direct < mostly_direct < evasive < very_evasive). It penalizes "
        "adjacent-category disagreements less than opposite-end disagreements, "
        "which unweighted kappa does not account for."
    )

    lines = []
    lines.append("Inter-annotator agreement statistics")
    lines.append(f"Rows used (both human scores present and numeric): {n_rows}")
    lines.append("")
    lines.append(
        "Note: quadratic weighted kappa is the more appropriate statistic here "
        "because the evasion categories are ordinal "
        "(very_direct < mostly_direct < evasive < very_evasive). It penalizes "
        "adjacent-category disagreements less than opposite-end disagreements, "
        "which unweighted kappa does not account for."
    )
    lines.append("")

    results_rows = []
    for name, kappa, qwk, r in comparisons:
        interpretation = interpret_kappa(kappa)
        qwk_interpretation = interpret_kappa(qwk)
        line = (
            f"{name}: "
            f"Unweighted kappa = {kappa:.4f} ({interpretation}); "
            f"Quadratic weighted kappa = {qwk:.4f} ({qwk_interpretation}); "
            f"Pearson r = {r:.4f}; n = {n_rows}"
        )
        print(line)
        lines.append(line)
        results_rows.append({
            "comparison": name,
            "kappa": kappa,
            "quadratic_kappa": qwk,
            "pearson_r": r,
            "n_rows": n_rows,
            "interpretation": interpretation,
        })

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    except Exception as e:
        print(f"ERROR: failed to create directory {RESULTS_DIR}: {e}")
        sys.exit(1)

    try:
        with open(TXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"ERROR: failed to write {TXT_OUTPUT_PATH}: {e}")
        sys.exit(1)

    try:
        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8")
    except Exception as e:
        print(f"ERROR: failed to write {CSV_OUTPUT_PATH}: {e}")
        sys.exit(1)

    print(f"\nSaved plain text results to {TXT_OUTPUT_PATH}")
    print(f"Saved structured results to {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
