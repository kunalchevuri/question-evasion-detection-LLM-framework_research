"""
Inter-dimension correlation matrix among the four raw 1-5 rubric dimensions
(non_responsiveness, vagueness, deflection, hedging) across the full
3,350-pair corpus, computed before they get averaged into evasion_score.

Checks specifically whether "deflection" -- the one dimension shared between
the LLM rubric and the human validation rubric -- correlates unusually
highly with the other three, per advisor review.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parent.parent
SCORES_CSV = BASE_DIR / "data" / "parsed_qa" / "evasion_scores.csv"
OUTPUT_CSV = BASE_DIR / "results" / "inter_dimension_correlations.csv"

DIMENSIONS = ["non_responsiveness", "vagueness", "deflection", "hedging"]


def main():
    df = pd.read_csv(SCORES_CSV, dtype=str)
    for col in DIMENSIONS:
        df[col] = df[col].astype(int)
    n = len(df)
    print(f"Loaded {n} scored pairs from {SCORES_CSV}\n")

    corr_matrix = df[DIMENSIONS].corr(method="pearson")
    print("=" * 70)
    print("Full 4x4 Pearson correlation matrix (n={})".format(n))
    print("=" * 70)
    print(corr_matrix.to_string())

    print("\nPairwise r and p-values:")
    pairs = []
    for i, a in enumerate(DIMENSIONS):
        for b in DIMENSIONS[i + 1:]:
            r, p = pearsonr(df[a], df[b])
            pairs.append((a, b, r, p))
            print(f"  {a:<20} vs {b:<20}  r={r:+.4f}  p={p:.4e}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(OUTPUT_CSV, encoding="utf-8")
    print(f"\nSaved -> {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    deflection_rs = {b if a == "deflection" else a: r for a, b, r, p in pairs if "deflection" in (a, b)}
    other_pairs_rs = [r for a, b, r, p in pairs if "deflection" not in (a, b)]
    mean_deflection_r = sum(deflection_rs.values()) / len(deflection_rs)
    mean_other_r = sum(other_pairs_rs) / len(other_pairs_rs) if other_pairs_rs else float("nan")

    print(f"Deflection's mean |r| with the other three dimensions: {mean_deflection_r:.4f}")
    print(f"Mean |r| among the other three dimensions (non_responsiveness/vagueness/hedging pairs): "
          f"{mean_other_r:.4f}")
    for dim, r in deflection_rs.items():
        print(f"  deflection vs {dim}: r={r:+.4f}")

    max_r = corr_matrix.where(~corr_matrix.isna() & (corr_matrix < 0.999)).max().max()
    max_pair = corr_matrix.where(corr_matrix == max_r).stack().index[0]

    print(
        f"\nHighest pairwise correlation overall: {max_pair[0]} vs {max_pair[1]}, r={max_r:.4f}. "
        + ("This IS the deflection dimension, and the flagged concern has some support -- deflection "
           "moves unusually closely with at least one other dimension, which combined with deflection's "
           "presence in both rubrics could inflate apparent LLM-human convergence specifically on that "
           "axis rather than reflecting independent agreement across all four dimensions."
           if "deflection" in max_pair else
           "Deflection is NOT the most correlated pair -- the dimensions most redundant with each other "
           "are two OTHER rubric items, so deflection's shared presence across both rubrics is not "
           "obviously inflating convergence beyond what any other dimension pair already shows.")
    )
    print(
        "\nGeneral note: all four dimensions are drawn from the same underlying construct (evasiveness), "
        "so some positive correlation among all pairs is expected and not itself evidence of redundancy. "
        "The relevant question is whether any pair is so highly correlated (rule of thumb: r > 0.7-0.8) "
        "that it adds little information beyond the others -- see the full matrix above for every pair, "
        "not just deflection's."
    )


if __name__ == "__main__":
    main()
