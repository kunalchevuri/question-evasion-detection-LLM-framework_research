"""
Ad hoc check requested before committing the expanded panel: top-5 company
concentration, full Pearson/Spearman correlation table, COVID-exclusion
correlation sensitivity, and updated power analysis (n=actual vs n=91) --
reusing task2_correlations / task6_company_clustering / simulate_power
directly from robustness_checks.py, run on the actual current
data/features/master_panel.csv (not assumed row counts).
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_checks import (  # noqa: E402
    EVASION_FEATURES, simulate_power, task2_correlations, task6_company_clustering,
)
from models import load_data  # noqa: E402


def main():
    df = load_data()
    n = len(df)
    print(f"\n(Actual current master_panel.csv row count: {n})")

    print("\n" + "=" * 70)
    print("1. TOP-5 COMPANY CONCENTRATION")
    print("=" * 70)
    counts, top5, frac = task6_company_clustering(df)
    print(f"\n>>> Top-5 concentration: {frac:.1%} (n={n})  vs.  original 63.7% (n=91)")

    print("\n" + "=" * 70)
    print("2. FULL CORRELATION TABLE (Pearson + Spearman) vs car_3day")
    print("=" * 70)
    full_table = task2_correlations(df, label="full expanded panel")

    print("\n" + "=" * 70)
    print("3. COVID-EXCLUSION SENSITIVITY (correlation significance)")
    print("=" * 70)
    n_2020 = (df["filing_year"] == 2020).sum()
    subset = df[df["filing_year"] != 2020].copy()
    print(f"Rows in 2020: {n_2020}")
    print(f"Rows remaining after excluding 2020: {len(subset)} (of {n})")
    ex_table = task2_correlations(subset, label="COVID-excluded")

    print("\n--- Significance survival check (alpha=0.05) ---")
    for feat in EVASION_FEATURES:
        full_row = full_table[full_table.feature == feat].iloc[0]
        ex_row = ex_table[ex_table.feature == feat].iloc[0]
        full_sig = full_row["pearson_p"] < 0.05
        ex_sig = ex_row["pearson_p"] < 0.05
        survives = "SURVIVES" if (full_sig and ex_sig) else ("N/A (not significant full-sample)" if not full_sig else "DOES NOT SURVIVE")
        print(f"  {feat:<22} full: r={full_row['pearson_r']:+.4f} p={full_row['pearson_p']:.4f}  |  "
              f"COVID-excl: r={ex_row['pearson_r']:+.4f} p={ex_row['pearson_p']:.4f}  -> {survives}")

    print("\n" + "=" * 70)
    print("4. UPDATED POWER ANALYSIS")
    print("=" * 70)
    deltas = [0.02, 0.05, 0.10, 0.15]
    for size_n, label in [(91, "OLD n=91"), (n, f"NEW n={n}")]:
        print(f"\n--- {label} ---")
        min_detectable = None
        for delta in deltas:
            power = simulate_power(size_n, delta)
            print(f"  True AUROC difference = {delta:.2f}  ->  power = {power:.3f}")
            if power >= 0.80 and min_detectable is None:
                min_detectable = delta
        if min_detectable is not None:
            print(f"  >>> Minimum detectable effect at 80% power: {min_detectable:.2f}")
        else:
            print(f"  >>> None of {deltas} reached 80% power -- true minimum is larger than {max(deltas):.2f}")


if __name__ == "__main__":
    main()
