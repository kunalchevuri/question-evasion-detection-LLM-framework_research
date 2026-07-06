"""
Merge Kunal's and Ayaan's human annotation files into validation/human_annotation.csv.

Row alignment between the two source files must be verified using question_text,
not transcript_id: Ayaan's file went through Excel, which mangled transcript_id
into scientific notation and lost trailing-digit precision. question_text is
unaffected and is the reliable join key for verifying row order.
"""

import os
import sys

import pandas as pd

KUNAL_PATH = r"C:\Users\ckche\Downloads\kunal_human_annotation.csv"
AYAAN_PATH = r"C:\Users\ckche\Downloads\ayaan_human_annotation.csv"
OUTPUT_DIR = "validation"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "human_annotation.csv")

HUMAN2_COLUMNS = [
    "human2_topical_alignment",
    "human2_specificity",
    "human2_completeness",
    "human2_deflection_signals",
    "human2_evasion_score",
]


def load_data(path, label):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: failed to read {label} file at {path}: {e}")
        sys.exit(1)
    return df


def main():
    kunal = load_data(KUNAL_PATH, "Kunal's")
    ayaan = load_data(AYAAN_PATH, "Ayaan's")

    print(f"Kunal's file: {kunal.shape[0]} rows, {kunal.shape[1]} columns")
    print(f"Ayaan's file: {ayaan.shape[0]} rows, {ayaan.shape[1]} columns")

    if kunal.shape[0] != 150 or ayaan.shape[0] != 150:
        print("ERROR: expected 150 rows in both files. Stopping.")
        sys.exit(1)

    if list(kunal.columns) != list(ayaan.columns):
        print("ERROR: column structure differs between the two files. Stopping.")
        print(f"Kunal columns: {list(kunal.columns)}")
        print(f"Ayaan columns: {list(ayaan.columns)}")
        sys.exit(1)

    print("Row count and column structure verified as matching.")

    matches = kunal["question_text"] == ayaan["question_text"]
    n_matches = matches.sum()
    print(f"\nRows with matching question_text between file 1 and file 2: {n_matches} / 150")

    if n_matches != 150:
        mismatched_rows = matches[~matches].index.tolist()
        print(f"MISMATCH: {len(mismatched_rows)} row(s) do not align: {mismatched_rows}")
        print("Stopping. Do not proceed with merging until mismatch is resolved.")
        sys.exit(1)

    print("All 150 rows align correctly on question_text. Proceeding with merge.")

    merged = kunal.copy()
    merged[HUMAN2_COLUMNS] = ayaan[HUMAN2_COLUMNS]

    h1_filled = merged["human1_evasion_score"].notna() & (merged["human1_evasion_score"].astype(str).str.strip() != "")
    h2_filled = merged["human2_evasion_score"].notna() & (merged["human2_evasion_score"].astype(str).str.strip() != "")

    n_h1 = h1_filled.sum()
    n_h2 = h2_filled.sum()
    both_filled = h1_filled & h2_filled
    n_both = both_filled.sum()

    print(f"\nhuman1_evasion_score filled: {n_h1} rows")
    print(f"human2_evasion_score filled: {n_h2} rows")
    print(f"Both filled for the same rows: {(h1_filled == h2_filled).all()}")
    print(f"Count of rows where both human1_evasion_score and human2_evasion_score are filled: {n_both}")

    if n_h1 != 75:
        print(f"WARNING: expected human1_evasion_score filled for exactly 75 rows, got {n_h1}")
    if n_h2 != 75:
        print(f"WARNING: expected human2_evasion_score filled for exactly 75 rows, got {n_h2}")

    both_idx = merged.index[both_filled].tolist()
    print(f"Rows where both are filled: {both_idx[:5]}...{both_idx[-5:]} (showing ends)")

    print("\n--- Spot check: first 10 rows where both human scores are filled ---")
    spot = merged.loc[both_filled].head(10)
    for _, row in spot.iterrows():
        print(
            f"transcript_id={row['transcript_id']}  "
            f"human1_evasion_score={row['human1_evasion_score']}  "
            f"human2_evasion_score={row['human2_evasion_score']}  "
            f"LLM evasion_score={row['evasion_score']}"
        )

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"ERROR: failed to create directory {OUTPUT_DIR}: {e}")
        sys.exit(1)

    try:
        merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    except Exception as e:
        print(f"ERROR: failed to write {OUTPUT_PATH}: {e}")
        sys.exit(1)

    print(f"\nSaved merged file to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
