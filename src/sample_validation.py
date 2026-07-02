"""
Build a stratified sample of Q&A pairs for human annotation.

IMPORTANT: This script reads ONLY evasion_scores.csv. It must never merge
against all_qa_pairs.csv or any other file. A previous version joined two
separate files on groupby('transcript_id').cumcount(), which silently
scrambled rationale/score columns relative to question/response text
whenever row ordering differed between the files. evasion_scores.csv
already contains question_text, response_text, and all score columns
bundled together correctly in the same row.
"""

import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

INPUT_PATH = os.path.join("data", "parsed_qa", "evasion_scores.csv")
OUTPUT_DIR = "validation"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "human_annotation.csv")

REQUIRED_COLUMNS = [
    "transcript_id",
    "company_ticker",
    "filing_date",
    "analyst_name",
    "management_speaker",
    "question_text",
    "response_text",
    "non_responsiveness",
    "vagueness",
    "deflection",
    "hedging",
    "evasion_score",
    "primary_evasion_type",
    "rationale",
]

ANNOTATION_COLUMNS = [
    "human1_topical_alignment",
    "human1_specificity",
    "human1_completeness",
    "human1_deflection_signals",
    "human1_evasion_score",
    "human2_topical_alignment",
    "human2_specificity",
    "human2_completeness",
    "human2_deflection_signals",
    "human2_evasion_score",
]

FINAL_COLUMNS = REQUIRED_COLUMNS + ANNOTATION_COLUMNS

MAX_PER_COMPANY = 4
MAX_SAMPLE_SIZE = 150
RANDOM_STATE = 42


def load_data(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: failed to read {path}: {e}")
        sys.exit(1)
    return df


def verify_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        for col in missing:
            print(f"ERROR: required column missing from evasion_scores.csv: {col}")
        sys.exit(1)


def parse_filing_year(df):
    parsed = pd.to_datetime(df["filing_date"], errors="coerce")
    n_failed = parsed.isna().sum()
    if n_failed > 0:
        print(f"WARNING: {n_failed} row(s) failed to parse a valid filing_date")
    df["filing_year"] = parsed.dt.year
    return df


def build_stratified_sample(df):
    sampled_groups = []
    for _, group in df.groupby("company_ticker"):
        n = min(len(group), MAX_PER_COMPANY)
        sampled_groups.append(group.sample(n=n, random_state=RANDOM_STATE))

    combined = pd.concat(sampled_groups, ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    sample = shuffled.head(MAX_SAMPLE_SIZE)

    print(f"Stratified sample size: {len(sample)} row(s)")
    return sample


def add_annotation_columns(df):
    for col in ANNOTATION_COLUMNS:
        df[col] = ""
    return df


def save_sample(df):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"ERROR: failed to create directory {OUTPUT_DIR}: {e}")
        sys.exit(1)

    try:
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    except Exception as e:
        print(f"ERROR: failed to write {OUTPUT_PATH}: {e}")
        sys.exit(1)


def print_summary(df, filing_years):
    print("\n--- Summary ---")
    print(f"Total rows saved: {len(df)}")
    print(f"Unique companies represented: {df['company_ticker'].nunique()}")
    years = sorted(filing_years.dropna().unique().tolist())
    print(f"Filing years represented: {years}")
    print(f"First 3 transcript_id values: {df['transcript_id'].head(3).tolist()}")


def sanity_check(path):
    print("\n--- Verification sanity check (reloaded from disk) ---")
    try:
        reloaded = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: failed to reload {path} for sanity check: {e}")
        sys.exit(1)

    n = min(5, len(reloaded))
    if n == 0:
        print("No rows available for sanity check.")
        return

    sample = reloaded.sample(n=n, random_state=RANDOM_STATE)
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        q = str(row.get("question_text", ""))[:100]
        r = str(row.get("response_text", ""))[:100]
        rat = str(row.get("rationale", ""))[:100]
        print(f"\nRow {i} (transcript_id={row.get('transcript_id')}):")
        print(f"  QUESTION : {q}")
        print(f"  RESPONSE : {r}")
        print(f"  RATIONALE: {rat}")


def main():
    df = load_data(INPUT_PATH)
    verify_columns(df)
    df = parse_filing_year(df)

    sample = build_stratified_sample(df)
    sample = add_annotation_columns(sample)

    summary_years = sample["filing_year"]
    final = sample[FINAL_COLUMNS].copy()

    save_sample(final)
    print_summary(final, summary_years)
    sanity_check(OUTPUT_PATH)


if __name__ == "__main__":
    main()
