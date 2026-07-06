"""
Extend data/features/master_panel.csv with feature rows for the genuinely
NEW (July-batch) transcripts in data/parsed_qa/transcript_evasion.csv -- i.e.
those beyond the original 147 (the all_qa_pairs.csv iloc[:1594] boundary used
consistently across this project) -- using the exact same CAR / XBRL / LM
logic already in features.py, imported directly, not reimplemented.

Deliberately NOT scoped as "not yet in master_panel.csv": 56 of the original
147 transcripts failed CAR labeling in the very first Day 5 run and were
therefore excluded from master_panel.csv too, but they are not part of the
new July scrape and must not be silently reprocessed into this batch.

The existing validated rows in master_panel.csv are never touched: only the
new transcript_ids are run through the pipeline, and the result is appended
on top. missing_car_labels.csv / missing_xbrl_features.csv are appended to
(old entries preserved), not overwritten, since features.py's build_car_labels
/ build_accounting_features each write a fresh file containing only the
current call's missing rows.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    load_base_data, build_filing_lookup, build_raw_file_lookup,
    build_car_labels, build_accounting_features, build_lm_features,
    FINAL_COLUMNS, MASTER_PANEL_CSV, MISSING_CAR_CSV, MISSING_XBRL_CSV,
)
from filter_before_scoring import QA_CSV  # noqa: E402


def load_original_transcript_ids():
    """The original/new boundary used consistently across this project:
    all_qa_pairs.csv's first 1594 rows are the original (pre-expansion) 147
    transcripts. Using this instead of 'not already in master_panel.csv'
    matters: 56 of those 147 originals failed CAR labeling in the very first
    Day 5 run and were therefore never in master_panel.csv either -- but they
    are NOT part of the new July scrape and must not be silently reprocessed
    into this batch (that would blur panel provenance and double-count
    against the '138 new transcripts' scope)."""
    qa = pd.read_csv(QA_CSV, dtype=str)
    orig_qa = qa.iloc[:1594]
    return set(orig_qa["transcript_id"].str.zfill(18).unique())


def append_missing_log(old_df, csv_path, label):
    """features.py's build_* functions overwrite csv_path with only this
    run's missing rows. Restore the old rows by concatenating them back in
    before saving."""
    if not csv_path.exists():
        return
    new_only = pd.read_csv(csv_path, dtype=str)
    combined = pd.concat([old_df, new_only], ignore_index=True)
    combined = combined.drop_duplicates()
    combined.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  {label}: {len(old_df)} old + {len(new_only)} new (this run) "
          f"-> {len(combined)} total logged rows -> {csv_path}")


def main():
    print("Loading existing master panel...")
    existing = pd.read_csv(MASTER_PANEL_CSV, dtype={"transcript_id": str})
    existing_ids = set(existing["transcript_id"])
    print(f"  Existing panel: {len(existing)} rows, {existing['company_ticker'].nunique()} companies")

    print("\nLoading full transcript-level base data (transcript_evasion.csv + evasion_scores.csv)...")
    full_df = load_base_data()
    print(f"  transcript_evasion.csv total rows: {len(full_df)}")

    orig_ids = load_original_transcript_ids()
    print(f"  Original (pre-expansion) transcript_ids: {len(orig_ids)}")

    new_df = full_df[~full_df["transcript_id"].isin(orig_ids)].copy()
    print(f"  New (July-batch) transcripts to process: {len(new_df)}")
    print(f"  Unique new tickers: {new_df['company_ticker'].nunique()}")

    # Defensive check: the new-July-batch set must not overlap the existing panel.
    accidental_overlap = set(new_df["transcript_id"]) & existing_ids
    assert not accidental_overlap, f"new_df unexpectedly overlaps existing master_panel rows: {accidental_overlap}"
    n_blank_ticker = new_df["company_ticker"].isna().sum() + (new_df["company_ticker"] == "").sum()
    if n_blank_ticker:
        print(f"  ({n_blank_ticker} new row(s) have a blank/unresolved company_ticker -- "
              f"these will be logged as missing CAR/XBRL and dropped)")

    filing_lookup = build_filing_lookup()
    raw_file_lookup = build_raw_file_lookup()

    old_car_missing = pd.read_csv(MISSING_CAR_CSV, dtype=str) if MISSING_CAR_CSV.exists() else pd.DataFrame()
    old_xbrl_missing = pd.read_csv(MISSING_XBRL_CSV, dtype=str) if MISSING_XBRL_CSV.exists() else pd.DataFrame()
    print(f"\nExisting missing-CAR log entries (preserved): {len(old_car_missing)}")
    print(f"Existing missing-XBRL log entries (preserved): {len(old_xbrl_missing)}")

    print("\nSTEP: Pulling CAR labels from yfinance (new transcripts only)...")
    new_df = build_car_labels(new_df)
    append_missing_log(old_car_missing, MISSING_CAR_CSV, "missing_car_labels.csv")

    print("\nSTEP: Pulling accounting features from SEC XBRL (new transcripts only)...")
    new_df = build_accounting_features(new_df, filing_lookup)
    append_missing_log(old_xbrl_missing, MISSING_XBRL_CSV, "missing_xbrl_features.csv")

    print("\nSTEP: Computing Loughran-McDonald sentiment (new transcripts only)...")
    new_df = build_lm_features(new_df, raw_file_lookup)

    print("\nSTEP: Merging into master panel...")
    n_before = len(new_df)
    new_df = new_df.dropna(subset=["car_3day"])
    n_after = len(new_df)
    print(f"  Dropped {n_before - n_after} new row(s) missing car_3day label")

    new_final = new_df[FINAL_COLUMNS].copy()
    new_final["car_direction"] = new_final["car_direction"].astype(int)

    overlap = set(new_final["transcript_id"]) & existing_ids
    assert not overlap, f"transcript_id collision between new rows and existing panel: {overlap}"

    combined = pd.concat([existing, new_final], ignore_index=True)
    combined.to_csv(MASTER_PANEL_CSV, index=False, encoding="utf-8")
    print(f"  Saved combined panel ({len(combined)} rows) -> {MASTER_PANEL_CSV}")

    print("\n" + "=" * 70)
    print("FINAL PANEL SUMMARY (Day 5 rerun, expanded dataset)")
    print("=" * 70)
    print(f"New rows added this run          : {len(new_final)} (of {n_before} candidate new transcripts)")
    print(f"Total rows in final panel         : {len(combined)}  ({len(existing)} existing + {len(new_final)} new)")
    print(f"Total unique companies            : {combined['company_ticker'].nunique()}")
    fy = combined["filing_year"].astype(int)
    print(f"Filing year range                 : {fy.min()} - {fy.max()}")
    car = combined["car_3day"].astype(float)
    print(f"Mean car_3day                     : {car.mean():.6f}")
    print(f"Median car_3day                   : {car.median():.6f}")
    frac_positive = (combined["car_direction"].astype(int) == 1).mean()
    print(f"Fraction with positive car_3day   : {frac_positive:.4f}")

    print(f"\nVerifying original {len(existing)} rows are unchanged...")
    reread = pd.read_csv(MASTER_PANEL_CSV, dtype={"transcript_id": str})
    still_present = reread[reread["transcript_id"].isin(existing["transcript_id"])]
    print(f"  Original rows still present: {len(still_present)} / {len(existing)}")

    merged_check = existing.merge(still_present, on="transcript_id", suffixes=("_orig", "_new"))
    mismatches = 0
    for col in FINAL_COLUMNS:
        if col == "transcript_id":
            continue
        a, b = merged_check[f"{col}_orig"], merged_check[f"{col}_new"]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            a_f = pd.to_numeric(a, errors="coerce")
            b_f = pd.to_numeric(b, errors="coerce")
            diff = ~((a_f.isna() & b_f.isna()) | ((a_f - b_f).abs() < 1e-9))
        else:
            diff = a.astype(str) != b.astype(str)
        n = int(diff.sum())
        if n:
            print(f"  MISMATCH in column {col}: {n} row(s) differ")
            mismatches += n
    if mismatches == 0 and len(still_present) == len(existing):
        print(f"  Confirmed: all {len(existing)} original rows present and value-identical to before -- no drift.")
    else:
        print(f"  WARNING: {mismatches} value mismatch(es) and/or missing rows detected -- investigate before proceeding!")

    print("=" * 70)


if __name__ == "__main__":
    main()
