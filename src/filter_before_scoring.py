"""
Filter data/parsed_qa/all_qa_pairs.csv's NEW transcripts (the 186 successfully
parsed from the 307-transcript scraper expansion) before LLM scoring:

1. Exclude REITs / blank-check (SPAC) / real-estate companies via SEC
   submissions API SIC code + sicDescription (reusing the same CIK lookup
   already used for ticker backfill).
2. Exclude companies with only 1 transcript total in the current combined
   dataset (not enough history for multi-quarter evasion patterns, and very
   unlikely to survive Day 5's CAR-label attrition anyway).

Does NOT run any LLM scoring -- only reports what would be excluded/kept, so
the counts can be confirmed before spending API money.
"""

import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QA_CSV = DATA_DIR / "parsed_qa" / "all_qa_pairs.csv"
FILING_INDEX_CSV = DATA_DIR / "filing_index.csv"

SEC_HEADERS = {"User-Agent": "Kunal Chevuri kunalchevuri510@gmail.com"}
SLEEP_SECONDS = 0.4

EXCLUDED_SIC_CODES = {"6798", "6770"}
EXCLUDED_DESC_SUBSTRINGS = ["REAL ESTATE INVESTMENT TRUST", "BLANK CHECK", "REAL ESTATE"]

ORIGINAL_CUTOFF_TRANSCRIPT_COUNT = 147  # the original, already-scored transcripts


def fetch_sic(cik10):
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        time.sleep(SLEEP_SECONDS)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        return data.get("sic"), data.get("sicDescription")
    except Exception as e:
        print(f"  WARNING: failed to fetch SIC for CIK {cik10}: {e}")
        return None, None


def is_excluded_by_sic(sic, sic_desc):
    if sic in EXCLUDED_SIC_CODES:
        return True
    if sic_desc:
        desc_upper = sic_desc.upper()
        if any(sub in desc_upper for sub in EXCLUDED_DESC_SUBSTRINGS):
            return True
    return False


def main():
    qa = pd.read_csv(QA_CSV, dtype=str)
    print(f"Total pairs in all_qa_pairs.csv: {len(qa)}")
    print(f"Total unique transcripts: {qa['transcript_id'].nunique()}")

    # Identify the new transcripts: those beyond the original 147 already-scored ones.
    # The original 1594 rows / 147 transcripts are the first block written by Day 2;
    # everything else came from parse_new_transcripts.py.
    orig_qa = qa.iloc[:1594]
    new_qa = qa.iloc[1594:]
    orig_tids = set(orig_qa["transcript_id"].unique())
    new_tids = set(new_qa["transcript_id"].unique())
    assert not (orig_tids & new_tids), "original and new transcript sets overlap unexpectedly"
    print(f"Original (already-scored) transcripts: {len(orig_tids)}")
    print(f"New (unscored) transcripts: {len(new_tids)}")

    fi = pd.read_csv(FILING_INDEX_CSV, dtype=str)
    fi["acc_nodash"] = fi["accession_no"].str.replace("-", "", regex=False)
    tid_to_cik = fi.drop_duplicates("acc_nodash").set_index("acc_nodash")["cik"].to_dict()

    new_tid_cik = {tid: tid_to_cik.get(tid) for tid in new_tids}
    unresolved_cik = [tid for tid, cik in new_tid_cik.items() if not cik]
    if unresolved_cik:
        print(f"WARNING: {len(unresolved_cik)} new transcript(s) have no resolvable CIK, "
              f"will be treated conservatively as NOT excluded by SIC filter (can't check)")

    unique_ciks = sorted({cik for cik in new_tid_cik.values() if cik})
    print(f"\nFetching SIC codes for {len(unique_ciks)} unique companies (new transcripts only)...")

    cik_sic = {}
    for i, cik in enumerate(unique_ciks, 1):
        sic, sic_desc = fetch_sic(cik)
        cik_sic[cik] = (sic, sic_desc)
        excluded_flag = " <-- EXCLUDED" if is_excluded_by_sic(sic, sic_desc) else ""
        print(f"  [{i:3d}/{len(unique_ciks)}] CIK {cik}  SIC={sic}  {sic_desc}{excluded_flag}")

    # Step 1: SIC-based exclusion
    sic_excluded_tids = {
        tid for tid, cik in new_tid_cik.items()
        if cik and is_excluded_by_sic(*cik_sic.get(cik, (None, None)))
    }
    sic_excluded_ciks = {cik for cik in new_tid_cik.values() if cik and is_excluded_by_sic(*cik_sic.get(cik, (None, None)))}

    print("\n" + "=" * 70)
    print("STEP 1: SIC-based exclusion (REIT / blank-check / real estate)")
    print("=" * 70)
    print(f"Transcripts excluded: {len(sic_excluded_tids)}")
    print(f"Companies excluded: {len(sic_excluded_ciks)}")
    for cik in sorted(sic_excluded_ciks):
        sic, desc = cik_sic[cik]
        n = sum(1 for tid, c in new_tid_cik.items() if c == cik)
        print(f"  CIK {cik}  SIC={sic}  {desc}  ({n} transcript(s))")

    remaining_after_sic = new_tids - sic_excluded_tids
    print(f"\nNew transcripts remaining after SIC filter: {len(remaining_after_sic)} / {len(new_tids)}")

    # Step 2: single-transcript-total exclusion.
    # "Total" = across the FULL current dataset (all 333 parsed transcripts), keyed by
    # CIK (not company_ticker, since ~17.5% of new pairs have no resolved ticker).
    print("\n" + "=" * 70)
    print("STEP 2: Single-transcript-company exclusion")
    print("=" * 70)

    all_tid_cik = {}
    for tid in qa["transcript_id"].unique():
        all_tid_cik[tid] = tid_to_cik.get(tid)

    cik_total_counts = pd.Series(list(all_tid_cik.values())).value_counts()

    single_transcript_ciks = set()
    for tid in remaining_after_sic:
        cik = new_tid_cik.get(tid)
        if cik and cik_total_counts.get(cik, 0) == 1:
            single_transcript_ciks.add(cik)

    single_transcript_excluded_tids = {
        tid for tid in remaining_after_sic
        if new_tid_cik.get(tid) in single_transcript_ciks
    }

    print(f"Companies excluded (1 transcript total in whole dataset): {len(single_transcript_ciks)}")
    print(f"Transcripts excluded by this filter: {len(single_transcript_excluded_tids)}")

    final_remaining = remaining_after_sic - single_transcript_excluded_tids
    final_ciks = {new_tid_cik[tid] for tid in final_remaining if new_tid_cik.get(tid)}

    print("\n" + "=" * 70)
    print("FINAL FILTERED COUNTS")
    print("=" * 70)
    print(f"New transcripts before filtering  : {len(new_tids)}")
    print(f"  Excluded (SIC: REIT/SPAC/RE)    : {len(sic_excluded_tids)}")
    print(f"  Excluded (single-transcript)     : {len(single_transcript_excluded_tids)}")
    print(f"New transcripts surviving BOTH filters : {len(final_remaining)}")
    print(f"New companies surviving BOTH filters   : {len(final_ciks)}")

    n_pairs_surviving = new_qa[new_qa["transcript_id"].isin(final_remaining)].shape[0]
    n_pairs_excluded = len(new_qa) - n_pairs_surviving
    print(f"\nNew Q&A pairs surviving (would be scored): {n_pairs_surviving}")
    print(f"New Q&A pairs excluded (would NOT be scored): {n_pairs_excluded}")

    print("\nNo LLM scoring has been run. Confirm these counts before proceeding.")

    return final_remaining, sic_excluded_tids, single_transcript_excluded_tids


if __name__ == "__main__":
    main()
