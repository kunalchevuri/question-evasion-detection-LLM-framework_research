"""
Re-derive the SIC/single-transcript filter (src/filter_before_scoring.py),
prune the 488 excluded new pairs out of all_qa_pairs.csv entirely, and print
the confirmation numbers requested before Day 3 scoring proceeds.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_before_scoring import main as run_filter, QA_CSV  # noqa: E402


def main():
    final_remaining, sic_excluded, single_excluded = run_filter()

    print("\n" + "=" * 70)
    print("PRUNING all_qa_pairs.csv")
    print("=" * 70)

    qa = pd.read_csv(QA_CSV, dtype=str)
    orig_qa = qa.iloc[:1594]
    new_qa = qa.iloc[1594:]

    excluded_tids = sic_excluded | single_excluded
    pruned_new_qa = new_qa[~new_qa["transcript_id"].isin(excluded_tids)]

    print(f"New pairs before pruning: {len(new_qa)}")
    print(f"New pairs removed (excluded transcripts): {len(new_qa) - len(pruned_new_qa)}")
    print(f"New pairs remaining: {len(pruned_new_qa)}")

    combined = pd.concat([orig_qa, pruned_new_qa], ignore_index=True)
    combined.to_csv(QA_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("CONFIRMATION NUMBERS")
    print("=" * 70)

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw_transcripts"
    n_raw_total = len(list(raw_dir.glob("*.htm")))

    orig_tids = set(orig_qa["transcript_id"].unique())
    new_tids_final = set(pruned_new_qa["transcript_id"].unique())

    fi = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "filing_index.csv", dtype=str)
    fi["acc_nodash"] = fi["accession_no"].str.replace("-", "", regex=False)
    tid_to_cik = fi.drop_duplicates("acc_nodash").set_index("acc_nodash")["cik"].to_dict()

    all_final_tids = orig_tids | new_tids_final
    all_final_ciks = {tid_to_cik.get(t) for t in all_final_tids if tid_to_cik.get(t)}

    print(f"1. Raw .htm files in data/raw_transcripts/: {n_raw_total} (216 original + 307 new, unpruned -- "
          f"raw files are not deleted by this filter, only which pairs get scored)")
    print(f"   Unique transcripts that will be PARSED AND SCORED: {len(all_final_tids)} "
          f"({len(orig_tids)} original + {len(new_tids_final)} new-filtered)")

    print(f"\n2. Final total unique companies across whole dataset: {len(all_final_ciks)}")

    print(f"\n3. Total NEW Q&A pairs added by this filtered batch: {len(pruned_new_qa)}")

    print(f"\n4. Total COMBINED Q&A pairs (original + new filtered): {len(combined)} "
          f"({len(orig_qa)} + {len(pruned_new_qa)})")

    dup_tids = orig_tids & new_tids_final
    print(f"\n5. Duplicate transcript_ids between old and new data: {len(dup_tids)}")

    dup_rows = combined.duplicated(subset=["transcript_id", "question_text", "response_text"]).sum()
    print(f"   Duplicate (transcript_id, question, response) rows in combined file: {dup_rows}")

    print(f"\nSaved pruned all_qa_pairs.csv -> {QA_CSV}")


if __name__ == "__main__":
    main()
