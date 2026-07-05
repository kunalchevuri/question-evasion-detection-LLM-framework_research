"""
Parse ONLY the 307 newly-scraped transcripts (the dataset expansion pass) and
append their Q&A pairs to the existing data/parsed_qa/all_qa_pairs.csv, which
already holds 1594 pairs from the original 216 transcripts.

Reuses the exact parsing logic from parser.py (html_to_lines,
extract_participants, extract_turns, build_pairs) and the exact ticker
resolution logic from backfill_tickers.py (fetch_sec_ticker_map,
build_accession_lookup, backfill) -- nothing is reimplemented.

The 307 new files are identified by file modification time (all scraped on
2026-07-04, vs. the original 216 from 2026-06-15/16) rather than by
"transcript_id not already in all_qa_pairs.csv", since only 147 of the
original 216 transcripts actually produced pairs -- the other 69 were
correctly excluded for failing the MIN_PAIRS/word-count thresholds during
the original Day 2 run, and must NOT be re-parsed here per instructions.
"""

import csv
import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser import (  # noqa: E402
    RAW_DIR, OUT_CSV, MIN_PAIRS, CSV_FIELDS,
    build_meta_lookup, accession_from_fname,
    html_to_lines, extract_participants, extract_turns, build_pairs,
)
from backfill_tickers import (  # noqa: E402
    FILING_INDEX_CSV, fetch_sec_ticker_map, build_accession_lookup, backfill,
)

BASE_DIR = Path(__file__).resolve().parent.parent
NEW_FILES_CUTOFF = datetime.datetime(2026, 6, 20)  # original 216 are 06-15/16; new 307 are 07-04


def get_new_files():
    all_files = sorted(RAW_DIR.glob("*.htm"))
    new_files = [
        f for f in all_files
        if datetime.datetime.fromtimestamp(f.stat().st_mtime) > NEW_FILES_CUTOFF
    ]
    old_files = [f for f in all_files if f not in new_files]
    print(f"Raw transcript files on disk: {len(all_files)}")
    print(f"  Original (by mtime, unchanged): {len(old_files)}")
    print(f"  New (scraped 2026-07-04, to be parsed now): {len(new_files)}")
    return new_files


def parse_files(htm_files, meta):
    all_pairs = []
    ok = skip = 0

    for idx, fpath in enumerate(htm_files, 1):
        try:
            acc = accession_from_fname(fpath.name)
            m = meta.get(acc, {})
            tid = acc or fpath.stem
            fdate = m.get("file_date", "")

            with open(fpath, encoding="utf-8", errors="replace") as f:
                html = f.read()

            lines = html_to_lines(html)
            full_text = "\n".join(lines)

            mgmt_names, analyst_names = extract_participants(full_text)
            turns = extract_turns(lines, mgmt_names, analyst_names)
            pairs = build_pairs(turns)
            n = len(pairs)

            if n < MIN_PAIRS:
                print(f"  [{idx:3d}/{len(htm_files)}] SKIP  {fpath.name[:55]:<55}  {n} pairs")
                skip += 1
                continue

            for p in pairs:
                p.update(transcript_id=tid, company_ticker="", filing_date=fdate)
            all_pairs.extend(pairs)
            ok += 1
            print(f"  [{idx:3d}/{len(htm_files)}] OK    {fpath.name[:55]:<55}  {n} pairs")

        except Exception as exc:
            print(f"  [{idx:3d}/{len(htm_files)}] ERROR {fpath.name}: {exc}")
            skip += 1

    return all_pairs, ok, skip


def backfill_new_tickers(new_df):
    print("\nBackfilling company_ticker for new rows (backfill_tickers.py logic)...")
    cik_to_ticker, name_to_ticker = fetch_sec_ticker_map()
    accession_lookup = build_accession_lookup(FILING_INDEX_CSV)

    tickers, matched_via, unresolved = backfill(new_df, cik_to_ticker, name_to_ticker, accession_lookup)
    new_df = new_df.copy()
    new_df["company_ticker"] = tickers

    n_filled = new_df["company_ticker"].notna().sum()
    print(f"  Tickers filled: {n_filled} / {len(new_df)}")
    print(f"    matched via CIK      : {matched_via['cik']}")
    print(f"    matched via name fbk : {matched_via['name_fallback']}")
    print(f"    matched via manual   : {matched_via['manual']}")
    print(f"    unresolved           : {matched_via['none']}")
    if unresolved:
        print(f"  Unresolved tickers ({len(unresolved)} transcripts, showing up to 15):")
        for tid, name in list(unresolved.items())[:15]:
            print(f"    {tid}  {name}")

    return new_df


def main():
    new_files = get_new_files()

    meta = build_meta_lookup()
    print(f"Metadata loaded: {len(meta)} filing index entries\n")

    print(f"Parsing {len(new_files)} new transcripts...")
    all_pairs, ok, skip = parse_files(new_files, meta)

    print(f"\nNew transcripts parsed OK : {ok}")
    print(f"New transcripts skipped   : {skip}")
    print(f"New Q&A pairs extracted   : {len(all_pairs)}")

    if not all_pairs:
        print("\nNo new pairs extracted -- nothing to append.")
        return

    new_df = pd.DataFrame(all_pairs)[CSV_FIELDS]
    new_df = backfill_new_tickers(new_df)

    existing_df = pd.read_csv(OUT_CSV, dtype=str)
    print(f"\nExisting all_qa_pairs.csv: {len(existing_df)} pairs, "
          f"{existing_df['transcript_id'].nunique()} unique transcripts")

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    dup_check = combined_df.duplicated(
        subset=["transcript_id", "question_text", "response_text"]
    ).sum()
    if dup_check:
        print(f"WARNING: {dup_check} duplicate (transcript_id, question, response) rows detected after concat")

    combined_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"New transcripts successfully parsed : {ok}")
    print(f"New transcripts skipped              : {skip}")
    print(f"New Q&A pairs extracted               : {len(new_df)}")
    print(f"Previous total pairs                  : {len(existing_df)}")
    print(f"New combined total pairs              : {len(combined_df)}")
    print(f"New combined unique transcripts        : {combined_df['transcript_id'].nunique()}")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
