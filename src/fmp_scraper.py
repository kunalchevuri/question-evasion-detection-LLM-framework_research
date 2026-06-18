"""
FMP earnings call transcript scraper.
Downloads transcripts via Financial Modeling Prep stable endpoint and saves
them as plain-text files under data/raw_transcripts/.
"""

import csv
import os
import time

import requests

API_KEY = os.environ.get("FMP_API_KEY")
if not API_KEY:
    raise SystemExit("FMP_API_KEY environment variable is not set.")

BASE_URL = "https://financialmodelingprep.com/stable/earning_call_transcript"
SLEEP = 0.5
MIN_WORDS = 1500
MAX_SUCCESS = 240

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_transcripts")
os.makedirs(RAW_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(DATA_DIR, "fmp_manifest.csv")

TICKERS = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC", "QCOM",
    "CSCO", "IBM",  "ORCL", "CRM",  "ADBE", "NOW",  "INTU", "AMAT", "LRCX", "KLAC",
    "MU",   "TXN",  "AVGO", "SNPS", "CDNS", "PANW", "CRWD", "NET",  "SNOW", "DDOG",
    # Health Care
    "JNJ",  "PFE",  "MRK",  "ABBV", "LLY",  "UNH",  "CVS",  "AMGN", "GILD", "BIIB",
    "TMO",  "ABT",  "MDT",  "BMY",  "ISRG", "BSX",  "ELV",  "HUM",  "CI",   "REGN",
]

YEARS = range(2019, 2024)
QUARTERS = range(1, 5)


def fetch_transcript(symbol: str, year: int, quarter: int) -> str | None:
    """Return transcript content string, or None on failure/empty."""
    params = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "apikey": API_KEY,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    ERROR fetching {symbol} {year} Q{quarter}: {exc}")
        return None

    if not data or not isinstance(data, list):
        return None

    content = data[0].get("content", "")
    if not content:
        return None
    return content


def main() -> None:
    print("=" * 60)
    print("FMP Earnings Call Transcript Scraper")
    print("=" * 60)

    manifest_rows: list[dict] = []
    success_count = 0
    skip_existing = 0
    skip_empty = 0
    skip_short = 0
    errors = 0

    # Load any prior manifest so we can append without re-downloading
    existing_manifest: set[str] = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_manifest.add(row.get("file", ""))

    total_combos = len(TICKERS) * len(YEARS) * len(QUARTERS)
    combo_idx = 0

    outer_done = False
    for symbol in TICKERS:
        if outer_done:
            break
        for year in YEARS:
            if outer_done:
                break
            for quarter in QUARTERS:
                if outer_done:
                    break

                combo_idx += 1
                fname = f"fmp_{symbol}_{year}_Q{quarter}.txt"
                dest = os.path.join(RAW_DIR, fname)

                # Skip already-downloaded files
                if os.path.exists(dest):
                    skip_existing += 1
                    success_count += 1  # counts toward the cap
                    print(
                        f"  [{combo_idx:5d}/{total_combos}] {symbol} {year} Q{quarter}"
                        f" | EXISTS | success={success_count}"
                    )
                    if success_count >= MAX_SUCCESS:
                        outer_done = True
                    continue

                time.sleep(SLEEP)
                content = fetch_transcript(symbol, year, quarter)

                if content is None:
                    skip_empty += 1
                    errors += 1
                    print(
                        f"  [{combo_idx:5d}/{total_combos}] {symbol} {year} Q{quarter}"
                        f" | EMPTY/ERROR"
                    )
                    continue

                word_count = len(content.split())
                if word_count < MIN_WORDS:
                    skip_short += 1
                    print(
                        f"  [{combo_idx:5d}/{total_combos}] {symbol} {year} Q{quarter}"
                        f" | SHORT ({word_count} words)"
                    )
                    continue

                with open(dest, "w", encoding="utf-8") as f:
                    f.write(content)

                success_count += 1
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "year": year,
                        "quarter": quarter,
                        "file": fname,
                        "word_count": word_count,
                    }
                )

                print(
                    f"  [{combo_idx:5d}/{total_combos}] {symbol} {year} Q{quarter}"
                    f" | SAVED {word_count:,} words | success={success_count}"
                )

                if success_count >= MAX_SUCCESS:
                    print(f"\nReached {MAX_SUCCESS} successful downloads. Stopping.")
                    outer_done = True

    # Write / append manifest
    write_header = not os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "year", "quarter", "file", "word_count"])
        if write_header:
            writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 60)
    print(f"Done. Results:")
    print(f"  Saved (new)     : {len(manifest_rows)}")
    print(f"  Already existed : {skip_existing}")
    print(f"  Empty / error   : {skip_empty}")
    print(f"  Too short       : {skip_short}")
    print(f"  Total success   : {success_count}")
    print(f"  Manifest        : {MANIFEST_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
