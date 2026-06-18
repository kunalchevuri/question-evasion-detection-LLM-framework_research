"""
Backfill company_ticker in data/parsed_qa/all_qa_pairs.csv using SEC's
official CIK->ticker mapping (https://www.sec.gov/files/company_tickers.json).

IMPORTANT: the first 10 digits of an accession number are the CIK of whoever
*submitted* the filing on EDGAR, which for many small/mid-cap filers is a
third-party filing agent (e.g. Donnelley/EDGAR Online, CIK 0001564590), not
the company itself. transcript_id[:10] is therefore NOT reliable as the
company CIK. Instead we look up the true company CIK per accession number
from data/filing_index.csv's `cik` column (populated independently from the
EFTS/submissions search results during scraping).

Fallback for CIKs not in the SEC mapping: look up entity_name for that
accession in data/filing_index.csv, normalize it, and match against
normalized SEC company titles.
"""

import re
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
QA_CSV = BASE_DIR / "data" / "parsed_qa" / "all_qa_pairs.csv"
STATS_CSV = BASE_DIR / "data" / "parsing_stats.csv"
FILING_INDEX_CSV = BASE_DIR / "data" / "filing_index.csv"

HEADERS = {"User-Agent": "Kunal Chevuri kunalchevuri510@gmail.com"}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC's company_tickers.json (and company_tickers_exchange.json) is NOT
# comprehensive — verified by direct query that it omits several currently-
# or formerly-listed companies present in our transcripts, including some
# still actively trading today (VOXX, Pacific Premier, Premier Inc, Plymouth
# Industrial REIT, Syros). For those + companies later acquired/delisted, the
# historically-accurate ticker (as of the filing date) was confirmed via
# SEC EDGAR company search and public exchange records.
MANUAL_TICKER_OVERRIDES = {
    "0000807707": "VOXX",   # VOXX International Corp (Nasdaq, still active)
    "0001028918": "PPBI",   # Pacific Premier Bancorp (Nasdaq, still active)
    "0001577916": "PINC",   # Premier, Inc. (Nasdaq, still active)
    "0001515816": "PLYM",   # Plymouth Industrial REIT (NYSE, still active)
    "0001556263": "SYRS",   # Syros Pharmaceuticals (Nasdaq, still active)
    "0000768835": "BIG",    # Big Lots (NYSE; bankrupt/delisted 2024, ticker valid for 2019-2023 filings)
    "0000727207": "AXDX",   # Accelerate Diagnostics (Nasdaq, still active)
    "0001522860": "AFIB",   # Acutus Medical (Nasdaq; went private 2024, valid for filing period)
    "0000892222": "BREW",   # Craft Brew Alliance (Nasdaq; acquired by AB InBev 2020)
    "0001305773": "CFMS",   # ConforMIS (Nasdaq; acquired/delisted 2022)
    "0001419600": "FLXN",   # Flexion Therapeutics (Nasdaq; acquired by Pacira 2021)
    "0001160958": "IPHI",   # Inphi Corp (Nasdaq; acquired by Marvell 2021)
    "0001487052": "LTXB",   # LegacyTexas Financial Group (Nasdaq; acquired by Prosperity Bancshares 2019)
    "0001643988": "LPTV",   # Loop Media (NYSE American, still active)
    "0001760173": "RTIX",   # RTI Surgical Holdings (Nasdaq at filing date; later renamed Surgalign/SGRL in 2020)
    "0000812128": "SAFM",   # Sanderson Farms (Nasdaq; acquired by Cargill/Continental Grain JV 2022)
    "0001499453": "STXB",   # Spirit of Texas Bancshares (Nasdaq; acquired by Simmons First National 2022)
    "0001651235": "ACIA",   # Acacia Communications (Nasdaq; acquired by Cisco 2021)
}

SUFFIX_RE = re.compile(
    r'\b(INC|INCORPORATED|CORP|CORPORATION|LLC|LTD|CO|COMPANY|GROUP|HOLDINGS|'
    r'PLC|LP|TRUST|REIT)\b\.?'
)


def normalize_name(name):
    name = name.upper()
    name = name.replace(",", "").replace(".", "")
    name = SUFFIX_RE.sub("", name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def fetch_sec_ticker_map():
    resp = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Some CIKs have multiple listed securities (e.g. common stock + warrants,
    # like Tidewater: TDW, TDDWW, TDGMW). SEC lists the primary common-stock
    # ticker first for each company, so first-wins avoids picking up a
    # warrant/preferred ticker for the common stock.
    cik_to_ticker = {}
    name_to_ticker = {}
    for entry in data.values():
        cik10 = str(entry["cik_str"]).zfill(10)
        ticker = entry["ticker"]
        if cik10 not in cik_to_ticker:
            cik_to_ticker[cik10] = ticker
        norm = normalize_name(entry["title"])
        if norm and norm not in name_to_ticker:
            name_to_ticker[norm] = ticker
    return cik_to_ticker, name_to_ticker


def build_accession_lookup(filing_index_csv):
    """accession_no (no dashes) -> {cik, entity_name}, sourced from filing_index.csv."""
    if not filing_index_csv.exists():
        return {}
    fi = pd.read_csv(filing_index_csv, dtype=str)
    lookup = {}
    for _, row in fi.iterrows():
        acc = str(row.get("accession_no", "")).replace("-", "")
        if acc and acc not in lookup:
            lookup[acc] = {
                "cik": str(row.get("cik", "")).zfill(10),
                "entity_name": row.get("entity_name", ""),
            }
    return lookup


def backfill(df, cik_to_ticker, name_to_ticker, accession_lookup):
    tickers = []
    unresolved = {}
    matched_via = {"cik": 0, "name_fallback": 0, "manual": 0, "none": 0}

    for tid in df["transcript_id"].astype(str):
        meta = accession_lookup.get(tid)
        cik10 = meta["cik"] if meta else None
        entity_name = meta["entity_name"] if meta else None

        ticker = cik_to_ticker.get(cik10) if cik10 else None

        if ticker:
            matched_via["cik"] += 1
        else:
            ticker = name_to_ticker.get(normalize_name(entity_name)) if entity_name else None
            if ticker:
                matched_via["name_fallback"] += 1
            else:
                ticker = MANUAL_TICKER_OVERRIDES.get(cik10) if cik10 else None
                if ticker:
                    matched_via["manual"] += 1
                else:
                    matched_via["none"] += 1
                    unresolved[tid] = f"{entity_name or '(no filing_index match)'} (cik={cik10})"

        tickers.append(ticker if ticker else pd.NA)

    return tickers, matched_via, unresolved


def main():
    print("Fetching SEC company_tickers.json ...")
    cik_to_ticker, name_to_ticker = fetch_sec_ticker_map()
    print(f"  SEC mapping loaded: {len(cik_to_ticker)} CIKs, {len(name_to_ticker)} normalized names")

    accession_lookup = build_accession_lookup(FILING_INDEX_CSV)
    print(f"  filing_index.csv accession lookup: {len(accession_lookup)} accessions")

    df = pd.read_csv(QA_CSV, dtype=str)
    print(f"\nLoaded {len(df)} rows from {QA_CSV}")

    tickers, matched_via, unresolved = backfill(df, cik_to_ticker, name_to_ticker, accession_lookup)
    df["company_ticker"] = tickers
    df.to_csv(QA_CSV, index=False)

    n_total = len(df)
    n_filled = df["company_ticker"].notna().sum()
    n_nan = n_total - n_filled
    uniq_tickers = sorted(df["company_ticker"].dropna().unique().tolist())

    print(f"\n{'='*60}")
    print(f"Rows with ticker filled : {n_filled} / {n_total}")
    print(f"Rows still NaN          : {n_nan}")
    print(f"  matched via CIK       : {matched_via['cik']}")
    print(f"  matched via name fbk  : {matched_via['name_fallback']}")
    print(f"  matched via manual    : {matched_via['manual']}")
    print(f"  unresolved            : {matched_via['none']}")
    print(f"Unique tickers ({len(uniq_tickers)}): {', '.join(uniq_tickers)}")
    if unresolved:
        print(f"\nUnresolved CIKs ({len(unresolved)}):")
        for cik, name in unresolved.items():
            print(f"  {cik}  {name}")
    print('='*60)

    # parsing_stats.csv: per-transcript pair counts + ticker, derived from the
    # backfilled QA csv (parser.py does not currently emit this file).
    stats = (
        df.groupby("transcript_id")
        .agg(
            company_ticker=("company_ticker", "first"),
            filing_date=("filing_date", "first"),
            n_pairs=("question_text", "count"),
        )
        .reset_index()
        .sort_values("transcript_id")
    )
    stats.to_csv(STATS_CSV, index=False)
    print(f"\nWrote {STATS_CSV} ({len(stats)} transcripts)")


if __name__ == "__main__":
    main()
