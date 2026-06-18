"""
EDGAR earnings call transcript scraper.
Fetches S&P 500 tech/healthcare companies, resolves CIKs, queries EFTS for
8-K filings, and downloads raw transcript exhibits.
"""

import csv
import io
import logging
import os
import time

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper_errors.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Kunal Chevuri kunalchevuri510@gmail.com"}
SLEEP = 0.5

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_transcripts")
os.makedirs(RAW_DIR, exist_ok=True)


def sleep():
    time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# Step 1: S&P 500 tech + healthcare companies from Wikipedia
# ---------------------------------------------------------------------------

def get_sp500_tech_healthcare() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print("Fetching S&P 500 list from Wikipedia...")
    try:
        # Wikipedia blocks Python's urllib; fetch with requests then parse HTML
        wiki_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=wiki_headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as exc:
        log.error("Failed to fetch S&P 500 Wikipedia table: %s", exc)
        raise

    sp500 = tables[0]
    # Normalise column names – Wikipedia occasionally changes them
    sp500.columns = [c.strip() for c in sp500.columns]

    sector_col = next(
        (c for c in sp500.columns if "sector" in c.lower()), None
    )
    ticker_col = next(
        (c for c in sp500.columns if "symbol" in c.lower() or "ticker" in c.lower()),
        None,
    )
    name_col = next(
        (c for c in sp500.columns if "security" in c.lower() or "name" in c.lower()),
        None,
    )

    if not sector_col:
        raise ValueError(f"Cannot find sector column. Columns: {list(sp500.columns)}")

    target_sectors = {"Information Technology", "Health Care"}
    mask = sp500[sector_col].isin(target_sectors)
    filtered = sp500[mask].copy()

    keep = {}
    if ticker_col:
        keep["ticker"] = ticker_col
    if name_col:
        keep["name"] = name_col
    keep["sector"] = sector_col

    result = filtered.rename(columns={v: k for k, v in keep.items()})[list(keep.keys())]
    print(f"  Found {len(result)} companies (IT + Health Care)")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Resolve CIK numbers from SEC
# ---------------------------------------------------------------------------

def get_cik_map() -> dict:
    url = "https://www.sec.gov/files/company_tickers.json"
    print("Fetching CIK map from SEC...")
    sleep()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        log.error("Failed to fetch CIK map: %s", exc)
        raise

    raw = resp.json()
    # Keys are integers; values have 'cik_str', 'ticker', 'title'
    ticker_to_cik = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in raw.values()
    }
    print(f"  CIK map loaded: {len(ticker_to_cik)} tickers")
    return ticker_to_cik


def build_cik_csv(companies: pd.DataFrame, ticker_to_cik: dict) -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, "sp500_tech_healthcare_ciks.csv")
    rows = []
    missing = []
    for _, row in companies.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik = ticker_to_cik.get(ticker)
        if cik:
            rows.append(
                {
                    "ticker": ticker,
                    "name": row.get("name", ""),
                    "sector": row.get("sector", ""),
                    "cik": cik,
                }
            )
        else:
            missing.append(ticker)

    if missing:
        log.warning("No CIK found for %d tickers: %s", len(missing), missing)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved {len(df)} rows -> {out_path}")
    if missing:
        print(f"  Warning: {len(missing)} tickers had no CIK match (see log)")
    return df


# ---------------------------------------------------------------------------
# Step 3: Query EDGAR EFTS for 8-K earnings call filings 2019-2023
#
# Strategy: The EFTS full-text index only indexes a small subset of 8-K
# exhibit content. We use it per-company (entity filter) first; if a company
# returns nothing, we fall back to the submissions API which is exhaustive.
# ---------------------------------------------------------------------------

def _fmt_accession(raw: str) -> str:
    """Normalise an accession number to the dashed form XXXXXXXXXX-YY-ZZZZZZ."""
    clean = raw.replace("-", "")
    if len(clean) == 18:
        return f"{clean[:10]}-{clean[10:12]}-{clean[12:]}"
    return raw  # already formatted or unexpected


def _fetch_submissions_page(url: str) -> dict:
    sleep()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.error("Submissions request failed %s: %s", url, exc)
        return {}


def _submissions_8k(cik: str, entity_name: str) -> list:
    """Return list of 8-K filing dicts for a single company via submissions API."""
    results = []
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data = _fetch_submissions_page(url)
    if not data:
        return results

    entity_name = data.get("name", entity_name)

    def _collect(recent: dict) -> None:
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        reports = recent.get("reportDate", [])
        for form, date, acc, rep in zip(forms, dates, accessions, reports):
            if form == "8-K" and "2019" <= date[:4] <= "2023":
                results.append(
                    {
                        "accession_no": _fmt_accession(acc),
                        "entity_name": entity_name,
                        "cik": cik_padded,
                        "file_date": date,
                        "period_of_report": rep,
                        "form_type": form,
                    }
                )

    _collect(data.get("filings", {}).get("recent", {}))

    # Paginate older filings if the most recent page doesn't reach 2019
    for extra in data.get("filings", {}).get("files", []):
        extra_url = f"https://data.sec.gov/submissions/{extra['name']}"
        extra_data = _fetch_submissions_page(extra_url)
        if extra_data:
            _collect(extra_data.get("filings", {}).get("recent", extra_data))

    return results



def query_efts(cik_df: pd.DataFrame) -> list:
    """
    Two-phase collection:
      A) Global EFTS phrase search – finds all 8-K exhibits that literally contain
         'CORPORATE PARTICIPANTS' and 'QUESTIONS AND ANSWERS' (verified transcripts).
      B) Submissions API per S&P 500 company – returns every 8-K 2019-2023; step 4
         content-filters these for the required phrases.
    EFTS results are prepended so step 4 tries them first.
    """
    all_hits = []
    seen_acc: set = set()

    # -- Part A: EFTS phrase searches – one per query, combined and deduplicated --
    print("Part A: EFTS phrase searches (5 queries, verified transcript exhibits)...")
    efts_url = "https://efts.sec.gov/LATEST/search-index"
    efts_params_base = {
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": "2019-01-01",
        "enddt": "2023-12-31",
    }

    for qi, query in enumerate(EFTS_QUERIES, 1):
        print(f"  Query {qi}/{len(EFTS_QUERIES)}: {query}")
        offset = 0
        query_count = 0
        while True:
            sleep()
            params = {**efts_params_base, "q": query, "from": offset}
            try:
                resp = requests.get(efts_url, params=params, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.error("EFTS search failed query=%r offset=%d: %s", query, offset, exc)
                break

            hits_page = data.get("hits", {}).get("hits", [])
            if not hits_page:
                break

            for hit in hits_page:
                src = hit.get("_source", {})
                cik_list = src.get("ciks", [])
                adsh = src.get("adsh", "")
                if not adsh or not cik_list:
                    continue
                names = src.get("display_names", [])
                entity_name = names[0].split(" (")[0].strip() if names else ""
                rec = {
                    "accession_no": adsh,
                    "entity_name": entity_name,
                    "cik": cik_list[0],
                    "file_date": src.get("file_date", ""),
                    "period_of_report": src.get("period_ending", ""),
                    "form_type": (src.get("root_forms") or ["8-K"])[0],
                    "file_type_hint": src.get("file_type", ""),
                    "source": "EFTS",
                }
                if adsh not in seen_acc:
                    seen_acc.add(adsh)
                    all_hits.append(rec)
                    query_count += 1

            total_for_query = (
                data.get("hits", {}).get("total", {}).get("value", 0)
                if isinstance(data.get("hits", {}).get("total"), dict)
                else data.get("hits", {}).get("total", 0)
            )
            offset += len(hits_page)
            if offset >= total_for_query or offset >= 10000:
                break

        print(f"    -> {query_count} new unique filings (running total: {len(all_hits)})")

    print(f"  Part A done: {len(all_hits)} phrase-verified filing exhibits")

    # -- Part B: submissions API for each S&P 500 tech/healthcare company --
    print("Part B: Submissions API for S&P 500 tech/healthcare 8-Ks (2019-2023)...")
    before_b = len(all_hits)
    for i, row in enumerate(cik_df.itertuples(), 1):
        cik = str(row.cik)
        name = str(row.name)
        subs = _submissions_8k(cik, name)
        added = 0
        for h in subs:
            h["file_type_hint"] = ""
            h["source"] = "submissions"
            if h["accession_no"] not in seen_acc:
                seen_acc.add(h["accession_no"])
                all_hits.append(h)
                added += 1
        print(
            f"  [{i:3d}/{len(cik_df)}] {name[:40]:<40} | +{added:3d} 8-Ks | total: {len(all_hits)}"
        )

    print(f"  Part B done: added {len(all_hits) - before_b} S&P 500 8-K filings")
    return all_hits


def save_filing_index(hits: list) -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, "filing_index.csv")
    df = pd.DataFrame(hits)
    df.drop_duplicates(subset=["accession_no"], inplace=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved {len(df)} unique filings -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# Step 4: Download raw transcript exhibit files
# ---------------------------------------------------------------------------

# All EFTS phrase queries run separately; results are combined and deduplicated.
EFTS_QUERIES = [
    '"CORPORATE PARTICIPANTS" "QUESTIONS AND ANSWERS"',
    '"QUESTION AND ANSWER SESSION"',
    '"QUESTIONS & ANSWERS"',
    '"Q&A SESSION"',
    '"CONFERENCE CALL PARTICIPANTS"',
]

# A valid transcript must contain at least one phrase from each group.
PARTICIPANT_PHRASES = ["CORPORATE PARTICIPANTS", "CONFERENCE CALL PARTICIPANTS"]
QA_PHRASES = [
    "QUESTIONS AND ANSWERS",
    "QUESTION AND ANSWER SESSION",
    "QUESTIONS & ANSWERS",
    "Q&A SESSION",
]

MIN_WORDS = 1500


def get_filing_documents(accession_no: str, cik: str, file_type_hint: str = "") -> list:
    """
    Parse the filing's -index.htm to find EX-99 exhibit URLs.
    When file_type_hint is set (e.g. 'EX-99.2'), that exhibit is returned first.
    """
    clean_acc = accession_no.replace("-", "")
    cik_num = cik.lstrip("0") or "0"
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{clean_acc}/"
    index_url = f"{base}{accession_no}-index.htm"

    sleep()
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        log.error("Failed to fetch filing index %s: %s", accession_no, exc)
        return []

    try:
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as exc:
        log.error("Failed to parse filing index HTML %s: %s", accession_no, exc)
        return []

    prioritised = []
    others = []
    hint = file_type_hint.upper().strip()

    for table in tables:
        cols_lower = {str(c).lower(): c for c in table.columns}
        if "document" not in cols_lower or "type" not in cols_lower:
            continue
        doc_col = cols_lower["document"]
        type_col = cols_lower["type"]
        for _, row in table.iterrows():
            doc_type = str(row[type_col]).upper().strip()
            doc_name = str(row[doc_col]).strip()
            if doc_name in ("nan", "") or doc_type == "nan":
                continue
            if doc_type.startswith("EX-99"):
                url = base + doc_name
                if hint and doc_type == hint:
                    prioritised.append(url)
                else:
                    others.append(url)
        break  # only need the first document table

    return prioritised + others


def download_transcript(url: str, dest_path: str) -> bool:
    """Download a document, validate it, and save if it passes checks."""
    sleep()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:
        log.error("Download failed %s: %s", url, exc)
        return False

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        log.info("Skipping %s: only %d words (< %d)", url, word_count, MIN_WORDS)
        return False

    text_upper = text.upper()
    if not any(p in text_upper for p in PARTICIPANT_PHRASES):
        log.info("Skipping %s: no participant header phrase found", url)
        return False
    if not any(p in text_upper for p in QA_PHRASES):
        log.info("Skipping %s: no Q&A section phrase found", url)
        return False

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def download_transcripts(filing_df: pd.DataFrame) -> None:
    downloaded = 0
    failed = 0
    skipped = 0
    error_log = []
    total_filings = len(filing_df)

    print(f"\nDownloading all valid transcript exhibits ({total_filings} filings to check)...")

    for _, row in filing_df.iterrows():
        accession_no = row["accession_no"]
        cik = str(row["cik"]).zfill(10)
        entity_name = row.get("entity_name", "unknown")
        file_type_hint = row.get("file_type_hint", "")

        docs = get_filing_documents(accession_no, cik, file_type_hint)
        if not docs:
            log.warning("No documents found for %s (%s)", accession_no, entity_name)
            skipped += 1
            continue

        found_for_filing = False
        for doc_url in docs:
            safe_name = accession_no.replace("-", "_")
            fname = f"{safe_name}_{os.path.basename(doc_url)}"
            dest = os.path.join(RAW_DIR, fname)

            if os.path.exists(dest):
                print(f"  Already exists: {fname}")
                downloaded += 1
                found_for_filing = True
                break

            print(f"  [saved={downloaded}] {entity_name} | {doc_url}")
            success = download_transcript(doc_url, dest)
            if success:
                downloaded += 1
                found_for_filing = True
                print(f"    Saved -> {fname}")
                break
            else:
                failed += 1
                error_log.append({"url": doc_url, "accession": accession_no})

        if not found_for_filing:
            skipped += 1

    print(f"\nDownload complete: {downloaded} saved, {failed} failed, {skipped} skipped")

    if error_log:
        err_path = os.path.join(DATA_DIR, "download_errors.csv")
        with open(err_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "accession"])
            writer.writeheader()
            writer.writerows(error_log)
        print(f"  Error log -> {err_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EDGAR Earnings Call Transcript Scraper")
    print("=" * 60)

    # Step 1
    companies = get_sp500_tech_healthcare()

    # Step 2
    ticker_to_cik = get_cik_map()
    cik_df = build_cik_csv(companies, ticker_to_cik)

    # Step 3
    hits = query_efts(cik_df)
    filing_df = save_filing_index(hits)

    # Step 4
    download_transcripts(filing_df)

    print("\nAll done.")


if __name__ == "__main__":
    main()
