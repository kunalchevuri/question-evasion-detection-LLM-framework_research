"""
EDGAR earnings call transcript scraper.
Resolves CIKs, queries EFTS + submissions API for 8-K filings (2015-2023),
and downloads raw transcript exhibits.

Part B originally restricted to S&P 500 tech/healthcare companies; that
restriction has been removed in favor of the 43 tickers already validated in
data/parsed_qa/evasion_scores.csv plus whatever companies the EFTS phrase
searches in Part A turn up. get_sp500_tech_healthcare()/build_cik_csv() are
kept for reference but are no longer called from main().
"""

import csv
import io
import logging
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from backfill_tickers import MANUAL_TICKER_OVERRIDES  # noqa: E402

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
# Step 3: Query EDGAR EFTS for 8-K earnings call filings 2015-2023
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
            if form == "8-K" and "2015" <= date[:4] <= "2023":
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



def build_expansion_cik_df(ticker_to_cik: dict) -> pd.DataFrame:
    """
    Part B company universe for the dataset-expansion pass: limited to the 43
    tickers already in data/parsed_qa/evasion_scores.csv -- companies we already
    know produce clean, parseable transcripts -- to backfill additional history
    (2015-2018, plus anything the wider EFTS queries newly surfaced for them).

    Earlier attempts also fed every company *discovered* via Part A's phrase
    searches into Part B's exhaustive "pull every 8-K this company has ever
    filed" step. That does not scale: even restricted to the original 5
    (narrower, transcript-specific) queries, this surfaced 1,654 additional
    companies -- mostly legitimate operating companies, not noise -- but each
    contributes on average ~89 mostly-irrelevant 8-Ks (executive changes,
    routine agreements, etc.), producing 150,000+ candidate filings and an
    estimated 40+ hour Step 4 (download+validate) runtime, with a very low
    validation hit-rate since most of any company's 8-Ks aren't earnings
    calls at all.

    New companies still come through -- just via Part A's own direct,
    phrase-verified hits (already phrase-verified copies of the actual
    transcript exhibit, so Step 4 validation hit-rate is much higher, and the
    candidate volume is bounded by what genuinely matched a transcript
    phrase, not by every 8-K a company has ever filed).
    """
    known_path = os.path.join(DATA_DIR, "parsed_qa", "evasion_scores.csv")
    known_df = pd.read_csv(known_path, dtype=str)
    known_tickers = sorted(known_df["company_ticker"].dropna().unique())

    rows = []
    seen_ciks: set = set()

    for ticker in known_tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if cik and cik not in seen_ciks:
            seen_ciks.add(cik)
            rows.append({"cik": cik, "name": ticker, "ticker": ticker, "known": True})
    n_known_resolved = len(rows)
    if n_known_resolved < len(known_tickers):
        log.warning(
            "Only resolved CIK for %d/%d known tickers via company_tickers.json",
            n_known_resolved, len(known_tickers),
        )

    df = pd.DataFrame(rows)
    print(
        f"  Part B company universe: {len(df)} companies "
        f"(the 43 known/validated tickers only -- new companies come through "
        f"Part A's direct phrase-verified hits instead of an exhaustive per-company pull)"
    )
    return df


def query_efts(ticker_to_cik: dict, seen_acc: set) -> list:
    """
    Two-phase collection:
      A) Global EFTS phrase search – finds all 8-K exhibits that literally contain
         one of the EFTS_QUERIES phrases (verified transcripts), 2015-2023. New
         companies (beyond the 43 known tickers) come through here.
      B) Submissions API for the 43 known tickers only – returns every 8-K
         2015-2023 for companies already validated to produce clean transcripts;
         step 4 content-filters these for the required phrases. See
         build_expansion_cik_df for why this isn't extended to Part A-discovered
         companies too.
    EFTS results are prepended so step 4 tries them first.

    seen_acc is pre-seeded by the caller with accession numbers already present
    in data/raw_transcripts/, so filings we've already downloaded are never
    re-added to the hit list in the first place.
    """
    all_hits = []

    # -- Part A: EFTS phrase searches – one per query, combined and deduplicated --
    print(f"Part A: EFTS phrase searches ({len(EFTS_QUERIES)} queries, verified transcript exhibits)...")
    efts_url = "https://efts.sec.gov/LATEST/search-index"
    efts_params_base = {
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": "2015-01-01",
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
                cik = str(cik_list[0]).zfill(10)

                if adsh in seen_acc:
                    continue

                names = src.get("display_names", [])
                entity_name = names[0].split(" (")[0].strip() if names else ""
                rec = {
                    "accession_no": adsh,
                    "entity_name": entity_name,
                    "cik": cik,
                    "file_date": src.get("file_date", ""),
                    "period_of_report": src.get("period_ending", ""),
                    "form_type": (src.get("root_forms") or ["8-K"])[0],
                    "file_type_hint": src.get("file_type", ""),
                    "source": "EFTS",
                }
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

    print(f"  Part A done: {len(all_hits)} new phrase-verified filing exhibits")

    # -- Part B: submissions API for the 43 known tickers only (see build_expansion_cik_df) --
    cik_df = build_expansion_cik_df(ticker_to_cik)
    print("Part B: Submissions API for the 43 known tickers (8-Ks, 2015-2023)...")
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
            f"  [{i:3d}/{len(cik_df)}] {name[:40]:<40} | +{added:3d} 8-Ks | total new: {len(all_hits)}"
        )

    print(f"  Part B done: added {len(all_hits) - before_b} 8-K filings")
    return all_hits


def save_filing_index(hits: list) -> pd.DataFrame:
    """
    Merges new hits into the existing filing_index.csv rather than overwriting
    it -- downstream scripts (parser.py, backfill_tickers.py, features.py) rely
    on the metadata already recorded there for the original 216 transcripts.
    """
    out_path = os.path.join(DATA_DIR, "filing_index.csv")
    new_df = pd.DataFrame(hits)

    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path, dtype=str)
        n_existing = len(existing_df)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        n_existing = 0
        combined = new_df

    combined.drop_duplicates(subset=["accession_no"], inplace=True)
    combined.to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"  filing_index.csv: {n_existing} existing rows + {len(new_df)} new candidate "
        f"rows -> {len(combined)} unique rows saved -> {out_path}"
    )
    return combined


def get_existing_accessions() -> set:
    """
    Accession numbers (dashed form, e.g. '0001234567-19-000123') for filings we
    already have a saved raw transcript for. Used to pre-seed seen_acc so Part
    A/B never re-add them to the hit list, and download_transcripts never
    re-fetches them -- satisfies "do not re-download the existing 216 files."
    """
    existing = set()
    for fname in os.listdir(RAW_DIR):
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        cik10, yr, seq = parts[0], parts[1], parts[2]
        if (len(cik10) == 10 and cik10.isdigit()
                and len(yr) == 2 and yr.isdigit()
                and len(seq) == 6 and seq.isdigit()):
            existing.add(f"{cik10}-{yr}-{seq}")
    return existing


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
    # Added for the dataset expansion pass (2015-2023 window): these catch
    # transcript format variations the original 5 queries miss.
    '"PRESENTATION" "QUESTIONS AND ANSWERS"',
    '"OPERATOR" "QUESTION-AND-ANSWER"',
    '"EARNINGS CALL TRANSCRIPT"',
    '"CONFERENCE CALL TRANSCRIPT"',
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


def download_transcripts(filing_df: pd.DataFrame, known_ciks: set) -> dict:
    """
    filing_df is expected to already exclude accessions matching the 216
    existing raw transcripts (seen_acc was pre-seeded before Part A/B ran), so
    every successful save here is a genuinely new transcript. The 'Already
    exists' branch is kept only as a safety net for the rare case where two
    filing_df rows resolve to the same destination filename.

    known_ciks: CIKs of the 43 already-validated tickers, used to classify
    each new download as an additional quarter from an existing company vs.
    a transcript from a brand-new company.
    """
    downloaded = 0
    failed = 0
    skipped = 0
    error_log = []
    new_company_ciks: set = set()
    existing_company_new_quarters = 0
    total_filings = len(filing_df)

    print(f"\nDownloading new transcript exhibits ({total_filings} candidate filings to check)...")

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
                print(f"  Already exists (unexpected at this stage): {fname}")
                found_for_filing = True
                break

            print(f"  [new saved={downloaded}] {entity_name} | {doc_url}")
            success = download_transcript(doc_url, dest)
            if success:
                downloaded += 1
                found_for_filing = True
                print(f"    Saved -> {fname}")
                if cik in known_ciks:
                    existing_company_new_quarters += 1
                else:
                    new_company_ciks.add(cik)
                break
            else:
                failed += 1
                error_log.append({"url": doc_url, "accession": accession_no})

        if not found_for_filing:
            skipped += 1

    print(f"\nDownload complete: {downloaded} new saved, {failed} failed validation, {skipped} skipped (no usable exhibit)")

    if error_log:
        err_path = os.path.join(DATA_DIR, "download_errors.csv")
        with open(err_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "accession"])
            writer.writeheader()
            writer.writerows(error_log)
        print(f"  Error log -> {err_path}")

    return {
        "downloaded": downloaded,
        "failed": failed,
        "skipped": skipped,
        "new_company_ciks": new_company_ciks,
        "existing_company_new_quarters": existing_company_new_quarters,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EDGAR Earnings Call Transcript Scraper -- dataset expansion pass")
    print("2015-2023, 9 EFTS phrase queries, 43 known tickers + EFTS-discovered")
    print("companies (S&P 500 tech/healthcare restriction removed from Part B)")
    print("=" * 60)

    existing_accessions = get_existing_accessions()
    n_existing_files = len(os.listdir(RAW_DIR))
    print(f"\nExisting raw transcripts on disk: {n_existing_files}")
    print(f"Existing accession numbers recognized for dedup: {len(existing_accessions)}")

    ticker_to_cik = get_cik_map()

    # SEC's company_tickers.json only covers currently-listed tickers; several
    # of the 43 known tickers are delisted/acquired/renamed (same issue
    # backfill_tickers.py hit) and need the verified manual CIK overrides,
    # merged in without clobbering the authoritative SEC mapping.
    ticker_to_cik_override = {v: k for k, v in MANUAL_TICKER_OVERRIDES.items()}
    n_override_added = 0
    for ticker, cik in ticker_to_cik_override.items():
        if ticker.upper() not in ticker_to_cik:
            ticker_to_cik[ticker.upper()] = cik
            n_override_added += 1
    print(f"Merged {n_override_added} manual CIK override(s) for delisted/renamed known tickers")

    known_path = os.path.join(DATA_DIR, "parsed_qa", "evasion_scores.csv")
    known_df = pd.read_csv(known_path, dtype=str)
    known_tickers = sorted(known_df["company_ticker"].dropna().unique())
    known_ciks = {ticker_to_cik[t.upper()] for t in known_tickers if t.upper() in ticker_to_cik}
    print(f"Resolved CIK for {len(known_ciks)}/{len(known_tickers)} known tickers")

    # Pre-seed seen_acc with everything already on disk so Part A/B never
    # re-add already-downloaded filings to the hit list in the first place.
    seen_acc = set(existing_accessions)

    hits = query_efts(ticker_to_cik, seen_acc)
    save_filing_index(hits)  # merges into the existing 8715-row filing_index.csv

    new_filing_df = pd.DataFrame(hits)
    stats = download_transcripts(new_filing_df, known_ciks)

    n_new_files_on_disk = len(os.listdir(RAW_DIR)) - n_existing_files

    print("\n" + "=" * 60)
    print("EXPANSION SUMMARY")
    print("=" * 60)
    print(f"Transcripts before this run:                {n_existing_files}")
    print(f"New transcripts downloaded this run:        {stats['downloaded']}")
    print(f"  (raw_transcripts/ file count confirms:    {n_new_files_on_disk} new files)")
    print(f"Transcripts after this run:                 {n_existing_files + stats['downloaded']}")
    print(f"New unique companies added:                 {len(stats['new_company_ciks'])}")
    print(f"Additional quarters from the 43 known companies: {stats['existing_company_new_quarters']}")
    print(f"Failed validation (word count / phrase check): {stats['failed']}")
    print(f"Skipped (no usable exhibit found):          {stats['skipped']}")

    print("\nAll done. No parsing or scoring was run -- raw files only.")


if __name__ == "__main__":
    main()
