"""
Build the modeling feature matrix: data/features/master_panel.csv.

Combines, per earnings-call transcript:
  - evasion scores (Day 3 LLM judge output)
  - forward cumulative abnormal return (CAR) around the NEXT earnings
    announcement, pulled from yfinance
  - accounting features from the PRIOR fiscal quarter (SEC XBRL), to avoid
    leaking the quarter being announced on the call itself
  - Loughran-McDonald sentiment word proportions computed over the prepared
    remarks section of the transcript

Network calls are cached per company (per ticker for yfinance, per CIK for
SEC XBRL) rather than per transcript row, since the same company appears in
multiple quarters. This cuts redundant calls by ~3x versus a naive per-row
loop while still sleeping between every actual HTTP request.
"""

import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser import html_to_lines, QA_MARKERS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_TRANSCRIPTS_DIR = DATA_DIR / "raw_transcripts"
RESULTS_DIR = BASE_DIR / "results"
FEATURES_DIR = DATA_DIR / "features"

TRANSCRIPT_EVASION_CSV = DATA_DIR / "parsed_qa" / "transcript_evasion.csv"
EVASION_SCORES_CSV = DATA_DIR / "parsed_qa" / "evasion_scores.csv"
FILING_INDEX_CSV = DATA_DIR / "filing_index.csv"
LM_DICT_CSV = DATA_DIR / "lm_dictionary.csv"
MISSING_CAR_CSV = RESULTS_DIR / "missing_car_labels.csv"
MISSING_XBRL_CSV = RESULTS_DIR / "missing_xbrl_features.csv"
MASTER_PANEL_CSV = FEATURES_DIR / "master_panel.csv"

SEC_HEADERS = {"User-Agent": "Kunal Chevuri kunalchevuri510@gmail.com"}
SLEEP_SECONDS = 0.3

LM_DICT_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"

REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
GROSS_PROFIT_TAGS = ["GrossProfit"]
OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
ASSETS_TAGS = ["Assets"]

FINAL_COLUMNS = [
    "transcript_id", "company_ticker", "filing_date", "filing_year",
    "mean_evasion_score", "evasion_variance", "max_evasion_score", "n_pairs",
    "car_3day", "car_direction", "car_date_estimated",
    "revenue_growth", "gross_margin", "operating_margin", "roa",
    "lm_positive", "lm_negative", "lm_uncertainty", "lm_litigious",
    "prepared_remarks_fallback",
]


# ── Step 1: base data ────────────────────────────────────────────────────────

def load_base_data():
    te = pd.read_csv(TRANSCRIPT_EVASION_CSV, dtype={"transcript_id": str})
    te["transcript_id"] = te["transcript_id"].str.zfill(18)

    es = pd.read_csv(EVASION_SCORES_CSV, dtype={"transcript_id": str})
    es["transcript_id"] = es["transcript_id"].str.zfill(18)
    agg = es.groupby("transcript_id")["evasion_score"].agg(
        evasion_variance="var", max_evasion_score="max"
    ).reset_index()

    df = te.merge(agg, on="transcript_id", how="left")
    df["filing_year"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.year
    return df


# ── Step 1b: accession / CIK / period_of_report lookup ─────────────────────

def build_filing_lookup():
    fi = pd.read_csv(FILING_INDEX_CSV, dtype=str)
    fi["acc_nodash"] = fi["accession_no"].str.replace("-", "", regex=False)
    lookup = {}
    for _, row in fi.iterrows():
        acc = row["acc_nodash"]
        if acc not in lookup:
            lookup[acc] = {
                "cik": row["cik"],
                "period_of_report": row["period_of_report"],
                "entity_name": row["entity_name"],
            }
    return lookup


def build_raw_file_lookup():
    """transcript_id (zero-padded, 18 char) -> Path to raw .htm transcript file."""
    lookup = {}
    for fpath in RAW_TRANSCRIPTS_DIR.glob("*.htm"):
        parts = fpath.name.replace(".htm", "").split("_")
        if len(parts) >= 3:
            key = parts[0] + parts[1] + parts[2]
            lookup[key] = fpath
    return lookup


# ── Step 2: CAR labels via yfinance ──────────────────────────────────────────

def fetch_spy_history(start, end):
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(start=start, end=end, auto_adjust=True)
        time.sleep(SLEEP_SECONDS)
        return hist
    except Exception as e:
        print(f"  WARNING: failed to fetch SPY history: {e}")
        return pd.DataFrame()


def fetch_ticker_earnings_dates(ticker):
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=60)
        time.sleep(SLEEP_SECONDS)
        if ed is None or ed.empty:
            return None
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None)
        return ed.sort_index()
    except Exception as e:
        print(f"  WARNING: failed to fetch earnings_dates for {ticker}: {e}")
        return None


def fetch_ticker_history(ticker, start, end):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start, end=end, auto_adjust=True)
        time.sleep(SLEEP_SECONDS)
        return hist
    except Exception as e:
        print(f"  WARNING: failed to fetch price history for {ticker}: {e}")
        return pd.DataFrame()


def find_next_earnings_date(earnings_dates, filing_date):
    if earnings_dates is None or earnings_dates.empty:
        return None
    future = earnings_dates.index[earnings_dates.index > filing_date]
    if len(future) == 0:
        return None
    return future.min()


def tz_naive_index(hist):
    idx = hist.index
    return idx.tz_localize(None) if idx.tz is not None else idx


def compute_car_3day(stock_hist_full, spy_hist_full, earnings_date):
    """
    stock_hist_full / spy_hist_full: FULL cached daily price history (not
    pre-sliced to a window around earnings_date). Returns are computed on the
    full series before any windowing, so day_before's return is never NaN
    just because a window boundary happened to cut off the previous trading
    day. Pre-slicing then computing pct_change() on the slice was the bug:
    the first row of any window always has a NaN return (no prior row in the
    slice to diff against), which surfaced whenever a market holiday sat
    right at the window's start edge.

    earnings_date must already be tz-naive and normalized to midnight —
    get_earnings_dates() returns timestamps with the real announcement time
    (e.g. 17:00:00), which if left as-is causes off-by-one-day exclusions
    when compared against midnight-indexed daily price bars.
    """
    if stock_hist_full.empty or spy_hist_full.empty:
        return None

    stock_idx = tz_naive_index(stock_hist_full)
    spy_idx = tz_naive_index(spy_hist_full)

    stock_ret = stock_hist_full["Close"].pct_change()
    stock_ret.index = stock_idx
    spy_ret = spy_hist_full["Close"].pct_change()
    spy_ret.index = spy_idx

    on_or_before = stock_idx[stock_idx <= earnings_date]
    if len(on_or_before) == 0:
        return None
    day0 = on_or_before.max()
    day0_pos = stock_idx.get_loc(day0)
    if day0_pos == 0 or day0_pos >= len(stock_idx) - 1:
        return None

    day_before = stock_idx[day0_pos - 1]
    day_after = stock_idx[day0_pos + 1]

    total = 0.0
    for day in (day_before, day0, day_after):
        if day not in stock_ret.index or day not in spy_ret.index:
            return None
        sr = stock_ret.loc[day]
        mr = spy_ret.loc[day]
        if pd.isna(sr) or pd.isna(mr):
            return None
        total += (sr - mr)
    return total


def build_car_labels(df):
    tickers = sorted(df["company_ticker"].dropna().unique())
    print(f"Fetching CAR data for {len(tickers)} unique tickers...")

    global_start = (pd.to_datetime(df["filing_date"]).min() - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    global_end = (pd.to_datetime(df["filing_date"]).max() + pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    spy_hist = fetch_spy_history(global_start, global_end)
    print(f"  SPY history rows: {len(spy_hist)}")

    earnings_cache = {}
    price_cache = {}
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}: fetching earnings_dates + price history")
        earnings_cache[ticker] = fetch_ticker_earnings_dates(ticker)
        price_cache[ticker] = fetch_ticker_history(ticker, global_start, global_end)

    missing_rows = []
    car_3day_list = []
    car_direction_list = []
    car_date_estimated_list = []

    for _, row in df.iterrows():
        ticker = row["company_ticker"]
        filing_date = pd.to_datetime(row["filing_date"], errors="coerce")
        car_3day = None
        estimated = False

        if pd.isna(filing_date) or not isinstance(ticker, str):
            missing_rows.append({"transcript_id": row["transcript_id"], "company_ticker": ticker,
                                  "reason": "missing filing_date or ticker"})
            car_3day_list.append(None)
            car_direction_list.append(None)
            car_date_estimated_list.append(None)
            continue

        earnings_dates = earnings_cache.get(ticker)
        next_earn = find_next_earnings_date(earnings_dates, filing_date)

        if next_earn is None:
            next_earn = filing_date + pd.Timedelta(days=91)
            estimated = True

        next_earn = pd.Timestamp(next_earn)
        if next_earn.tz is not None:
            next_earn = next_earn.tz_localize(None)
        next_earn = next_earn.normalize()

        stock_hist_full = price_cache.get(ticker, pd.DataFrame())
        if stock_hist_full.empty:
            missing_rows.append({"transcript_id": row["transcript_id"], "company_ticker": ticker,
                                  "reason": "no price history available"})
            car_3day_list.append(None)
            car_direction_list.append(None)
            car_date_estimated_list.append(None)
            continue

        try:
            car_3day = compute_car_3day(stock_hist_full, spy_hist, next_earn)
        except Exception as e:
            car_3day = None

        if car_3day is None:
            missing_rows.append({"transcript_id": row["transcript_id"], "company_ticker": ticker,
                                  "reason": "insufficient price data around earnings window"})
            car_3day_list.append(None)
            car_direction_list.append(None)
            car_date_estimated_list.append(None)
            continue

        car_3day_list.append(car_3day)
        car_direction_list.append(1 if car_3day > 0 else 0)
        car_date_estimated_list.append(estimated)

    df = df.copy()
    df["car_3day"] = car_3day_list
    df["car_direction"] = car_direction_list
    df["car_date_estimated"] = car_date_estimated_list

    if missing_rows:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(missing_rows).to_csv(MISSING_CAR_CSV, index=False, encoding="utf-8")
            print(f"  Logged {len(missing_rows)} row(s) with missing CAR labels to {MISSING_CAR_CSV}")
        except Exception as e:
            print(f"  WARNING: failed to write {MISSING_CAR_CSV}: {e}")

    return df


# ── Step 3: SEC XBRL accounting features ────────────────────────────────────

def fetch_companyfacts(cik10):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        time.sleep(SLEEP_SECONDS)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        print(f"  WARNING: failed to fetch companyfacts for CIK {cik10}: {e}")
        return None


def get_merged_usd_entries(facts, tag_list):
    """
    Merge USD fact entries across every candidate tag in tag_list instead of
    locking onto just the first one present. Companies sometimes switch XBRL
    tags mid-history (e.g. FTI Consulting reported under 'Revenues' through
    2020 then switched to an ASC-606 contract-revenue tag afterward) — picking
    only the first available tag silently stops picking up newer quarters and
    the prior-quarter lookup then repeats the same stale values forever.
    build_quarterly_duration_series/build_instant_series dedupe by end date
    (keeping the entry with the latest 'filed' date), so merging entries from
    multiple tags is safe even if two tags briefly overlap on the same period.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged = []
    for tag in tag_list:
        if tag in gaap:
            merged.extend(gaap[tag]["units"].get("USD", []))
    return merged


def build_quarterly_duration_series(usd_entries):
    """Dedup single-quarter (~80-100 day) duration facts by end date, keep latest filed."""
    by_end = {}
    for e in usd_entries:
        start = e.get("start")
        end = e.get("end")
        val = e.get("val")
        filed = e.get("filed", "")
        if not start or not end or val is None:
            continue
        try:
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        except Exception:
            continue
        if not (80 <= days <= 100):
            continue
        if end not in by_end or filed > by_end[end]["filed"]:
            by_end[end] = {"val": val, "filed": filed}
    series = sorted(
        [(pd.Timestamp(end), v["val"]) for end, v in by_end.items()],
        key=lambda x: x[0],
    )
    return series


def build_instant_series(usd_entries):
    by_end = {}
    for e in usd_entries:
        end = e.get("end")
        val = e.get("val")
        filed = e.get("filed", "")
        if not end or val is None:
            continue
        if end not in by_end or filed > by_end[end]["filed"]:
            by_end[end] = {"val": val, "filed": filed}
    series = sorted(
        [(pd.Timestamp(end), v["val"]) for end, v in by_end.items()],
        key=lambda x: x[0],
    )
    return series


def nearest_instant_value(series, target_date, tolerance_days=10):
    best = None
    best_diff = None
    for d, v in series:
        diff = abs((d - target_date).days)
        if diff <= tolerance_days and (best_diff is None or diff < best_diff):
            best = v
            best_diff = diff
    return best


def compute_accounting_features(facts, period_of_report):
    result = {"revenue_growth": None, "gross_margin": None, "operating_margin": None, "roa": None}

    rev_entries = get_merged_usd_entries(facts, REVENUE_TAGS)
    rev_series = build_quarterly_duration_series(rev_entries)
    quarters_before = [q for q in rev_series if q[0] < period_of_report]

    if len(quarters_before) == 0:
        return result

    q_minus_1_date, q_minus_1_rev = quarters_before[-1]

    if len(quarters_before) >= 2:
        q_minus_2_date, q_minus_2_rev = quarters_before[-2]
        if q_minus_2_rev not in (0, None):
            result["revenue_growth"] = (q_minus_1_rev - q_minus_2_rev) / abs(q_minus_2_rev) * 100

    gp_entries = get_merged_usd_entries(facts, GROSS_PROFIT_TAGS)
    gp_series = dict(build_quarterly_duration_series(gp_entries))
    gp_val = gp_series.get(q_minus_1_date)
    if gp_val is not None and q_minus_1_rev not in (0, None):
        result["gross_margin"] = gp_val / q_minus_1_rev

    oi_entries = get_merged_usd_entries(facts, OPERATING_INCOME_TAGS)
    oi_series = dict(build_quarterly_duration_series(oi_entries))
    oi_val = oi_series.get(q_minus_1_date)
    if oi_val is not None and q_minus_1_rev not in (0, None):
        result["operating_margin"] = oi_val / q_minus_1_rev

    ni_entries = get_merged_usd_entries(facts, NET_INCOME_TAGS)
    ni_series = dict(build_quarterly_duration_series(ni_entries))
    ni_val = ni_series.get(q_minus_1_date)

    assets_entries = get_merged_usd_entries(facts, ASSETS_TAGS)
    assets_series = build_instant_series(assets_entries)
    assets_val = nearest_instant_value(assets_series, q_minus_1_date)

    if ni_val is not None and assets_val not in (0, None):
        result["roa"] = ni_val / assets_val

    return result


def build_accounting_features(df, filing_lookup):
    ciks = {}
    for _, row in df.iterrows():
        meta = filing_lookup.get(row["transcript_id"])
        if meta:
            ciks[row["company_ticker"]] = meta["cik"]

    print(f"Fetching XBRL companyfacts for {len(ciks)} unique CIKs...")
    facts_cache = {}
    for i, (ticker, cik10) in enumerate(sorted(ciks.items()), 1):
        print(f"  [{i}/{len(ciks)}] {ticker} (CIK {cik10})")
        facts_cache[ticker] = fetch_companyfacts(cik10)

    missing_rows = []
    rev_growth_list, gross_margin_list, op_margin_list, roa_list = [], [], [], []

    for _, row in df.iterrows():
        ticker = row["company_ticker"]
        meta = filing_lookup.get(row["transcript_id"])
        facts = facts_cache.get(ticker)

        if meta is None or facts is None:
            missing_rows.append({"transcript_id": row["transcript_id"], "company_ticker": ticker,
                                  "reason": "no filing_index match or companyfacts fetch failed"})
            rev_growth_list.append(None)
            gross_margin_list.append(None)
            op_margin_list.append(None)
            roa_list.append(None)
            continue

        try:
            period_of_report = pd.Timestamp(meta["period_of_report"])
            feats = compute_accounting_features(facts, period_of_report)
        except Exception as e:
            missing_rows.append({"transcript_id": row["transcript_id"], "company_ticker": ticker,
                                  "reason": f"error computing accounting features: {e}"})
            feats = {"revenue_growth": None, "gross_margin": None, "operating_margin": None, "roa": None}

        if all(v is None for v in feats.values()):
            missing_rows.append({"transcript_id": row["transcript_id"], "company_ticker": ticker,
                                  "reason": "no usable XBRL quarterly data found for prior quarter"})

        rev_growth_list.append(feats["revenue_growth"])
        gross_margin_list.append(feats["gross_margin"])
        op_margin_list.append(feats["operating_margin"])
        roa_list.append(feats["roa"])

    df = df.copy()
    df["revenue_growth"] = rev_growth_list
    df["gross_margin"] = gross_margin_list
    df["operating_margin"] = op_margin_list
    df["roa"] = roa_list

    if missing_rows:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(missing_rows).to_csv(MISSING_XBRL_CSV, index=False, encoding="utf-8")
            print(f"  Logged {len(missing_rows)} row(s) with missing XBRL features to {MISSING_XBRL_CSV}")
        except Exception as e:
            print(f"  WARNING: failed to write {MISSING_XBRL_CSV}: {e}")

    return df


# ── Step 4: Loughran-McDonald sentiment ─────────────────────────────────────

def download_lm_dictionary():
    if LM_DICT_CSV.exists():
        print(f"LM dictionary already cached at {LM_DICT_CSV}, reusing.")
        return pd.read_csv(LM_DICT_CSV)

    print("Downloading LM master dictionary...")
    try:
        resp = requests.get(LM_DICT_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
        lm = pd.read_csv(io.BytesIO(resp.content))
    except Exception as e:
        print(f"ERROR: failed to download LM dictionary: {e}")
        sys.exit(1)

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        lm.to_csv(LM_DICT_CSV, index=False, encoding="utf-8")
        print(f"Saved LM dictionary to {LM_DICT_CSV} ({len(lm)} words)")
    except Exception as e:
        print(f"WARNING: failed to cache LM dictionary: {e}")

    return lm


def build_lm_word_sets(lm_df):
    sets = {}
    for cat, col in [("positive", "Positive"), ("negative", "Negative"),
                      ("uncertainty", "Uncertainty"), ("litigious", "Litigious")]:
        sets[cat] = set(lm_df.loc[lm_df[col] != 0, "Word"].astype(str).str.upper())
    return sets


WORD_RE = re.compile(r"[A-Za-z]+")


MARKER_LINE_MAX_LEN = 60


def extract_prepared_remarks(html_content):
    """
    Everything before the Q&A section boundary. The QA_MARKERS phrase can also
    appear mid-sentence in the operator's spoken introduction (e.g. "...we
    will have a question and answer session..."), which is not the real
    section boundary. Real section headers are short standalone lines, so
    prefer the first short-line match; only fall back to a substring match
    (e.g. from the operator's intro) if no standalone header line is found.
    """
    lines = html_to_lines(html_content)

    qa_idx = -1
    for i, line in enumerate(lines):
        if len(line) <= MARKER_LINE_MAX_LEN and any(m in line.upper() for m in QA_MARKERS):
            qa_idx = i
            break

    if qa_idx < 0:
        for i, line in enumerate(lines):
            if any(m in line.upper() for m in QA_MARKERS):
                qa_idx = i
                break

    if qa_idx <= 0:
        return None
    return " ".join(lines[:qa_idx])


def compute_lm_proportions(text, word_sets):
    words = [w.upper() for w in WORD_RE.findall(text)]
    total = len(words)
    if total == 0:
        return {"lm_positive": None, "lm_negative": None, "lm_uncertainty": None, "lm_litigious": None}
    return {
        "lm_positive": sum(1 for w in words if w in word_sets["positive"]) / total,
        "lm_negative": sum(1 for w in words if w in word_sets["negative"]) / total,
        "lm_uncertainty": sum(1 for w in words if w in word_sets["uncertainty"]) / total,
        "lm_litigious": sum(1 for w in words if w in word_sets["litigious"]) / total,
    }


def build_lm_features(df, raw_file_lookup):
    lm_df = download_lm_dictionary()
    word_sets = build_lm_word_sets(lm_df)

    lm_positive_list, lm_negative_list, lm_uncertainty_list, lm_litigious_list = [], [], [], []
    fallback_list = []

    print(f"Computing LM sentiment for {len(df)} transcripts...")
    for _, row in df.iterrows():
        fpath = raw_file_lookup.get(row["transcript_id"])
        fallback = False
        proportions = {"lm_positive": None, "lm_negative": None, "lm_uncertainty": None, "lm_litigious": None}

        if fpath is None:
            fallback_list.append(None)
            lm_positive_list.append(None)
            lm_negative_list.append(None)
            lm_uncertainty_list.append(None)
            lm_litigious_list.append(None)
            continue

        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                html_content = f.read()

            prepared = extract_prepared_remarks(html_content)
            if prepared is None or len(prepared.split()) < 20:
                fallback = True
                lines = html_to_lines(html_content)
                prepared = " ".join(lines)

            proportions = compute_lm_proportions(prepared, word_sets)
        except Exception as e:
            print(f"  WARNING: failed LM extraction for {row['transcript_id']}: {e}")

        fallback_list.append(fallback)
        lm_positive_list.append(proportions["lm_positive"])
        lm_negative_list.append(proportions["lm_negative"])
        lm_uncertainty_list.append(proportions["lm_uncertainty"])
        lm_litigious_list.append(proportions["lm_litigious"])

    df = df.copy()
    df["lm_positive"] = lm_positive_list
    df["lm_negative"] = lm_negative_list
    df["lm_uncertainty"] = lm_uncertainty_list
    df["lm_litigious"] = lm_litigious_list
    df["prepared_remarks_fallback"] = fallback_list
    return df


# ── Step 5 & 6: merge + summary ──────────────────────────────────────────────

def print_summary(df):
    print("\n" + "=" * 70)
    print("FINAL PANEL SUMMARY")
    print("=" * 70)
    print(f"Total company-quarter observations: {len(df)}")
    print(f"Filing year range: {df['filing_year'].min()} - {df['filing_year'].max()}")
    print(f"Unique companies: {df['company_ticker'].nunique()}")
    print(f"Mean car_3day: {df['car_3day'].mean():.6f}")
    print(f"Median car_3day: {df['car_3day'].median():.6f}")
    frac_positive = (df['car_direction'] == 1).mean()
    print(f"Fraction with positive car_3day: {frac_positive:.4f}")
    print(f"Rows with car_date_estimated=True: {(df['car_date_estimated'] == True).sum()}")

    acct_cols = ["revenue_growth", "gross_margin", "operating_margin", "roa"]
    n_missing_acct = df[acct_cols].isna().any(axis=1).sum()
    print(f"Rows with any missing accounting feature: {n_missing_acct}")

    lm_cols = ["lm_positive", "lm_negative", "lm_uncertainty", "lm_litigious"]
    n_missing_lm = df[lm_cols].isna().any(axis=1).sum()
    print(f"Rows with any missing LM feature: {n_missing_lm}")

    print("\nFirst 5 rows:")
    print(df.head(5).to_string())
    print("=" * 70)


def main():
    print("STEP 1: Loading base data...")
    df = load_base_data()
    print(f"  Loaded {len(df)} transcript-level rows")

    filing_lookup = build_filing_lookup()
    raw_file_lookup = build_raw_file_lookup()

    print("\nSTEP 2: Pulling CAR labels from yfinance...")
    df = build_car_labels(df)

    print("\nSTEP 3: Pulling accounting features from SEC XBRL...")
    df = build_accounting_features(df, filing_lookup)

    print("\nSTEP 4: Computing Loughran-McDonald sentiment...")
    df = build_lm_features(df, raw_file_lookup)

    print("\nSTEP 5: Merging into master panel...")
    n_before = len(df)
    df = df.dropna(subset=["car_3day"])
    n_after = len(df)
    print(f"  Dropped {n_before - n_after} row(s) missing car_3day label")

    final = df[FINAL_COLUMNS].copy()
    final["car_direction"] = final["car_direction"].astype(int)

    try:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        final.to_csv(MASTER_PANEL_CSV, index=False, encoding="utf-8")
        print(f"  Saved master panel to {MASTER_PANEL_CSV}")
    except Exception as e:
        print(f"ERROR: failed to write {MASTER_PANEL_CSV}: {e}")
        sys.exit(1)

    print("\nSTEP 6: Summary")
    print_summary(final)


if __name__ == "__main__":
    main()
