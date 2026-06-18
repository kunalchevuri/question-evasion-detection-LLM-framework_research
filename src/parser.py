"""
EDGAR Earnings Call Q&A Parser
Extracts Q&A pairs from .htm transcript files in data/raw_transcripts/
Output: data/parsed_qa/all_qa_pairs.csv

Handles three HTML formats found in EDGAR transcripts:
  1. NG Converter / Refinitiv (CBIZ-style) - bold <p> speaker labels
  2. Workiva (Pacific Premier-style)        - bold-weight <div> labels
  3. Refinitiv image+OCR (Fossil-style)     - text in hidden <FONT size=1> blocks
"""

import csv
import html as html_mod
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw_transcripts"
OUT_DIR  = BASE_DIR / "data" / "parsed_qa"
OUT_CSV  = OUT_DIR / "all_qa_pairs.csv"

FILING_INDEX_CSV = BASE_DIR / "data" / "filing_index.csv"
CIK_CSV          = BASE_DIR / "data" / "sp500_tech_healthcare_ciks.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_Q_WORDS = 30
MIN_R_WORDS = 50
MIN_PAIRS   = 3

QA_MARKERS = [
    "QUESTIONS AND ANSWERS",
    "QUESTION AND ANSWER SESSION",
    "QUESTIONS & ANSWERS",
    "Q&A SESSION",
]

CORP_MARKERS = ["CORPORATE PARTICIPANTS"]
CONF_MARKERS = ["CONFERENCE CALL PARTICIPANTS"]

COMPANY_STOP = {
    'INC', 'LLC', 'LTD', 'CORP', 'GROUP', 'CO', 'COMPANY', 'CORPORATION',
    'CAPITAL', 'SECURITIES', 'PARTNERS', 'MANAGEMENT', 'INVESTMENTS',
    'RESEARCH', 'ADVISORS', 'ASSOCIATES', 'FINANCIAL', 'BANK', 'ASSET',
    'FUND', 'TRUST', 'HOLDINGS', 'INTERNATIONAL', 'GLOBAL', 'SERVICES',
}
ARTICLE_STOP = {'the', 'a', 'an', 'of', 'for', 'at', 'by', 'and', 'in',
                'with', 'from', 'its', 'on'}

# Matches "First Last" or "First M. Last" or "First A. B. Last" (2-4 words, all capitalized)
_NAME_RE = re.compile(r'^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z.]+){1,3}$')

CSV_FIELDS = [
    "transcript_id", "company_ticker", "filing_date",
    "question_text", "response_text", "analyst_name", "management_speaker",
]


# ── Metadata lookup ──────────────────────────────────────────────────────────

def build_meta_lookup():
    """Return dict: accession_no_no_dashes -> {ticker, entity_name, file_date}."""
    cik_to_ticker = {}
    if CIK_CSV.exists():
        df = pd.read_csv(CIK_CSV, dtype=str)
        for _, row in df.iterrows():
            cik = str(row.get("cik", "")).lstrip("0")
            cik_to_ticker[cik] = row.get("ticker", "")

    lookup = {}
    if FILING_INDEX_CSV.exists():
        df = pd.read_csv(FILING_INDEX_CSV, dtype=str)
        for _, row in df.iterrows():
            acc = str(row.get("accession_no", "")).replace("-", "")
            cik = str(row.get("cik", "")).lstrip("0")
            lookup[acc] = {
                "ticker":      cik_to_ticker.get(cik, ""),
                "entity_name": str(row.get("entity_name", "")),
                "file_date":   str(row.get("file_date", "")),
            }
    return lookup


def accession_from_fname(fname):
    """XXXX_YY_ZZZ_docname.htm -> XXXXYYZZZ (normalized, no dashes)."""
    parts = fname.replace(".htm", "").split("_")
    if len(parts) >= 3:
        return parts[0] + parts[1] + parts[2]
    return ""


# ── Name extraction from participant-listing lines ───────────────────────────

def extract_name(line):
    """
    Extract speaker's proper name from a participant directory line.
    Handles:
      "Jim MacDonald - First Analysis Securities; Analyst"   -> "Jim MacDonald"
      "Jerome P. Grisko CBIZ, Inc. - President, CEO"        -> "Jerome P. Grisko"
      "Christine Greany The Blueshirt Group, LLC - MD"       -> "Christine Greany"
      "Matthew Clark"                                         -> "Matthew Clark"
    """
    line = line.strip()
    # Strategy 1: split on " - " or " -- "; name must be <= 4 words
    parts = re.split(r'\s+[–—\-]+\s+', line, maxsplit=1)
    if len(parts) == 2 and 1 <= len(parts[0].split()) <= 4:
        return parts[0].strip()

    # Strategy 2: take first 1-3 words, stopping at company/article words
    words = line.split()
    name_words = []
    for word in words:
        clean = word.rstrip('.,;').upper()
        if clean in COMPANY_STOP:
            break
        if word.lower() in ARTICLE_STOP and name_words:
            break
        name_words.append(word)
        if len(name_words) >= 3:
            break
    return ' '.join(name_words)


# ── Participant section extraction ───────────────────────────────────────────

def extract_participants(text):
    """
    Scan plain text for CORPORATE/CONFERENCE CALL PARTICIPANTS sections.
    Returns (mgmt_names: set[UPPER], analyst_names: set[UPPER]).
    """
    mgmt = set()
    analyst = set()
    mode = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        lu = line.upper()

        if any(m in lu for m in CORP_MARKERS):
            mode = 'mgmt'
            continue
        if any(m in lu for m in CONF_MARKERS):
            mode = 'analyst'
            continue
        # End participant section when PRESENTATION or Q&A begins
        if mode and (lu.startswith('PRESENTATION') or any(m in lu for m in QA_MARKERS)):
            mode = None
            continue

        if mode is None or len(line) < 2 or len(line) > 150:
            continue

        # Skip continuation lines like "- Vertical Research Partners; Analyst"
        # that follow a name on its own line in some formats — these are firm/
        # title text, not a second participant.
        if re.match(r'^[\s\-–—]', line):
            continue

        name = extract_name(line)
        name = html_mod.unescape(name).strip()
        if name and 1 < len(name) < 55:
            (mgmt if mode == 'mgmt' else analyst).add(name.upper())

    return mgmt, analyst


# ── Speaker detection ────────────────────────────────────────────────────────

def is_name_like(text):
    """
    Return True if text looks like a speaker name (2-4 capitalized words).
    Rejects body-text starters like "Good morning" or "Thank you".
    """
    text = text.strip().rstrip('.')
    if not text or len(text) > 60:
        return False
    return bool(_NAME_RE.match(text))


def match_speaker(line, mgmt_names, analyst_names):
    """
    Return (clean_name, role) if line is a speaker label, else None.
    role in: 'management', 'analyst', 'operator'
    Uses prefix matching so "Name Company Title" still matches on "Name".
    """
    s = line.strip()
    if not s:
        return None
    up = s.upper()

    if up in ('OPERATOR', 'MODERATOR'):
        return s, 'operator'

    for name in mgmt_names:
        if up.startswith(name):
            return name.title(), 'management'
    for name in analyst_names:
        if up.startswith(name):
            return name.title(), 'analyst'

    return None


# ── HTML -> lines (handles all three formats) ────────────────────────────────

def html_to_lines(html_content):
    """
    Parse HTML and return clean list of text lines.
    For the Refinitiv image+OCR format (Fossil-style), the entire transcript
    text is packed into hidden <FONT size=1> blocks as one long string with
    sections separated by multiple spaces.  We split long lines on 2+ spaces.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    raw = soup.get_text(separator='\n', strip=True)

    result = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Long lines (>300 chars) indicate the hidden-text / OCR format
        if len(line) > 300:
            parts = re.split(r'  +', line)
            result.extend(p.strip() for p in parts if p.strip())
        else:
            result.append(line)
    return result


# ── Turn extraction ──────────────────────────────────────────────────────────

def _filter_body(lines):
    """
    Join body lines into text, skipping a leading title/firm line.
    Those appear as "- Robert W. Baird & Co.; Analyst" right after the speaker name
    when the transcript splits speaker-name and firm onto separate lines.
    """
    filtered = list(lines)
    if filtered and re.match(r'^[\s\-–]', filtered[0]) and len(filtered[0]) < 200:
        filtered = filtered[1:]
    return ' '.join(filtered)


def extract_turns(lines, mgmt_names, analyst_names):
    """
    Find the Q&A section in lines, then walk it collecting (speaker, role, text) turns.

    Fallback for transcripts with no CONFERENCE CALL PARTICIPANTS section:
    - After an Operator turn, an unknown name-like short line is classified as 'analyst'.
    - Discovered analyst names are tracked so their follow-up turns are also classified.
    """
    qa_idx = -1
    for i, line in enumerate(lines):
        if any(m in line.upper() for m in QA_MARKERS):
            qa_idx = i + 1
            break
    if qa_idx < 0:
        return []

    turns = []
    cur_spk = cur_role = None
    cur_body = []
    after_operator = False
    discovered = set()  # analyst names found via fallback

    for line in lines[qa_idx:]:
        result = match_speaker(line, mgmt_names, analyst_names)

        if result is None:
            s_up = line.strip().upper()
            if s_up in discovered:
                # Repeat appearance of a previously discovered analyst
                result = (line.strip(), 'analyst')
            elif after_operator and is_name_like(line):
                # First appearance: unknown name right after Operator = analyst
                result = (line.strip(), 'analyst')
                discovered.add(s_up)

        if result:
            if cur_spk and cur_body:
                body = _filter_body(cur_body)
                if body:
                    turns.append((cur_spk, cur_role, body))
            cur_spk, cur_role = result
            after_operator = (cur_role == 'operator')
            cur_body = []
        elif cur_spk:
            cur_body.append(line)

    if cur_spk and cur_body:
        body = _filter_body(cur_body)
        if body:
            turns.append((cur_spk, cur_role, body))

    return turns


# ── Q&A pair building ────────────────────────────────────────────────────────

def build_pairs(turns):
    """
    From (speaker, role, text) turns, build Q&A pair dicts.
    Pattern: one or more analyst turns -> one or more management turns.
    """
    pairs = []
    i = 0
    while i < len(turns):
        spk, role, text = turns[i]

        if role != 'analyst':
            i += 1
            continue

        # Accumulate consecutive analyst turns (multi-part questions)
        q_parts = [text]
        analyst_name = spk
        j = i + 1
        while j < len(turns) and turns[j][1] == 'analyst':
            q_parts.append(turns[j][2])
            j += 1

        # Accumulate management / unknown response turns
        r_parts = []
        mgmt_spks = []
        while j < len(turns) and turns[j][1] in ('management', 'unknown'):
            r_parts.append(turns[j][2])
            mgmt_spks.append(turns[j][0])
            j += 1

        if r_parts:
            q = ' '.join(q_parts)
            r = ' '.join(r_parts)
            if len(q.split()) >= MIN_Q_WORDS and len(r.split()) >= MIN_R_WORDS:
                pairs.append({
                    'question_text':      q,
                    'response_text':      r,
                    'analyst_name':       analyst_name,
                    'management_speaker': ', '.join(dict.fromkeys(mgmt_spks)),
                })
        i = j

    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    meta = build_meta_lookup()
    print(f"Metadata loaded: {len(meta)} filing index entries")

    htm_files = sorted(RAW_DIR.glob("*.htm"))
    print(f"Found {len(htm_files)} .htm files\n")

    all_pairs = []
    ok = skip = 0

    for idx, fpath in enumerate(htm_files, 1):
        try:
            acc = accession_from_fname(fpath.name)
            m = meta.get(acc, {})
            tid   = acc or fpath.stem
            tick  = m.get("ticker", "")
            fdate = m.get("file_date", "")

            with open(fpath, encoding='utf-8', errors='replace') as f:
                html = f.read()

            lines = html_to_lines(html)
            full_text = '\n'.join(lines)

            mgmt_names, analyst_names = extract_participants(full_text)
            turns = extract_turns(lines, mgmt_names, analyst_names)
            pairs = build_pairs(turns)
            n = len(pairs)

            if n < MIN_PAIRS:
                print(
                    f"  [{idx:3d}/{len(htm_files)}] SKIP  "
                    f"{fpath.name[:52]:<52}  "
                    f"{n} pairs  (mgmt={len(mgmt_names)} anl={len(analyst_names)})"
                )
                skip += 1
                continue

            for p in pairs:
                p.update(transcript_id=tid, company_ticker=tick, filing_date=fdate)
            all_pairs.extend(pairs)
            ok += 1
            print(
                f"  [{idx:3d}/{len(htm_files)}] OK    "
                f"{fpath.name[:52]:<52}  "
                f"{n} pairs"
            )

        except Exception as exc:
            print(f"  [{idx:3d}/{len(htm_files)}] ERROR {fpath.name}: {exc}")
            skip += 1

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_pairs)

    print(f"\n{'='*60}")
    print(f"Transcripts kept : {ok}")
    print(f"Transcripts skip : {skip}")
    print(f"Total Q&A pairs  : {len(all_pairs)}")
    print(f"Output           : {OUT_CSV}")
    if len(all_pairs) >= 800:
        print("Target 800+      : ACHIEVED")
    else:
        print(f"Target 800+      : NOT MET ({len(all_pairs)} < 800)")
    print('='*60)


if __name__ == "__main__":
    main()
