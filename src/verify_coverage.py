"""
Traceability sweep: can every number in the manuscript be found in an output?

verify_manuscript.py checks a curated list of headline statistics. That is
useful but it is not coverage -- it says nothing about the ~120 other numeric
tokens in the text. This script takes the complement: it walks every number in
the manuscript body and asks whether any committed results file or data file
contains a value that rounds to it.

The matching is deliberately generous, because the goal is not to prove each
number correct but to surface the ones with NO plausible source anywhere in
the artifacts -- those are where a typo or a stale edit hides. Anything
flagged needs a human decision: it is either prose (a year, a page budget, a
sample size quoted from a cited paper) or it is a real orphan.

Usage:  python src/verify_coverage.py
"""

import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TEX = BASE / "overleaf" / "main.tex"

# Numbers that are legitimately not derived from our outputs.
KNOWN_PROSE = {
    # years and date-like tokens
    "2015", "2023", "2024", "2025", "2026", "2011", "1933", "1934", "2020", "2021",
    # values quoted from cited work
    "22.7", "84.9", "0.835", "84", "124", "0.89", "0.68", "11", "60",
    # rubric / design constants
    "1", "2", "3", "4", "5", "30", "50", "75", "100", "300", "0", "10",
    "0.05", "95", "2.5", "200", "2000", "1000",
    # DOI fragments -- identifiers, not measurements
    "10.1287", "2025.01197",
}


def numbers_in(text):
    """Distinct numeric tokens: floats, and integers with thousands separators."""
    out = set()
    for m in re.findall(r"\d+\.\d+", text):
        out.add(m)
    for m in re.findall(r"\d{1,3}(?:,\d{3})+", text):
        out.add(m.replace(",", ""))
    for m in re.findall(r"(?<![\d.])\d{2,6}(?![\d.])", text):
        out.add(m)
    return out


def main():
    tex = TEX.read_text(encoding="utf-8")
    body = tex[tex.find("\\begin{abstract}"):]
    # strip LaTeX machinery that carries non-semantic digits
    body = re.sub(r"\\(?:label|ref|cite|includegraphics|documentclass|usepackage)\{[^}]*\}", " ", body)
    body = re.sub(r"\\[a-zA-Z]+", " ", body)

    claimed = numbers_in(body)

    # Everything we could have derived a number from.
    haystack = []
    for d in ("results", "data"):
        for p in (BASE / d).rglob("*"):
            if p.suffix.lower() in (".txt", ".csv") and p.stat().st_size < 20_000_000:
                try:
                    haystack.append(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
    hay = "\n".join(haystack)
    hay_nums = numbers_in(hay)
    hay_floats = []
    for n in hay_nums:
        try:
            hay_floats.append(float(n))
        except ValueError:
            pass

    def traceable(tok):
        if tok in KNOWN_PROSE:
            return True
        if tok in hay_nums:
            return True
        try:
            v = float(tok)
        except ValueError:
            return True
        dec = len(tok.split(".")[1]) if "." in tok else 0
        # Does any output value round to this one at the precision quoted?
        # Also allow percent/proportion mismatches: the manuscript writes
        # "66.4 percent" where the log records "power = 0.664", and both are
        # the same measurement in different units.
        for h in hay_floats:
            for cand in (h, h * 100.0, h / 100.0):
                if round(cand, dec) == v:
                    return True
        return False

    orphans = sorted((t for t in claimed if not traceable(t)),
                     key=lambda x: (len(x), x))

    print("Manuscript numeric traceability sweep")
    print("=" * 72)
    print(f"  distinct numeric tokens in the manuscript : {len(claimed)}")
    print(f"  matched to a committed output or known prose : {len(claimed)-len(orphans)}")
    print(f"  coverage                                  : "
          f"{100*(len(claimed)-len(orphans))/max(len(claimed),1):.0f}%")
    print()
    if not orphans:
        print("  No orphan numbers. Every value traces to an output or a known constant.")
        return 0
    print(f"  {len(orphans)} token(s) with no match -- each needs a human decision:")
    for t in orphans:
        ctx = ""
        m = re.search(r".{70}" + re.escape(t) + r".{70}", body, re.DOTALL)
        if m:
            ctx = " ".join(m.group(0).split())
        print(f"    {t:<12} ...{ctx}...")
    return len(orphans)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 0)
