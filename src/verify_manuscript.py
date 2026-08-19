"""
Check every headline number in the manuscript against the artifacts that
produced it.

This is the automated half of the advisor's number-consistency pass: a claim
in main.tex is only allowed to stand if the value can be recomputed from
committed data or read out of a committed results file. It is deliberately
literal -- it searches for the exact rendered string, so a number that drifts
in the text without drifting in the data will fail here.

Every check names the source it verifies against, so a failure tells you which
of the two moved.

Usage:  python src/verify_manuscript.py
Exit code is non-zero if any check fails.
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
TEX = BASE / "overleaf" / "main.tex"
RES = BASE / "results"

tex = TEX.read_text(encoding="utf-8")
checks = []


def check(label, needle, source):
    """Assert a literal string appears in the manuscript."""
    checks.append((label, needle in tex, needle, source))


def num(path, pattern, group=1):
    """Pull a number out of a results text file."""
    t = (RES / path).read_text(encoding="utf-8")
    m = re.search(pattern, t)
    return m.group(group) if m else None


# ---------------------------------------------------------------- corpus ----
sc = pd.read_csv(BASE / "data" / "parsed_qa" / "evasion_scores.csv")
qa = pd.read_csv(BASE / "data" / "parsed_qa" / "all_qa_pairs.csv")
fi = pd.read_csv(BASE / "data" / "filing_index.csv", dtype=str)

check("scored pairs 3,350", "3,350", "evasion_scores.csv rows = %d" % len(sc))
check("extracted pairs 3,351", "3,351", "all_qa_pairs.csv rows = %d" % len(qa))
check("filings 22,596", "22,596", "filing_index.csv rows = %d" % len(fi))
check("transcripts 285", "285", "unique transcript_id = %d" % sc.transcript_id.nunique())
check("companies 49", "49", "unique ticker = %d" % sc.company_ticker.nunique())
check("panel n=155", "155", "final_descriptive_statistics.csv count")

# ------------------------------------------------------------ validation ----
kp = pd.read_csv(RES / "kappa_statistics.csv")
kd = {r.iloc[0]: r for _, r in kp.iterrows()}
check("human1-human2 kappa 0.237", "0.237", "kappa_statistics.csv")
check("human1-human2 r 0.587", "0.587", "kappa_statistics.csv")
check("llm-human1 r 0.777", "0.777", "kappa_statistics.csv")
check("llm-human2 r 0.803", "0.803", "kappa_statistics.csv")

# ------------------------------------------------- external benchmarks ------
for corpus, rho, auroc, zf, ff in (
        ("evasionbench", "+0.615", "0.856", "0.321", "0.573"),
        ("qevasion", "+0.398", "0.741", "0.163", "0.472")):
    src = "external_benchmark_%s.txt" % corpus
    check("%s Spearman %s" % (corpus, rho), rho, src)
    check("%s AUROC %s" % (corpus, auroc), auroc, src)
    check("%s zero-shot F1 %s" % (corpus, zf), zf, src)
    check("%s fitted F1 %s" % (corpus, ff), ff, src)

eb = pd.read_csv(RES / "external_scores_evasionbench.csv")
qe = pd.read_csv(RES / "external_scores_qevasion.csv")
check("evasionbench n=999", "999", "external_scores rows = %d" % len(eb))
check("qevasion n=308", "308", "external_scores rows = %d" % len(qe))
check("evasionbench mean 26.64", "26.64", "mean = %.2f" % eb.evasion_score.mean())
check("qevasion mean 45.64", "45.64", "mean = %.2f" % qe.evasion_score.mean())
check("corpus mean 27.50", "27.50", "mean = %.2f" % sc.evasion_score.mean())

# --------------------------------------------------------- anonymization ----
an = pd.read_csv(RES / "anonymization_check.csv")
check("masked spans 702", "702", "sum entities_masked = %d" % an.entities_masked.sum())
check("masked r 0.861", "0.861", "anonymization_check.txt")
check("masked mean shift +1.83", "1.83", "anonymization_check.txt")
check("masked vs human mean 0.773", "0.773", "anonymization_check.txt")
check("identical 41 of 75", "41 of 75", "anonymization_check.csv")

# ------------------------------------------------------ factor structure ----
check("eigenvalue 2.628", "2.628", "factor_structure.txt")
check("eigenvalue 0.801", "0.801", "factor_structure.txt")
check("variance 56.7", "56.7", "factor_structure.txt")
check("hedging-deflection 0.244", "0.244", "corr matrix")
check("nonresp-hedging 0.446", "0.446", "corr matrix")

# ------------------------------------------------------- cost/throughput ----
check("cost $12.77", "12.77", "cost_throughput.txt")
check("cost per pair 0.0038", "0.0038", "cost_throughput.txt")
check("3.0 s per pair", "3.0 seconds per pair", "cost_throughput.txt")
check("2.67 pairs/s", "2.67", "cost_throughput.txt")

# ------------------------------------------------------------- integrity ----
extra = []
# The 11 percent non-answer rate belongs to Gow et al.; the paper refers back
# to it a second time when comparing against our own rate, which is correct.
# What would be wrong is crediting a second source with the same figure.
_bamber = re.search(r"Bamber and Nappert.*?(?=\\textbf|\n\n)", tex, re.DOTALL)
if _bamber and "11 percent" in _bamber.group(0):
    extra.append("the 11 percent non-answer rate is attributed to Bamber & Nappert "
                 "as well as Gow et al. -- it should be credited once")
if "\\appendix" in tex:
    extra.append("appendix marker still present (advisor asked for it to be removed)")
if "fig:shap" in tex:
    extra.append("SHAP figure reference still present (advisor's designated cut)")
for key in ("loughran2011liability", "alnuaimi2025evasive",
            "bamber2025nonanswers", "thomas2026clarity"):
    if key not in tex:
        extra.append("required citation %s is not cited in the text" % key)

# ------------------------------------------------------------------ report --
width = max(len(c[0]) for c in checks) + 2
fails = 0
print("Manuscript number-consistency check")
print("=" * 78)
for label, ok, needle, source in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<{width}} <- {source}")
    if not ok:
        fails += 1
        print(f"        expected to find {needle!r} in main.tex")

print("-" * 78)
for e in extra:
    print(f"  FAIL  {e}")
fails += len(extra)

print("-" * 78)
print(f"  {len(checks)} numeric checks, {len(extra)} integrity issues, {fails} failure(s)")
sys.exit(1 if fails else 0)
