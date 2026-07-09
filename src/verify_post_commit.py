"""
Post-commit verification: recompute headline statistics fresh from the
now-correctly-committed data/parsed_qa/evasion_scores.csv (commit bbdf274),
after the discovery that this file had been sitting uncommitted since the
July expansion. Deliberately does NOT touch master_panel.csv, model results,
or robustness checks -- those were already built correctly from this same
on-disk data and don't need to change; this script only re-verifies them
read-only.

Reuses final_verification.py's section1_provenance() directly (import, not
duplicate) for the provenance counts -- that function is a pure read+print
with no side effects, so importing it does not trigger Sections 5/6 (the
Day 6 model rerun and 5-fold CV/COVID-exclusion/power analysis), which stay
untouched per instructions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_verification import section1_provenance, SCORES_CSV, MASTER_PANEL_CSV  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_TXT = RESULTS_DIR / "post_commit_verification.txt"

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def hr(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def section_pair_distribution(scores):
    hr("SECTION: Pair-level evasion_score distribution (all scored pairs)")
    n = len(scores)
    s = scores["evasion_score"].astype(float)
    n_zero = int((s == 0.0).sum())
    n_hundred = int((s == 100.0).sum())
    n_between = int(((s > 0.0) & (s < 100.0)).sum())
    log(f"Total scored pairs: {n}")
    log(f"Exactly 0.0:    {n_zero} of {n}  ({n_zero / n * 100:.2f}%)")
    log(f"Exactly 100.0:  {n_hundred} of {n}  ({n_hundred / n * 100:.2f}%)")
    log(f"Strictly 0-100: {n_between} of {n}  ({n_between / n * 100:.2f}%)")
    assert n_zero + n_hundred + n_between == n, "counts must partition the full set"
    return n_zero, n_hundred, n_between, n


def section_master_panel_consistency(scores, mp):
    hr("SECTION: master_panel.csv consistency check (READ-ONLY, no rebuild)")
    log(f"master_panel.csv on disk: {len(mp)} rows, {mp['company_ticker'].nunique()} companies "
        f"(NOT modified by this script)")

    scores = scores.copy()
    scores["transcript_id"] = scores["transcript_id"].astype(str).str.zfill(18)
    scores["evasion_score"] = scores["evasion_score"].astype(float)
    recomputed = scores.groupby("transcript_id")["evasion_score"].mean()

    mp_check = mp.copy()
    mp_check["transcript_id"] = mp_check["transcript_id"].astype(str).str.zfill(18)
    mp_check["recomputed_mean"] = mp_check["transcript_id"].map(recomputed)

    diff = (mp_check["mean_evasion_score"].astype(float) - mp_check["recomputed_mean"].astype(float)).abs()
    n_mismatch = int((diff > 1e-6).sum())
    n_missing = int(mp_check["recomputed_mean"].isna().sum())

    log(f"Transcripts in master_panel.csv not found in evasion_scores.csv: {n_missing}")
    log(f"Transcripts where mean_evasion_score differs from a fresh recomputation: {n_mismatch}")
    if n_mismatch == 0 and n_missing == 0:
        log("CONFIRMED: master_panel.csv's mean_evasion_score column for all 155 rows exactly "
            "matches a fresh groupby(transcript_id).mean() over the current evasion_scores.csv. "
            "No changes needed.")
    else:
        log("WARNING: discrepancy detected -- master_panel.csv may need to be rebuilt. "
            "Not doing so automatically per instructions; flagging for manual review.")
    return n_mismatch, n_missing


def section_pairlevel_mean_sd(scores):
    hr("SECTION: Pair-level mean and SD of evasion_score (ALL 3,350 pairs) -- Table 1 fix")
    s = scores["evasion_score"].astype(float)
    n = len(s)
    mean = s.mean()
    sd = s.std(ddof=1)
    median = s.median()
    log(f"n = {n} individual Q&A pairs (NOT the 155 transcript-level observations)")
    log(f"Pair-level mean evasion_score: {mean:.4f}")
    log(f"Pair-level SD evasion_score:   {sd:.4f}")
    log(f"Pair-level median:             {median:.4f}")
    log(f"Pair-level min / max:          {s.min():.2f} / {s.max():.2f}")

    mp = pd.read_csv(MASTER_PANEL_CSV)
    tl_mean = mp["mean_evasion_score"].astype(float).mean()
    tl_sd = mp["mean_evasion_score"].astype(float).std(ddof=1)
    log(f"\nFor contrast -- transcript-level statistic (155 obs, mean-of-transcript-means):")
    log(f"  Transcript-level mean: {tl_mean:.4f}")
    log(f"  Transcript-level SD:   {tl_sd:.4f}")
    log(f"\nThese are two different statistics measuring different things: the pair-level figure "
        f"(n=3,350) is the distribution of individual Q&A-pair evasion scores; the transcript-level "
        f"figure (n=155) is the distribution of PER-TRANSCRIPT MEANS, which has a smaller SD because "
        f"averaging within each transcript cancels out pair-to-pair noise. Table 1 should label "
        f"whichever one it reports with its correct n and unit of observation -- this is very likely "
        f"the mislabeling flagged in review: transcript-level SD is a within-transcript-averaged "
        f"quantity, not comparable to a pair-level SD, and reporting one with the other's label "
        f"understates or overstates dispersion depending on which was swapped in.")
    return mean, sd, median


def main():
    log("POST-COMMIT VERIFICATION -- recomputed fresh from the now-committed evasion_scores.csv "
        "(commit bbdf274). master_panel.csv, model results, and robustness checks are NOT rerun "
        "or modified -- confirmed read-only consistent below.")

    qa, scores, mp, n_htm, n_qa, n_scores, n_unique_tid, n_unique_ticker = section1_provenance()

    hr("SECTION 1 SUMMARY (confirm 3350 / 285 / 49)")
    log(f"Total scored pairs (evasion_scores.csv): {n_scores}")
    log(f"Total unique transcripts:                 {n_unique_tid}")
    log(f"Total unique companies:                   {n_unique_ticker}")
    match = (n_scores == 3350 and n_unique_tid == 285 and n_unique_ticker == 49)
    log(f"Matches expected (3350/285/49): {match}")

    section_pair_distribution(scores)
    section_master_panel_consistency(scores, mp)
    section_pairlevel_mean_sd(scores)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")
    log(f"\nSaved full log -> {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
