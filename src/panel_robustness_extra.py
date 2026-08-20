"""
Micro-robustness on the financial panel -- the B3 "if time permits" extras.

Two were requested:

  1. Exclude imputed-margin observations. Gross margin is observed for 109 of
     155 panel rows and operating margin for 113; the rest are filled with
     training-set medians. If the null depended on that imputation it would be
     an artifact, so we re-run the central correlations on the subset where
     both margins are genuinely observed. RUN -- see output.

  2. Fama-French-adjusted CAR. NOT RUN, and the reason is recorded here rather
     than left implicit. The Ken French daily factor file downloads cleanly
     (26,274 rows through 2026-06-30), so that half is not the obstacle. The
     obstacle is that data/features/master_panel.csv stores only the finished
     car_3day value: it carries no raw daily returns, no per-observation event
     date (only filing_date, with car_date_estimated flagging which
     announcement dates were inferred), and no cached price series exists in
     the repository. Producing a risk-adjusted CAR therefore means
     re-deriving the entire event-study pipeline -- re-downloading prices for
     32 firms over 2015-2023, reconstructing event windows, and estimating
     factor loadings on a pre-event window -- which is not the micro-check the
     instruction contemplated. Recorded as attempted and cut, per the
     instruction to drop it if it is not trivially clean.

Output: results/panel_robustness_extra.txt
"""

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent.parent
PANEL = BASE / "data" / "features" / "master_panel.csv"
OUT = BASE / "results" / "panel_robustness_extra.txt"

FEATURES = ["mean_evasion_score", "evasion_variance", "max_evasion_score"]


def main():
    df = pd.read_csv(PANEL).dropna(subset=["car_3day"])
    complete = df["gross_margin"].notna() & df["operating_margin"].notna()
    sub = df[complete]

    L = []
    def log(s=""):
        L.append(s); print(s)

    log("Financial-panel micro-robustness (B3 extras)")
    log("=" * 72)
    log(f"full panel                       : n = {len(df)}")
    log(f"gross margin observed            : {df['gross_margin'].notna().sum()} of {len(df)}")
    log(f"operating margin observed        : {df['operating_margin'].notna().sum()} of {len(df)}")
    log(f"both observed (no imputation)    : n = {len(sub)}")
    log("")
    log("-" * 72)
    log("1. EXCLUDING IMPUTED-MARGIN OBSERVATIONS")
    log("-" * 72)
    log("Does the null survive when we keep only rows whose accounting features")
    log("were actually observed rather than filled with training-set medians?")
    log("")
    log(f"{'feature':<22}{'full panel':>24}{'complete margins':>26}")
    all_ns = True
    for f in FEATURES:
        a = df.dropna(subset=[f])
        b = sub.dropna(subset=[f])
        r1, p1 = pearsonr(a[f], a["car_3day"])
        r2, p2 = pearsonr(b[f], b["car_3day"])
        if p2 < 0.05:
            all_ns = False
        log(f"  {f:<20}r={r1:+.4f} p={p1:.4f}   r={r2:+.4f} p={p2:.4f}")
    log("")
    for f in FEATURES:
        b = sub.dropna(subset=[f])
        rho, p = spearmanr(b[f], b["car_3day"])
        log(f"  {f:<20}Spearman rho={rho:+.4f} p={p:.4f}  (complete margins)")
    log("")
    log("  RESULT: " + ("the null holds; no evasion measure reaches significance "
                        "on the complete-margin subset."
                        if all_ns else
                        "a measure becomes significant -- INVESTIGATE before reporting."))
    log("")
    log("-" * 72)
    log("2. FAMA-FRENCH-ADJUSTED CAR -- ATTEMPTED, CUT")
    log("-" * 72)
    log("  Ken French daily factors download cleanly (26,274 rows, through")
    log("  2026-06-30), so factor availability is not the blocker. The panel")
    log("  stores only the finished car_3day: no raw daily returns, no event")
    log("  date, no cached prices. A risk-adjusted CAR would require re-running")
    log("  the whole event study rather than a micro-check, so it is cut per")
    log("  the instruction to drop anything not trivially clean. The existing")
    log("  CAR remains market-adjusted, which the Limitations section states.")
    log("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
