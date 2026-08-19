"""
Factor structure of the four evasion dimensions -- Task B2 of the sprint.

The advisor asked for parallel analysis plus a two-factor EFA with promax
rotation, expecting to confirm a predicted "directness factor + hedging factor"
structure.

There is a hard obstacle, and this script reports it rather than working
around it. Exploratory factor analysis with p indicators and m factors has

    df = 0.5 * ((p - m)^2 - (p + m))

degrees of freedom. With p = 4 dimensions and m = 2 factors that is df = -1:
the two-factor model is UNDER-IDENTIFIED. An estimator will still return
numbers, but they are one of infinitely many equivalent solutions, so they
cannot confirm or refute a hypothesised structure. Four indicators is simply
not enough to identify two factors; you need at least five (m=2, p=5 -> df=1).

What this script therefore reports:
  1. Parallel analysis        -- how many factors the data actually supports
  2. One-factor EFA           -- identified (df = 2), so genuinely estimable
  3. The two-factor solution  -- shown, but explicitly labelled under-identified
                                 and NOT usable as confirmation

Output: results/factor_structure.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCORES = BASE_DIR / "data" / "parsed_qa" / "evasion_scores.csv"
RESULTS_DIR = BASE_DIR / "results"
DIMS = ["non_responsiveness", "vagueness", "deflection", "hedging"]
RNG = np.random.default_rng(20260819)


def efa_df(p, m):
    return 0.5 * ((p - m) ** 2 - (p + m))


def parallel_analysis(X, n_iter=1000):
    """Horn's parallel analysis: keep factors whose eigenvalue beats the 95th
    percentile of eigenvalues from random data of the same shape."""
    n, p = X.shape
    real = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))[::-1]
    sim = np.empty((n_iter, p))
    for i in range(n_iter):
        R = RNG.standard_normal((n, p))
        sim[i] = np.linalg.eigvalsh(np.corrcoef(R, rowvar=False))[::-1]
    return real, sim.mean(axis=0), np.percentile(sim, 95, axis=0)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import (
        calculate_bartlett_sphericity, calculate_kmo)

    df = pd.read_csv(SCORES)
    X = df[DIMS].apply(pd.to_numeric, errors="coerce").dropna()
    Xv = X.to_numpy(float)

    L = []
    def log(s=""):
        L.append(s); print(s)

    log("Factor structure of the four evasion dimensions")
    log("=" * 70)
    log(f"n = {len(X):,} scored pairs, p = {len(DIMS)} dimensions")
    log("")

    log("Correlation matrix")
    log("-" * 70)
    log(X.corr().round(3).to_string())
    log("")

    chi2, p_bart = calculate_bartlett_sphericity(X)
    kmo_per, kmo_all = calculate_kmo(X)
    log(f"Bartlett sphericity : chi2 = {chi2:,.1f}, p = {p_bart:.3g} "
        f"({'factorable' if p_bart < .05 else 'NOT factorable'})")
    log(f"KMO overall         : {kmo_all:.3f} "
        f"({'adequate' if kmo_all > .6 else 'marginal/inadequate'})")
    log("")

    log("-" * 70)
    log("1. PARALLEL ANALYSIS (Horn) -- how many factors are supported?")
    log("-" * 70)
    real, sim_mean, sim_95 = parallel_analysis(Xv)
    log(f"{'factor':<9}{'eigenvalue':>13}{'random mean':>14}{'random p95':>13}   retain?")
    keep = 0
    for i in range(len(real)):
        r = real[i] > sim_95[i]
        keep += int(r)
        log(f"{i+1:<9}{real[i]:>13.3f}{sim_mean[i]:>14.3f}{sim_95[i]:>13.3f}"
            f"   {'YES' if r else 'no'}")
    log("")
    log(f"  Factors retained: {keep}")
    log("")

    log("-" * 70)
    log("2. ONE-FACTOR EFA (identified: df = %.0f)" % efa_df(4, 1))
    log("-" * 70)
    fa1 = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
    fa1.fit(X)
    load1 = pd.DataFrame(fa1.loadings_, index=DIMS, columns=["F1"])
    load1["communality"] = fa1.get_communalities()
    log(load1.round(3).to_string())
    ev, _ = fa1.get_eigenvalues()
    log("")
    log(f"  variance explained by F1: {fa1.get_factor_variance()[1][0]*100:.1f}%")
    log("")

    log("-" * 70)
    log("3. TWO-FACTOR SOLUTION -- UNDER-IDENTIFIED (df = %.0f)" % efa_df(4, 2))
    log("-" * 70)
    log("  Reported for completeness ONLY. With 4 indicators and 2 factors the")
    log("  model has negative degrees of freedom, so this is one of infinitely")
    log("  many equivalent solutions. It cannot confirm a hypothesised structure")
    log("  and must not be presented as if it did.")
    log("")
    try:
        fa2 = FactorAnalyzer(n_factors=2, rotation="promax", method="minres")
        fa2.fit(X)
        load2 = pd.DataFrame(fa2.loadings_, index=DIMS, columns=["F1", "F2"])
        log(load2.round(3).to_string())
    except Exception as exc:
        log(f"  estimator failed, as expected for an unidentified model: {exc}")
    log("")

    log("-" * 70)
    log("WHAT THE DATA ACTUALLY SHOWS")
    log("-" * 70)
    c = X.corr()
    log(f"  Strongest pair : non_responsiveness-vagueness  r = {c.loc['non_responsiveness','vagueness']:.3f}")
    log(f"                   non_responsiveness-deflection r = {c.loc['non_responsiveness','deflection']:.3f}")
    log(f"  Weakest pair   : deflection-hedging            r = {c.loc['deflection','hedging']:.3f}")
    log("")
    log("  Hedging is the least redundant dimension: it correlates only 0.24")
    log("  with deflection while the other three inter-correlate at 0.52-0.73.")
    log("  That is the descriptive pattern behind the predicted 'directness vs")
    log("  hedging' split, and it is reportable as a correlation result.")
    log("")
    if keep <= 1:
        log("  BUT parallel analysis retains only ONE factor. The predicted")
        log("  two-factor structure is NOT confirmed. Report the correlation")
        log("  pattern descriptively and drop the two-factor claim -- with four")
        log("  indicators it is not testable here regardless of what the data")
        log("  looked like. Adding dimensions in future work would make it so.")
    else:
        log(f"  Parallel analysis retains {keep} factors, consistent with the")
        log("  predicted structure -- but note the identification caveat above.")
    log("")

    out = RESULTS_DIR / "factor_structure.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
