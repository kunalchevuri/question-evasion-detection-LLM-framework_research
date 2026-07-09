"""
Config B / Config C logistic regression, refit with statsmodels to get
firm-clustered standard errors (cluster on company_ticker), compared against
naive (non-clustered) standard errors on the same fitted coefficients.

This is an INFERENCE exercise (are the coefficients statistically reliable
given repeated observations per company?), not the out-of-sample PREDICTION
exercise in models.py -- so unlike models.py's chronological train/test
split with train-only median imputation, this fits once on the full
155-row panel with full-sample median imputation for missing accounting
features. Reuses CONFIGS/ACCOUNTING_FEATURES/LM_FEATURES from models.py so
the feature sets are identical to the ones already used elsewhere.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CONFIGS, ACCOUNTING_FEATURES  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PANEL_CSV = BASE_DIR / "data" / "features" / "master_panel.csv"
OUTPUT_TXT = BASE_DIR / "results" / "clustered_se_results.txt"

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def fit_logit(df, feature_cols, cluster_groups=None):
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    X_scaled = sm.add_constant(X_scaled)
    y = df["car_direction"]

    model = sm.Logit(y, X_scaled)
    if cluster_groups is not None:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_groups}, disp=0)
    else:
        result = model.fit(disp=0)
    return result


def main():
    df = pd.read_csv(MASTER_PANEL_CSV, dtype={"transcript_id": str})
    log(f"Loaded master_panel.csv: {len(df)} rows, {df['company_ticker'].nunique()} companies")
    log("NOTE: this is an inference exercise on the FULL panel (single full-sample median "
        "imputation), not the chronological train/test prediction split used in models.py -- "
        "that split is for out-of-sample AUROC, not for estimating population parameters.\n")

    df = df.copy()
    for col in ACCOUNTING_FEATURES:
        median_val = df[col].median()
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        log(f"  Imputed {col}: full-sample median={median_val:.4f}, {n_missing} row(s) filled")

    for config_name in ("B", "C"):
        log("\n" + "=" * 70)
        log(f"CONFIG {config_name}: {CONFIGS[config_name]}")
        log("=" * 70)

        feature_cols = CONFIGS[config_name]

        naive = fit_logit(df, feature_cols, cluster_groups=None)
        clustered = fit_logit(df, feature_cols, cluster_groups=df["company_ticker"])

        log("\n--- Naive (non-clustered) standard errors ---")
        log(naive.summary().tables[1].as_text())

        log("\n--- Firm-clustered standard errors (cluster on company_ticker) ---")
        log(clustered.summary().tables[1].as_text())

        if config_name == "C":
            feat = "mean_evasion_score"
            naive_coef = naive.params[feat]
            naive_se = naive.bse[feat]
            naive_z = naive.tvalues[feat]
            naive_p = naive.pvalues[feat]

            clus_coef = clustered.params[feat]
            clus_se = clustered.bse[feat]
            clus_z = clustered.tvalues[feat]
            clus_p = clustered.pvalues[feat]

            log("\n" + "=" * 70)
            log(f"COEFFICIENT OF INTEREST: {feat} (Config C)")
            log("=" * 70)
            log(f"{'':20}{'Coefficient':>14}{'Std Err':>12}{'z':>10}{'p-value':>12}")
            log(f"{'Naive':20}{naive_coef:>14.4f}{naive_se:>12.4f}{naive_z:>10.4f}{naive_p:>12.4f}")
            log(f"{'Firm-clustered':20}{clus_coef:>14.4f}{clus_se:>12.4f}{clus_z:>10.4f}{clus_p:>12.4f}")
            log(f"\nSE inflation factor (clustered / naive): {clus_se / naive_se:.2f}x")

            naive_sig = naive_p < 0.05
            clus_sig = clus_p < 0.05
            log(f"\nNaive SE:      p={naive_p:.4f}  -> {'SIGNIFICANT' if naive_sig else 'not significant'} at alpha=0.05")
            log(f"Clustered SE:  p={clus_p:.4f}  -> {'SIGNIFICANT' if clus_sig else 'not significant'} at alpha=0.05")

            log("\n" + "=" * 70)
            log("SUMMARY")
            log("=" * 70)
            if naive_sig and not clus_sig:
                log(
                    "Accounting for firm clustering CHANGES the conclusion: the evasion coefficient "
                    "looks significant with naive (i.i.d.-assumed) standard errors, but loses "
                    "significance once repeated observations per company are properly accounted for. "
                    "This matches the concentration concern already on record (58.7% of observations "
                    "from 5 companies) -- naive SEs were overstating precision."
                )
            elif not naive_sig and not clus_sig:
                log(
                    "The evasion coefficient is NOT significant under either naive or clustered "
                    "standard errors -- consistent with the null correlation and null classification "
                    "results already established elsewhere. Clustering does not change the conclusion "
                    "here because there was no naive significance to lose."
                )
            elif naive_sig and clus_sig:
                log(
                    "The evasion coefficient remains significant under both naive and clustered "
                    "standard errors, though the clustered SE is larger -- the significance is more "
                    "robust than a single naive fit would suggest is necessary to check, but note it "
                    "was significant either way here."
                )
            else:
                log(
                    "The evasion coefficient is significant with clustered SEs but not naive SEs -- "
                    "this is an unusual direction (clustering almost always widens, not narrows, SEs) "
                    "and warrants double-checking the cluster specification."
                )

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")
    log(f"\nSaved full log -> {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
