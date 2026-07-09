"""
Table 2 bootstrap CI: Config B vs Config C AUROC difference, for BOTH
LogisticRegression and XGBoost, on the current (155-obs) master_panel.csv.

final_verification.py's section5_classification() already computes this
bootstrap CI, but only for XGBoost, and only prints it to stdout -- it is
never written to a results file (ROBUSTNESS_OUT only captures Section 6,
not Section 5's bootstrap line). This script reuses the identical helper
functions from models.py (chronological_split, impute_accounting_features,
CONFIGS, train_logistic_regression, train_xgb_classifier,
bootstrap_auroc_diff) so the split, imputation, and feature sets are exactly
the ones already used everywhere else, and additionally runs the same
bootstrap for LogisticRegression, which no existing script does.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    CONFIGS, bootstrap_auroc_diff, chronological_split,
    impute_accounting_features, prepare_xy, train_logistic_regression,
    train_xgb_classifier,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PANEL_CSV = BASE_DIR / "data" / "features" / "master_panel.csv"
OUTPUT_TXT = BASE_DIR / "results" / "bootstrap_ci_table2.txt"

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def main():
    df = pd.read_csv(MASTER_PANEL_CSV, dtype={"transcript_id": str})
    log(f"Loaded master_panel.csv: {len(df)} rows, {df['company_ticker'].nunique()} companies")

    train, test = chronological_split(df)
    log(f"Chronological 70/30 split: train n={len(train)}, test n={len(test)}")

    train, test = impute_accounting_features(train, test)

    fitted = {}
    for config_name in ("B", "C"):
        feature_cols = CONFIGS[config_name]
        X_train, y_train = prepare_xy(train, feature_cols, "car_direction")
        X_test, y_test = prepare_xy(test, feature_cols, "car_direction")

        lr_model, scaler, lr_pred, lr_proba = train_logistic_regression(X_train, y_train, X_test)
        xgb_model, xgb_pred, xgb_proba = train_xgb_classifier(X_train, y_train, X_test)

        fitted[config_name] = {"y_test": y_test, "lr_proba": lr_proba, "xgb_proba": xgb_proba}

    log("\n" + "=" * 70)
    log("TABLE 2: Bootstrap 95% CI, Config C - Config B AUROC difference")
    log("=" * 70)

    results = {}
    for model_type, proba_key in (("LogisticRegression", "lr_proba"), ("XGBoost", "xgb_proba")):
        y_test = fitted["B"]["y_test"]
        proba_b = fitted["B"][proba_key]
        proba_c = fitted["C"][proba_key]

        auroc_b = roc_auc_score(y_test, proba_b)
        auroc_c = roc_auc_score(y_test, proba_c)
        raw_diff = auroc_c - auroc_b

        diffs, ci_low, ci_high, skipped = bootstrap_auroc_diff(y_test, proba_b, proba_c)

        log(f"\n--- {model_type} ---")
        log(f"AUROC B: {auroc_b:.4f}")
        log(f"AUROC C: {auroc_c:.4f}")
        log(f"Point estimate (C - B): {raw_diff:+.4f}")
        log(f"Bootstrap: 1000 resamples ({skipped} skipped for landing on a single class)")
        log(f"95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
        log(f"  --> {'CI includes 0: not significant' if ci_low <= 0 <= ci_high else 'CI excludes 0: significant'}")

        results[model_type] = {"point_estimate": raw_diff, "ci_low": ci_low, "ci_high": ci_high}

    log("\n" + "=" * 70)
    log("SUMMARY (for Table 2)")
    log("=" * 70)
    for model_type, r in results.items():
        log(f"{model_type:20} point estimate = {r['point_estimate']:+.4f}   "
            f"95% CI = [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")
    log(f"\nSaved full log -> {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
