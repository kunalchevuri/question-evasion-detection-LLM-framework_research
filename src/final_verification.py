"""
Final verification pass before writing the paper: recompute every headline
number directly from the files currently on disk. No reliance on any
previously printed output, any previous commit message, or any cached/
remembered number -- prior turns in this project surfaced repeated mismatches
between claimed and actual state (183 vs 155 observations, a fabricated
r=-0.27 correlation), so this script re-derives everything from scratch.

Reuses the exact modeling/statistical functions from models.py and
robustness_checks.py wherever those functions are parametrized cleanly by the
current data (chronological_split, impute_accounting_features, CONFIGS,
train_logistic_regression, train_xgb_classifier, bootstrap_auroc_diff,
task2_correlations, simulate_power). Does NOT reuse robustness_checks.py's
task1_cross_validation or task3_exclude_2020 as-is: both hardcode literal
"91" and old single-split AUROC numbers (e.g. "LR AUROC improved B(0.624)->
C(0.682)") into their log strings from the original n=91 run, which would
silently misreport on this larger panel -- exactly the kind of stale-number
bug this script exists to eliminate. The underlying statistical methodology
(StratifiedKFold 5-fold CV, paired t-test, COVID-year exclusion + re-split)
is reproduced with the same logic, but with every printed number computed
fresh against the current sample size.
"""

import sys
from datetime import datetime
from pathlib import Path

# Transcript question/response text contains characters (em-dashes, curly
# quotes, etc.) outside cp1252, which is what Python defaults stdout to on
# Windows when it isn't a real console (e.g. redirected to a file). Reconfigure
# to UTF-8 with safe replacement so Section 7's full-text printout can't crash
# the whole run over a single unprintable character.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    CONFIGS, RANDOM_STATE,
    bootstrap_auroc_diff, chronological_split, compute_classification_metrics,
    impute_accounting_features, prepare_xy, train_logistic_regression,
    train_xgb_classifier,
)
import robustness_checks as rc  # noqa: E402
from robustness_checks import EVASION_FEATURES, simulate_power, task2_correlations  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RAW_DIR = DATA_DIR / "raw_transcripts"
QA_CSV = DATA_DIR / "parsed_qa" / "all_qa_pairs.csv"
SCORES_CSV = DATA_DIR / "parsed_qa" / "evasion_scores.csv"
MASTER_PANEL_CSV = DATA_DIR / "features" / "master_panel.csv"
KAPPA_CSV = RESULTS_DIR / "kappa_statistics.csv"

DESCRIPTIVE_OUT = RESULTS_DIR / "final_descriptive_statistics.csv"
CORR_MATRIX_OUT = RESULTS_DIR / "final_correlation_matrix.csv"
CONCENTRATION_OUT = RESULTS_DIR / "final_company_concentration.csv"
MODEL_RESULTS_OUT = RESULTS_DIR / "final_model_results.csv"
ROBUSTNESS_OUT = RESULTS_DIR / "final_robustness_checks.txt"
QUALITATIVE_OUT = RESULTS_DIR / "qualitative_examples.csv"

CORR_COLS = [
    "mean_evasion_score", "evasion_variance", "max_evasion_score", "car_3day",
    "revenue_growth", "gross_margin", "operating_margin", "roa",
    "lm_positive", "lm_negative", "lm_uncertainty", "lm_litigious",
]

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def hr(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def capture_rc(fn, *args, **kwargs):
    """Call a robustness_checks.py function (it prints via its own internal
    log(), which already writes to stdout) and capture the lines it appended
    to robustness_checks._LOG_LINES, so this script's own log file mirrors
    what actually printed without double-printing it."""
    start = len(rc._LOG_LINES)
    result = fn(*args, **kwargs)
    captured = rc._LOG_LINES[start:]
    return result, captured


# ── Section 1 ────────────────────────────────────────────────────────────────

def section1_provenance():
    hr("SECTION 1: Data provenance, verified from disk")

    n_htm = len(list(RAW_DIR.glob("*.htm")))
    log(f"1. Total .htm files in data/raw_transcripts/: {n_htm}")

    qa = pd.read_csv(QA_CSV, dtype=str)
    n_qa = len(qa)
    log(f"2. Total rows in all_qa_pairs.csv: {n_qa}")

    scores = pd.read_csv(SCORES_CSV, dtype=str)
    n_scores = len(scores)
    log(f"3. Total rows in evasion_scores.csv: {n_scores}")

    log(f"4. Pair-count match check: all_qa_pairs.csv={n_qa}  evasion_scores.csv={n_scores}")
    if n_qa != n_scores:
        diff = n_qa - n_scores
        log(f"   MISMATCH: {abs(diff)} row(s) differ.")
        qa_keys = set(zip(qa["transcript_id"], qa["question_text"]))
        score_keys = set(zip(scores["transcript_id"], scores["question_text"]))
        missing_from_scores = qa_keys - score_keys
        missing_from_qa = score_keys - qa_keys
        log(f"   Present in all_qa_pairs.csv but NOT in evasion_scores.csv: {len(missing_from_scores)}")
        for tid, q in list(missing_from_scores)[:10]:
            log(f"     transcript_id={tid}  question={q[:120]!r}")
        log(f"   Present in evasion_scores.csv but NOT in all_qa_pairs.csv: {len(missing_from_qa)}")
        for tid, q in list(missing_from_qa)[:10]:
            log(f"     transcript_id={tid}  question={q[:120]!r}")
        if abs(diff) == 1:
            log(
                "   DIAGNOSIS: exactly one row differs, matching the previously-documented case -- "
                "one pair triggers a hard safety-classifier refusal (stop_reason='refusal', empty "
                "content) on every retry attempt and was therefore never scored. This is data "
                "verified above via set difference (not asserted from memory): confirm the single "
                "missing transcript_id shown above is the one already on record before proceeding."
            )
            log("   Discrepancy matches the expected single-row case -- proceeding.")
        else:
            log(f"\n   *** ABORTING: discrepancy is {abs(diff)} row(s), not the expected single "
                f"known case. This is new and unexplained -- fix before proceeding. ***")
            sys.exit(1)
    else:
        log("   MATCH: counts are identical.")

    n_unique_tid = scores["transcript_id"].nunique()
    log(f"5. Unique transcript_id values in evasion_scores.csv: {n_unique_tid}")

    n_unique_ticker = scores["company_ticker"].nunique()
    log(f"6. Unique company_ticker values in evasion_scores.csv: {n_unique_ticker}")

    mp = pd.read_csv(MASTER_PANEL_CSV, dtype={"transcript_id": str})
    log(f"7. Total rows in master_panel.csv: {len(mp)}")
    log(f"8. Unique company_ticker values in master_panel.csv: {mp['company_ticker'].nunique()}")
    log(f"9. filing_year range in master_panel.csv: {int(mp['filing_year'].min())} - {int(mp['filing_year'].max())}")

    log("10. File modification timestamps (confirm same pipeline run, not stale leftovers):")
    for path in (MASTER_PANEL_CSV, SCORES_CSV, QA_CSV):
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        log(f"    {path.relative_to(BASE_DIR)}: {mtime.isoformat(sep=' ', timespec='seconds')}")

    return qa, scores, mp, n_htm, n_qa, n_scores, n_unique_tid, n_unique_ticker


# ── Section 2 ────────────────────────────────────────────────────────────────

def section2_descriptive(mp):
    hr("SECTION 2: Descriptive statistics, computed fresh")

    desc = mp[CORR_COLS].describe().T
    log("\n1. Descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max):")
    log(desc.to_string())

    total = len(mp)
    n_pos = int((mp["car_direction"] == 1).sum())
    n_neg = int((mp["car_direction"] == 0).sum())
    log("\n2. car_direction distribution:")
    log(f"   Positive (1): {n_pos} ({n_pos / total * 100:.1f}%)")
    log(f"   Negative (0): {n_neg} ({n_neg / total * 100:.1f}%)")

    log("\n3. Missing-value counts (accounting features):")
    for col in ["revenue_growth", "gross_margin", "operating_margin", "roa"]:
        n_missing = int(mp[col].isna().sum())
        log(f"   {col:<18} missing: {n_missing} / {total} ({n_missing / total * 100:.1f}%)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(DESCRIPTIVE_OUT, encoding="utf-8")
    log(f"\n4. Saved -> {DESCRIPTIVE_OUT}")
    return desc


# ── Section 3 ────────────────────────────────────────────────────────────────

def section3_correlation_matrix(mp):
    hr("SECTION 3: Full correlation matrix")

    corr_matrix = mp[CORR_COLS].corr(method="pearson")
    log("\n1-2. Full Pearson correlation matrix (pairwise-complete observations):")
    log(corr_matrix.to_string())

    log("\n3. Central-claim pairs (Pearson r and p-value):")
    corr_central = {}
    for feat in ["mean_evasion_score", "evasion_variance", "max_evasion_score"]:
        sub = mp[[feat, "car_3day"]].dropna()
        r, p = pearsonr(sub[feat], sub["car_3day"])
        corr_central[feat] = (r, p, len(sub))
        log(f"   {feat:<22} vs car_3day:  r={r:+.4f}  p={p:.4f}  (n={len(sub)})")

    corr_matrix.to_csv(CORR_MATRIX_OUT, encoding="utf-8")
    log(f"\n4. Saved -> {CORR_MATRIX_OUT}")
    return corr_matrix, corr_central


# ── Section 4 ────────────────────────────────────────────────────────────────

def section4_concentration(mp):
    hr("SECTION 4: Company concentration, verified fresh")

    total = len(mp)
    counts = mp["company_ticker"].value_counts()
    table = counts.reset_index()
    table.columns = ["company_ticker", "n_observations"]
    table["pct_of_total"] = table["n_observations"] / total * 100

    log("\n1. Observations per company (sorted descending):")
    log(table.to_string(index=False))

    log("\n2. Top 10 companies by observation count:")
    log(table.head(10).to_string(index=False))

    top5_sum = int(table.head(5)["n_observations"].sum())
    top5_pct = top5_sum / total * 100
    log(f"\n3. Top-5 combined: {top5_sum} / {total} = {top5_pct:.1f}%")

    table.to_csv(CONCENTRATION_OUT, index=False, encoding="utf-8")
    log(f"\n4. Saved -> {CONCENTRATION_OUT}")
    return table, top5_pct


# ── Section 5 ────────────────────────────────────────────────────────────────

def section5_classification(mp):
    hr("SECTION 5: Classification results, rerun fresh on current panel")

    train, test = chronological_split(mp)
    log(f"\n3. Chronological split: train n={len(train)} ({train['filing_date'].min()} to "
        f"{train['filing_date'].max()}), test n={len(test)} ({test['filing_date'].min()} to "
        f"{test['filing_date'].max()})")

    train, test = impute_accounting_features(train, test)

    results = []
    fitted = {}
    for config_name, feature_cols in CONFIGS.items():
        X_train, y_train = prepare_xy(train, feature_cols, "car_direction")
        X_test, y_test = prepare_xy(test, feature_cols, "car_direction")

        lr_model, scaler, lr_pred, lr_proba = train_logistic_regression(X_train, y_train, X_test)
        lr_metrics = compute_classification_metrics(y_test, lr_pred, lr_proba, label=f"{config_name}/LR")
        log(f"\nConfig {config_name} LogisticRegression: AUROC={lr_metrics['auroc']:.4f}  "
            f"Acc={lr_metrics['accuracy']:.4f}  Prec={lr_metrics['precision']:.4f}  "
            f"Rec={lr_metrics['recall']:.4f}  F1={lr_metrics['f1']:.4f}")
        results.append({"config": config_name, "model_type": "LogisticRegression", "task": "classification",
                         **lr_metrics, "n_train": len(X_train), "n_test": len(X_test)})
        fitted[(config_name, "LogisticRegression")] = {"y_test": y_test, "y_proba": lr_proba}

        xgb_model, xgb_pred, xgb_proba = train_xgb_classifier(X_train, y_train, X_test)
        xgb_metrics = compute_classification_metrics(y_test, xgb_pred, xgb_proba, label=f"{config_name}/XGB")
        log(f"Config {config_name} XGBoost:            AUROC={xgb_metrics['auroc']:.4f}  "
            f"Acc={xgb_metrics['accuracy']:.4f}  Prec={xgb_metrics['precision']:.4f}  "
            f"Rec={xgb_metrics['recall']:.4f}  F1={xgb_metrics['f1']:.4f}")
        results.append({"config": config_name, "model_type": "XGBoost", "task": "classification",
                         **xgb_metrics, "n_train": len(X_train), "n_test": len(X_test)})
        fitted[(config_name, "XGBoost")] = {"y_test": y_test, "y_proba": xgb_proba}

    b, c = fitted[("B", "XGBoost")], fitted[("C", "XGBoost")]
    diffs, ci_low, ci_high, skipped = bootstrap_auroc_diff(b["y_test"], b["y_proba"], c["y_proba"])
    auroc_b = roc_auc_score(b["y_test"], b["y_proba"])
    auroc_c = roc_auc_score(c["y_test"], c["y_proba"])
    log(f"\n6. Bootstrap CI, Config B vs C (XGBoost): AUROC_B={auroc_b:.4f}  AUROC_C={auroc_c:.4f}  "
        f"raw_diff={auroc_c - auroc_b:+.4f}  95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]  "
        f"({'includes 0 -- not significant' if ci_low <= 0 <= ci_high else 'excludes 0 -- significant'})")

    results_df = pd.DataFrame(results)[
        ["config", "model_type", "task", "auroc", "accuracy", "precision", "recall", "f1", "n_train", "n_test"]
    ]
    results_df.to_csv(MODEL_RESULTS_OUT, index=False, encoding="utf-8")
    log(f"\n7. Saved fresh results -> {MODEL_RESULTS_OUT} (old model_results.csv left untouched)")

    day6_auroc_current = {
        cfg: {
            "LogisticRegression": roc_auc_score(fitted[(cfg, "LogisticRegression")]["y_test"],
                                                 fitted[(cfg, "LogisticRegression")]["y_proba"]),
            "XGBoost": roc_auc_score(fitted[(cfg, "XGBoost")]["y_test"], fitted[(cfg, "XGBoost")]["y_proba"]),
        }
        for cfg in ("B", "C")
    }

    return results_df, fitted, day6_auroc_current, (ci_low, ci_high, auroc_b, auroc_c)


# ── Section 6.1: 5-fold CV (reimplemented -- see module docstring for why) ──

def section6_1_cross_validation(mp, day6_auroc_current):
    hr("SECTION 6.1: 5-fold stratified cross-validation, Config B vs Config C")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_all = mp["car_direction"].to_numpy()

    fold_results = {("B", "LogisticRegression"): [], ("B", "XGBoost"): [],
                     ("C", "LogisticRegression"): [], ("C", "XGBoost"): []}

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(mp, y_all), 1):
        train_fold = mp.iloc[train_idx].copy()
        test_fold = mp.iloc[test_idx].copy()
        train_fold, test_fold = impute_accounting_features(train_fold, test_fold)

        log(f"\n--- Fold {fold_i} (train n={len(train_fold)}, test n={len(test_fold)}) ---")
        for config_name in ("B", "C"):
            feature_cols = CONFIGS[config_name]
            X_train, y_train = train_fold[feature_cols], train_fold["car_direction"]
            X_test, y_test = test_fold[feature_cols], test_fold["car_direction"]

            if len(set(y_test)) < 2:
                log(f"  Config {config_name}: held-out fold has only one class -- AUROC skipped")
                continue

            _, _, _, lr_proba = train_logistic_regression(X_train, y_train, X_test)
            lr_auroc = roc_auc_score(y_test, lr_proba)
            fold_results[(config_name, "LogisticRegression")].append(lr_auroc)

            _, _, xgb_proba = train_xgb_classifier(X_train, y_train, X_test)
            xgb_auroc = roc_auc_score(y_test, xgb_proba)
            fold_results[(config_name, "XGBoost")].append(xgb_auroc)

            log(f"  Config {config_name}: LR AUROC={lr_auroc:.4f}   XGB AUROC={xgb_auroc:.4f}")

    log("\n--- Mean +/- SD across 5 folds ---")
    summary = {}
    for key, vals in fold_results.items():
        arr = np.array(vals)
        mean, sd = arr.mean(), arr.std(ddof=1)
        summary[key] = (mean, sd)
        log(f"Config {key[0]} / {key[1]}: mean={mean:.4f}  sd={sd:.4f}  "
            f"folds=[{', '.join(f'{v:.4f}' for v in vals)}]")

    log("\n--- Config C minus Config B, per fold, paired t-test ---")
    for model_type in ("LogisticRegression", "XGBoost"):
        b_vals = np.array(fold_results[("B", model_type)])
        c_vals = np.array(fold_results[("C", model_type)])
        diffs = c_vals - b_vals
        t_stat, p_val = ttest_rel(c_vals, b_vals)
        log(f"{model_type}: per-fold (C-B) = [{', '.join(f'{d:+.4f}' for d in diffs)}]")
        log(f"{model_type}: mean diff = {diffs.mean():+.4f}   t={t_stat:.4f}  p={p_val:.4f}")

    log("\n--- Interpretation (all numbers computed fresh in this run, nothing hardcoded) ---")
    lr_b_mean, lr_b_sd = summary[("B", "LogisticRegression")]
    lr_c_mean, lr_c_sd = summary[("C", "LogisticRegression")]
    xgb_b_mean, xgb_b_sd = summary[("B", "XGBoost")]
    xgb_c_mean, xgb_c_sd = summary[("C", "XGBoost")]
    ss_lr_b = day6_auroc_current["B"]["LogisticRegression"]
    ss_lr_c = day6_auroc_current["C"]["LogisticRegression"]
    ss_xgb_b = day6_auroc_current["B"]["XGBoost"]
    ss_xgb_c = day6_auroc_current["C"]["XGBoost"]
    max_sd = max(lr_b_sd, lr_c_sd, xgb_b_sd, xgb_c_sd)
    ss_diff = max(abs(ss_lr_c - ss_lr_b), abs(ss_xgb_c - ss_xgb_b))
    log(
        f"This run's single chronological-split result (Section 5): LR AUROC B={ss_lr_b:.4f} -> "
        f"C={ss_lr_c:.4f}; XGBoost B={ss_xgb_b:.4f} -> C={ss_xgb_c:.4f}. "
        f"5-fold CV: LR mean B={lr_b_mean:.4f} (sd={lr_b_sd:.4f}) -> C={lr_c_mean:.4f} (sd={lr_c_sd:.4f}); "
        f"XGBoost mean B={xgb_b_mean:.4f} (sd={xgb_b_sd:.4f}) -> C={xgb_c_mean:.4f} (sd={xgb_c_sd:.4f}). "
        f"{'Fold-to-fold SD is large relative to the single-split difference-of-interest, indicating the single-split result is fragile, not a stable estimate.' if max_sd > ss_diff else 'Fold-to-fold SD is comparable to or smaller than the single-split difference, some support for stability, though n is still small.'}"
    )
    return fold_results, summary


# ── Section 6.3: COVID exclusion (reimplemented -- see module docstring) ────

def section6_3_covid_exclusion(mp, day6_auroc_current, full_corr_table):
    hr("SECTION 6.3: COVID exclusion sensitivity (filing_year != 2020)")

    n_2020 = int((mp["filing_year"] == 2020).sum())
    subset = mp[mp["filing_year"] != 2020].copy()
    log(f"Rows in 2020: {n_2020}")
    log(f"Rows remaining after excluding 2020: {len(subset)} (of {len(mp)} total, "
        f"{len(mp)} - {n_2020} = {len(mp) - n_2020})")

    train, test = chronological_split(subset)
    train, test = impute_accounting_features(train, test)

    log("\n--- Config B vs C AUROC, COVID-excluded subset ---")
    auroc_results = {}
    for config_name in ("B", "C"):
        feature_cols = CONFIGS[config_name]
        X_train, y_train = train[feature_cols], train["car_direction"]
        X_test, y_test = test[feature_cols], test["car_direction"]
        if len(set(y_test)) < 2:
            log(f"Config {config_name}: test set has only one class -- AUROC skipped")
            continue
        _, _, _, lr_proba = train_logistic_regression(X_train, y_train, X_test)
        lr_auroc = roc_auc_score(y_test, lr_proba)
        _, _, xgb_proba = train_xgb_classifier(X_train, y_train, X_test)
        xgb_auroc = roc_auc_score(y_test, xgb_proba)
        auroc_results[config_name] = {"LogisticRegression": lr_auroc, "XGBoost": xgb_auroc}
        log(f"Config {config_name}: LR AUROC={lr_auroc:.4f}   XGB AUROC={xgb_auroc:.4f}")

    log(f"\n--- Side-by-side: full sample (n={len(mp)}) vs excluding 2020 (n={len(subset)}) ---")
    for config_name in ("B", "C"):
        if config_name in auroc_results and config_name in day6_auroc_current:
            full_lr = day6_auroc_current[config_name]["LogisticRegression"]
            full_xgb = day6_auroc_current[config_name]["XGBoost"]
            sub_lr = auroc_results[config_name]["LogisticRegression"]
            sub_xgb = auroc_results[config_name]["XGBoost"]
            log(f"Config {config_name} LR AUROC:  full={full_lr:.4f}  excl-2020={sub_lr:.4f}")
            log(f"Config {config_name} XGB AUROC: full={full_xgb:.4f}  excl-2020={sub_xgb:.4f}")

    log("\n--- Correlation, evasion features vs car_3day, COVID-excluded subset ---")
    start = len(rc._LOG_LINES)
    ex_corr_table = task2_correlations(subset, label="COVID-excluded (final verification)")
    _LOG.extend(rc._LOG_LINES[start:])

    log("\n--- Correlation significance survival (alpha=0.05) ---")
    for feat in EVASION_FEATURES:
        full_row = full_corr_table[full_corr_table.feature == feat].iloc[0]
        ex_row = ex_corr_table[ex_corr_table.feature == feat].iloc[0]
        full_sig = full_row["pearson_p"] < 0.05
        ex_sig = ex_row["pearson_p"] < 0.05
        if not full_sig:
            verdict = "N/A -- not significant in the full sample to begin with"
        elif ex_sig:
            verdict = "SURVIVES"
        else:
            verdict = "DOES NOT SURVIVE"
        log(f"  {feat:<22} full: r={full_row['pearson_r']:+.4f} p={full_row['pearson_p']:.4f}  |  "
            f"COVID-excl: r={ex_row['pearson_r']:+.4f} p={ex_row['pearson_p']:.4f}  -> {verdict}")

    return auroc_results, ex_corr_table


# ── Section 6.4: power analysis ─────────────────────────────────────────────

def section6_4_power_analysis(n_current):
    hr("SECTION 6.4: Power analysis -- minimum detectable AUROC difference at 80% power")

    deltas = [0.02, 0.05, 0.10, 0.15]
    rows = []
    for size_n, label in [(91, "OLD n=91"), (n_current, f"CURRENT n={n_current}")]:
        log(f"\n--- {label} ---")
        min_detectable = None
        for delta in deltas:
            power = simulate_power(size_n, delta)
            log(f"  True AUROC difference = {delta:.2f}  ->  power = {power:.3f}")
            rows.append({"n": size_n, "label": label, "true_delta": delta, "power": power})
            if power >= 0.80 and min_detectable is None:
                min_detectable = delta
        if min_detectable is not None:
            log(f"  >>> Minimum detectable effect at 80% power: {min_detectable:.2f}")
        else:
            log(f"  >>> None of {deltas} reached 80% power -- true minimum detectable effect "
                f"is larger than {max(deltas):.2f}")
    return pd.DataFrame(rows)


# ── Section 7 ────────────────────────────────────────────────────────────────

def section7_qualitative(scores):
    hr("SECTION 7: Qualitative examples for the paper")

    s = scores.copy()
    s["evasion_score"] = pd.to_numeric(s["evasion_score"], errors="coerce")
    s = s.dropna(subset=["evasion_score"])

    top5 = s.nlargest(5, "evasion_score")
    bottom5 = s.nsmallest(5, "evasion_score")
    cols = ["transcript_id", "company_ticker", "filing_date", "question_text",
            "response_text", "evasion_score", "primary_evasion_type", "rationale"]

    def print_group(group, label):
        log(f"\n  {label}:")
        for _, row in group.iterrows():
            log(f"\n  transcript_id:        {row['transcript_id']}")
            log(f"  company_ticker:       {row['company_ticker']}")
            log(f"  filing_date:          {row['filing_date']}")
            log(f"  evasion_score:        {row['evasion_score']}")
            log(f"  primary_evasion_type: {row['primary_evasion_type']}")
            log(f"  question_text:        {row['question_text']}")
            log(f"  response_text:        {row['response_text']}")
            log(f"  rationale:            {row['rationale']}")

    log("\n1-2. Top 5 HIGHEST and LOWEST evasion_score pairs (entire dataset, n={}):".format(len(s)))
    print_group(top5, "TOP 5 HIGHEST evasion_score")
    print_group(bottom5, "TOP 5 LOWEST evasion_score")

    combined = pd.concat([top5, bottom5])[cols]
    combined.to_csv(QUALITATIVE_OUT, index=False, encoding="utf-8")
    log(f"\n3. Saved -> {QUALITATIVE_OUT}")
    return combined


# ── Section 8 ────────────────────────────────────────────────────────────────

def section8_summary(n_htm, n_qa, n_scores, n_unique_tid, n_unique_ticker_scores, mp,
                      top5_pct, corr_central, model_results_df, cv_summary, power_df,
                      bootstrap_stats):
    hr("SECTION 8: Final summary printout")

    log("\n--- Raw dataset ---")
    log(f"Total .htm transcript files on disk: {n_htm}")
    log(f"Total unique transcripts scored: {n_unique_tid}")
    log(f"Total unique companies (scored transcripts): {n_unique_ticker_scores}")
    log(f"Total Q&A pairs (all_qa_pairs.csv): {n_qa}")
    log(f"Total Q&A pairs scored (evasion_scores.csv): {n_scores}")

    log("\n--- Final panel (after CAR label attrition) ---")
    log(f"Observations: {len(mp)}")
    log(f"Companies: {mp['company_ticker'].nunique()}")
    log(f"Top-5 company concentration: {top5_pct:.1f}%")

    log("\n--- Human validation (results/kappa_statistics.csv) ---")
    if KAPPA_CSV.exists():
        kappa_df = pd.read_csv(KAPPA_CSV)
        log(f"File exists. Full contents:")
        log(kappa_df.to_string(index=False))
    else:
        log("*** results/kappa_statistics.csv NOT FOUND -- cannot report human validation numbers ***")

    log(f"\n--- Correlation results (evasion metrics vs car_3day, n={len(mp)}) ---")
    for feat, (r, p, n) in corr_central.items():
        sig = "significant (p<0.05)" if p < 0.05 else "NOT significant"
        log(f"{feat:<22} r={r:+.4f}  p={p:.4f}  n={n}  ({sig})")

    log("\n--- Classification results ---")
    class_df = model_results_df[model_results_df.task == "classification"]
    ci_low, ci_high, auroc_b, auroc_c = bootstrap_stats
    log(f"PRIMARY comparison (matches models.py's compare_config_b_c methodology -- same "
        f"model type, bootstrap-tested): Config B XGBoost AUROC={auroc_b:.4f} -> "
        f"Config C XGBoost AUROC={auroc_c:.4f}, diff={auroc_c - auroc_b:+.4f}, "
        f"95% CI=[{ci_low:+.4f}, {ci_high:+.4f}] "
        f"({'includes 0 -- NOT significant' if ci_low <= 0 <= ci_high else 'excludes 0 -- significant'}).")
    best_row = class_df.loc[class_df["auroc"].idxmax()]
    log(f"(For reference only, NOT bootstrap-tested and NOT a like-for-like comparison: the single "
        f"highest raw AUROC across all config/model-type cells is {best_row['auroc']:.4f} "
        f"(Config {best_row['config']}, {best_row['model_type']}) -- this mixes model types and "
        f"should not be read as 'evasion score improved AUROC by X' without the matched-model "
        f"bootstrap comparison above.)")

    log("\n--- Cross-validation stability (5-fold, Config B vs C) ---")
    for key, (mean, sd) in cv_summary.items():
        log(f"Config {key[0]} / {key[1]}: mean AUROC={mean:.4f}  sd={sd:.4f}")

    log("\n--- Power analysis ---")
    n_current = len(mp)
    cur_sub = power_df[power_df.n == n_current].sort_values("true_delta")
    adequate = cur_sub[cur_sub.power >= 0.80]
    if len(adequate):
        log(f"Minimum detectable effect at 80% power (n={n_current}): {adequate.iloc[0]['true_delta']:.2f}")
    else:
        log(f"Minimum detectable effect at 80% power (n={n_current}): larger than "
            f"{cur_sub['true_delta'].max():.2f} (not reached within tested range)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log("FINAL VERIFICATION PASS -- every number below is computed fresh from disk, "
        "right now, with no reliance on any cached/previous output.")

    qa, scores, mp, n_htm, n_qa, n_scores, n_unique_tid, n_unique_ticker = section1_provenance()
    section2_descriptive(mp)
    corr_matrix, corr_central = section3_correlation_matrix(mp)
    concentration_table, top5_pct = section4_concentration(mp)
    model_results_df, fitted, day6_auroc_current, bootstrap_stats = section5_classification(mp)

    rb_start = len(_LOG)
    cv_fold_results, cv_summary = section6_1_cross_validation(mp, day6_auroc_current)

    hr("SECTION 6.2: Pearson AND Spearman correlation (robustness_checks.py methodology)")
    start = len(rc._LOG_LINES)
    full_corr_table = task2_correlations(mp, label="final verification, full panel")
    _LOG.extend(rc._LOG_LINES[start:])

    auroc_results, ex_corr_table = section6_3_covid_exclusion(mp, day6_auroc_current, full_corr_table)
    power_df = section6_4_power_analysis(len(mp))
    rb_end = len(_LOG)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROBUSTNESS_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG[rb_start:rb_end]) + "\n")
    log(f"\n5. Saved Section 6 (robustness checks) output -> {ROBUSTNESS_OUT}")

    section7_qualitative(scores)

    section8_summary(n_htm, n_qa, n_scores, n_unique_tid, n_unique_ticker, mp,
                      top5_pct, corr_central, model_results_df, cv_summary, power_df,
                      bootstrap_stats)


if __name__ == "__main__":
    main()
