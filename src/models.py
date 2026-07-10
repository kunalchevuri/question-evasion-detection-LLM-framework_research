"""
Prediction models for master_panel.csv.

Central question: does the LLM-judged evasion score add out-of-sample
predictive value for post-earnings market reaction (car_3day / car_direction),
beyond accounting fundamentals and Loughran-McDonald sentiment already
computed from the same call? Config A/B are the baseline; C/D add evasion.

n=91 is small. Every choice here (chronological split, median imputation
from train only, bootstrap instead of DeLong, shallow trees) is made to keep
this honest rather than to make the numbers look better than they are.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                              precision_score, r2_score, recall_score,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MASTER_PANEL_CSV = DATA_DIR / "features" / "master_panel.csv"
MODEL_RESULTS_CSV = RESULTS_DIR / "model_results.csv"
SHAP_IMPORTANCE_CSV = RESULTS_DIR / "shap_importance.csv"

RANDOM_STATE = 42
N_BOOTSTRAP = 1000

ACCOUNTING_FEATURES = ["revenue_growth", "gross_margin", "operating_margin", "roa"]
LM_FEATURES = ["lm_positive", "lm_negative", "lm_uncertainty", "lm_litigious"]

CONFIGS = {
    "A": ACCOUNTING_FEATURES,
    "B": ACCOUNTING_FEATURES + LM_FEATURES,
    "C": ACCOUNTING_FEATURES + LM_FEATURES + ["mean_evasion_score"],
    "D": ACCOUNTING_FEATURES + LM_FEATURES + ["mean_evasion_score", "evasion_variance", "max_evasion_score"],
}

# Validated categorical palette (dataviz skill reference palette).
COLOR_LR = "#2a78d6"    # blue, categorical slot 1
COLOR_XGB = "#1baf7a"   # aqua, categorical slot 2
COLOR_POS = "#0072B2"   # colorblind-safe blue (Okabe-Ito) -- positive car_direction
COLOR_NEG = "#E69F00"   # colorblind-safe orange (Okabe-Ito) -- negative car_direction
COLOR_SHAP = "#2a78d6"
COLOR_TREND = "#0b0b0b"


# ── Step 1: load + chronological split ──────────────────────────────────────

def load_data():
    df = pd.read_csv(MASTER_PANEL_CSV, dtype={"transcript_id": str})
    print(f"Total rows: {len(df)}")
    print(f"Unique companies: {df['company_ticker'].nunique()}")
    print("Filing year distribution:")
    print(df["filing_year"].value_counts().sort_index().to_string())
    return df


def chronological_split(df, test_frac=0.3):
    df = df.sort_values("filing_date").reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * test_frac))
    n_train = n - n_test
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()

    print(f"\nTrain: {len(train)} rows, {train['filing_date'].min()} to {train['filing_date'].max()}")
    print(f"Test:  {len(test)} rows, {test['filing_date'].min()} to {test['filing_date'].max()}")

    if len(test) < 15:
        print("WARNING: test set has fewer than 15 rows -- results will be statistically "
              "unreliable. Proceeding anyway per instructions.")
    return train, test


# ── Step 2: median imputation from train only ───────────────────────────────

def impute_accounting_features(train, test):
    train = train.copy()
    test = test.copy()
    print("\nImputing missing accounting features using TRAIN median only (never test statistics):")
    for col in ACCOUNTING_FEATURES:
        median_val = train[col].median()
        train_missing = train[col].isna()
        test_missing = test[col].isna()
        flag_col = f"{col}_imputed"
        train[flag_col] = train_missing
        test[flag_col] = test_missing
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
        print(f"  {col:<18} train median={median_val:>9.4f}  "
              f"imputed: {train_missing.sum():>2d} train row(s), {test_missing.sum():>2d} test row(s)")
    return train, test


# ── Step 4: models ───────────────────────────────────────────────────────────

def prepare_xy(df, feature_cols, target_col):
    return df[feature_cols].copy(), df[target_col].copy()


def compute_classification_metrics(y_true, y_pred, y_proba, label=""):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) < 2:
        print(f"  WARNING [{label}]: test set has only one class present -- AUROC cannot be computed, skipping.")
        metrics["auroc"] = np.nan
    else:
        metrics["auroc"] = roc_auc_score(y_true, y_proba)
    return metrics


def train_logistic_regression(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    return model, scaler, y_pred, y_proba


def train_xgb_classifier(X_train, y_train, X_test):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = XGBClassifier(
        max_depth=3, n_estimators=100, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_proba


def train_xgb_regressor(X_train, y_train, X_test):
    model = XGBRegressor(
        max_depth=3, n_estimators=100, learning_rate=0.05,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_train, y_train)
    return model, model.predict(X_test)


def run_classification(train, test):
    print("\n" + "=" * 70)
    print("STEP 4: Classification models (target = car_direction)")
    print("Note: logistic regression features are standardized (fit on train "
          "only); XGBoost is scale-invariant so uses raw features.")
    print("=" * 70)

    results = []
    fitted = {}

    for config_name, feature_cols in CONFIGS.items():
        X_train, y_train = prepare_xy(train, feature_cols, "car_direction")
        X_test, y_test = prepare_xy(test, feature_cols, "car_direction")

        print(f"\n--- Config {config_name}: {feature_cols} ---")

        lr_model, scaler, lr_pred, lr_proba = train_logistic_regression(X_train, y_train, X_test)
        lr_metrics = compute_classification_metrics(y_test, lr_pred, lr_proba, label=f"{config_name}/LR")
        print(f"  LogisticRegression: AUROC={lr_metrics['auroc']:.4f}  Acc={lr_metrics['accuracy']:.4f}  "
              f"Prec={lr_metrics['precision']:.4f}  Rec={lr_metrics['recall']:.4f}  F1={lr_metrics['f1']:.4f}")
        results.append({"config": config_name, "model_type": "LogisticRegression", "task": "classification",
                         **lr_metrics, "r_squared": np.nan, "rmse": np.nan,
                         "n_train": len(X_train), "n_test": len(X_test)})
        fitted[(config_name, "LogisticRegression")] = {
            "model": lr_model, "scaler": scaler,
            "X_train": X_train, "X_test": X_test, "y_test": y_test, "y_proba": lr_proba,
        }

        xgb_model, xgb_pred, xgb_proba = train_xgb_classifier(X_train, y_train, X_test)
        xgb_metrics = compute_classification_metrics(y_test, xgb_pred, xgb_proba, label=f"{config_name}/XGB")
        print(f"  XGBoost:            AUROC={xgb_metrics['auroc']:.4f}  Acc={xgb_metrics['accuracy']:.4f}  "
              f"Prec={xgb_metrics['precision']:.4f}  Rec={xgb_metrics['recall']:.4f}  F1={xgb_metrics['f1']:.4f}")
        results.append({"config": config_name, "model_type": "XGBoost", "task": "classification",
                         **xgb_metrics, "r_squared": np.nan, "rmse": np.nan,
                         "n_train": len(X_train), "n_test": len(X_test)})
        fitted[(config_name, "XGBoost")] = {
            "model": xgb_model, "scaler": None,
            "X_train": X_train, "X_test": X_test, "y_test": y_test, "y_proba": xgb_proba,
        }

    return results, fitted


# ── Step 5: Config B vs C comparison ────────────────────────────────────────

def bootstrap_auroc_diff(y_test, proba_b, proba_c, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_test)
    proba_b = np.asarray(proba_b)
    proba_c = np.asarray(proba_c)
    n = len(y_arr)

    diffs = []
    skipped = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_boot = y_arr[idx]
        if len(set(y_boot)) < 2:
            skipped += 1
            continue
        auroc_b = roc_auc_score(y_boot, proba_b[idx])
        auroc_c = roc_auc_score(y_boot, proba_c[idx])
        diffs.append(auroc_c - auroc_b)

    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return diffs, ci_low, ci_high, skipped


def compare_config_b_c(fitted, model_type="XGBoost"):
    print("\n" + "=" * 70)
    print(f"STEP 5: Config B vs Config C comparison ({model_type})")
    print("=" * 70)
    print(f"Using {model_type} for this comparison (the stronger, non-linear learner and "
          f"the same model family used for the SHAP analysis in Step 7).")

    b = fitted[("B", model_type)]
    c = fitted[("C", model_type)]

    if not b["y_test"].equals(c["y_test"]):
        raise ValueError("Config B and C test sets do not align -- cannot do a paired comparison.")

    y_test = b["y_test"]
    has_both_classes = len(set(y_test)) > 1

    auroc_b = roc_auc_score(y_test, b["y_proba"]) if has_both_classes else np.nan
    auroc_c = roc_auc_score(y_test, c["y_proba"]) if has_both_classes else np.nan
    raw_diff = auroc_c - auroc_b

    print(f"Config B (accounting + LM sentiment)              AUROC: {auroc_b:.4f}")
    print(f"Config C (Config B + mean_evasion_score)           AUROC: {auroc_c:.4f}")
    print(f"Raw AUROC difference (C - B): {raw_diff:+.4f}")

    diffs, ci_low, ci_high, skipped = bootstrap_auroc_diff(y_test, b["y_proba"], c["y_proba"])
    print(f"\nBootstrap: {N_BOOTSTRAP} resamples of the test set ({skipped} skipped for "
          f"landing on a single class)")
    print(f"95% CI of AUROC difference (C - B): [{ci_low:+.4f}, {ci_high:+.4f}]")
    if ci_low <= 0 <= ci_high:
        print("  --> The 95% CI includes 0: no statistically significant difference detected.")
    else:
        print("  --> The 95% CI excludes 0: suggestive of a real difference.")

    print(f"\nNOTE: DeLong's test was not used -- its asymptotic-normality assumption is not "
          f"reliable at n_test={len(y_test)} (n=91 total). With this few observations, any "
          f"statistical comparison has limited power; treat this result as suggestive, not "
          f"definitive.")

    return {"auroc_b": auroc_b, "auroc_c": auroc_c, "raw_diff": raw_diff,
            "ci_low": ci_low, "ci_high": ci_high, "n_skipped": skipped, "diffs": diffs}


# ── Step 6: regression ───────────────────────────────────────────────────────

def run_regression(train, test):
    print("\n" + "=" * 70)
    print("STEP 6: Regression models (target = car_3day, continuous)")
    print("=" * 70)

    results = []
    for config_name, feature_cols in CONFIGS.items():
        X_train, y_train = prepare_xy(train, feature_cols, "car_3day")
        X_test, y_test = prepare_xy(test, feature_cols, "car_3day")

        model, y_pred = train_xgb_regressor(X_train, y_train, X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        flag = "  <-- WORSE than predicting the mean" if r2 < 0 else ""
        print(f"Config {config_name}: R^2={r2:+.4f}{flag}   RMSE={rmse:.4f}")

        results.append({"config": config_name, "model_type": "XGBoost", "task": "regression",
                         "auroc": np.nan, "accuracy": np.nan, "precision": np.nan,
                         "recall": np.nan, "f1": np.nan,
                         "r_squared": r2, "rmse": rmse,
                         "n_train": len(X_train), "n_test": len(X_test)})
    return results


# ── Step 7: SHAP ──────────────────────────────────────────────────────────────

def compute_shap_for_config(fitted, config_name, model_type="XGBoost"):
    entry = fitted[(config_name, model_type)]
    model = entry["model"]
    feature_cols = CONFIGS[config_name]

    # Explain on train+test combined: with n as small as this, using test alone
    # (~27 rows) gives a noisy importance ranking. This does not affect any
    # reported predictive-performance metric -- it only interprets what the
    # already-fitted model has learned.
    X_full = pd.concat([entry["X_train"], entry["X_test"]], axis=0)

    explainer = shap.TreeExplainer(model)
    sv = explainer(X_full)
    values = sv.values
    if values.ndim == 3:
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    return imp, values, X_full


def run_shap_analysis(fitted, classification_results):
    print("\n" + "=" * 70)
    print("STEP 7: SHAP feature importance")
    print("=" * 70)

    xgb_results = [r for r in classification_results
                   if r["model_type"] == "XGBoost" and not pd.isna(r["auroc"])]
    if not xgb_results:
        print("No XGBoost classifier has a valid AUROC -- cannot select a best model for SHAP.")
        return None, None

    best = max(xgb_results, key=lambda r: r["auroc"])
    best_config = best["config"]
    print(f"Best-performing XGBoost classifier: Config {best_config} (test AUROC = {best['auroc']:.4f})")

    imp, values, X_full = compute_shap_for_config(fitted, best_config)
    print(f"\nMean |SHAP value| per feature, Config {best_config} (highest to lowest):")
    print(imp[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

    evasion_features = ["mean_evasion_score", "evasion_variance", "max_evasion_score"]
    present = [f for f in evasion_features if f in CONFIGS[best_config]]
    if present:
        print(f"\nEvasion feature ranking within Config {best_config}:")
        for f in present:
            row = imp[imp["feature"] == f].iloc[0]
            print(f"  {f:<20} rank {int(row['rank'])} of {len(imp)}  (mean |SHAP| = {row['mean_abs_shap']:.5f})")
    else:
        print(f"\nConfig {best_config} (the overall best XGBoost model) does not include any evasion "
              f"features, so they cannot be ranked within it. Showing the best XGBoost model among "
              f"Configs C/D (the ones that do include evasion features) instead:")
        ce_results = [r for r in xgb_results if r["config"] in ("C", "D")]
        if ce_results:
            best_ce = max(ce_results, key=lambda r: r["auroc"])
            ce_config = best_ce["config"]
            print(f"  Best of {{C, D}}: Config {ce_config} (test AUROC = {best_ce['auroc']:.4f})")
            imp_ce, _, _ = compute_shap_for_config(fitted, ce_config)
            print(imp_ce[["rank", "feature", "mean_abs_shap"]].to_string(index=False))
            for f in evasion_features:
                if f in CONFIGS[ce_config]:
                    row = imp_ce[imp_ce["feature"] == f].iloc[0]
                    print(f"  {f:<20} rank {int(row['rank'])} of {len(imp_ce)}  (mean |SHAP| = {row['mean_abs_shap']:.5f})")

    return best_config, imp


# ── Step 8: figures ───────────────────────────────────────────────────────────

def plot_config_auroc_comparison(classification_results, out_path):
    df = pd.DataFrame(classification_results)
    configs = list(CONFIGS.keys())

    lr_auroc = [df[(df.config == c) & (df.model_type == "LogisticRegression")]["auroc"].iloc[0] for c in configs]
    xgb_auroc = [df[(df.config == c) & (df.model_type == "XGBoost")]["auroc"].iloc[0] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, lr_auroc, width, label="Logistic Regression", color=COLOR_LR)
    bars2 = ax.bar(x + width / 2, xgb_auroc, width, label="XGBoost", color=COLOR_XGB)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color="#0b0b0b")

    ax.axhline(0.5, color="#9a9a94", linestyle="--", linewidth=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Config {c}" for c in configs])
    ax.set_ylabel("Test AUROC")
    ax.set_title("Out-of-sample AUROC by feature configuration (car_direction)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=-1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_shap_importance(imp, best_config, out_path):
    imp_sorted = imp.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(imp_sorted) + 1)))
    ax.barh(imp_sorted["feature"], imp_sorted["mean_abs_shap"], color=COLOR_SHAP)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP feature importance -- best XGBoost model (Config {best_config})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.8, zorder=-1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_evasion_vs_car(df, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for direction, color, label in [(1, COLOR_POS, "Positive CAR"), (0, COLOR_NEG, "Negative CAR")]:
        sub = df[df["car_direction"] == direction]
        ax.scatter(sub["mean_evasion_score"], sub["car_3day"], color=color, label=label,
                   s=28, alpha=0.85, edgecolors="white", linewidths=0.5)

    x = df["mean_evasion_score"].to_numpy()
    y = df["car_3day"].to_numpy()
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=COLOR_TREND, linewidth=2,
                linestyle="--", label="Linear trend")

    ax.axhline(0, color="#9a9a94", linewidth=1, zorder=0)
    ax.set_xlabel("Mean evasion score (transcript-level, 0-100)")
    ax.set_ylabel("car_3day (next-quarter 3-day CAR)")
    ax.set_title("Evasion score vs. subsequent-quarter market reaction")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e5e4df", linewidth=0.8, zorder=-1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Load data + chronological split")
    print("=" * 70)
    df = load_data()
    train, test = chronological_split(df)

    print("\n" + "=" * 70)
    print("STEP 2: Impute missing accounting features (train-median only)")
    print("=" * 70)
    train, test = impute_accounting_features(train, test)

    classification_results, fitted = run_classification(train, test)
    b_vs_c = compare_config_b_c(fitted, model_type="XGBoost")
    regression_results = run_regression(train, test)
    best_config, shap_imp = run_shap_analysis(fitted, classification_results)

    print("\n" + "=" * 70)
    print("STEP 8: Saving results")
    print("=" * 70)
    all_results = classification_results + regression_results
    results_df = pd.DataFrame(all_results)[
        ["config", "model_type", "task", "auroc", "accuracy", "precision", "recall", "f1",
         "r_squared", "rmse", "n_train", "n_test"]
    ]
    results_df.to_csv(MODEL_RESULTS_CSV, index=False, encoding="utf-8")
    print(f"Saved {MODEL_RESULTS_CSV}")

    if shap_imp is not None:
        shap_imp.to_csv(SHAP_IMPORTANCE_CSV, index=False, encoding="utf-8")
        print(f"Saved {SHAP_IMPORTANCE_CSV}")

    plot_config_auroc_comparison(classification_results, RESULTS_DIR / "config_auroc_comparison.png")
    if shap_imp is not None:
        plot_shap_importance(shap_imp, best_config, RESULTS_DIR / "shap_importance.png")
    plot_evasion_vs_car(df, RESULTS_DIR / "evasion_vs_car.png")

    print("\n" + "=" * 70)
    print("STEP 9: FULL SUMMARY")
    print("=" * 70)

    print("\n--- Classification + Regression results table ---")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(results_df.to_string(index=False))

    print("\n--- Config B vs Config C (XGBoost) ---")
    print(f"AUROC B (accounting+LM): {b_vs_c['auroc_b']:.4f}")
    print(f"AUROC C (+mean_evasion_score): {b_vs_c['auroc_c']:.4f}")
    print(f"Raw diff (C - B): {b_vs_c['raw_diff']:+.4f}")
    print(f"Bootstrap 95% CI: [{b_vs_c['ci_low']:+.4f}, {b_vs_c['ci_high']:+.4f}]")

    xgb_df = results_df[(results_df.model_type == "XGBoost") & (results_df.task == "classification")]
    auroc_b = xgb_df[xgb_df.config == "B"]["auroc"].iloc[0]
    auroc_c = xgb_df[xgb_df.config == "C"]["auroc"].iloc[0]
    auroc_d = xgb_df[xgb_df.config == "D"]["auroc"].iloc[0]
    winner = max([("C", auroc_c), ("D", auroc_d)], key=lambda t: t[1] if not pd.isna(t[1]) else -np.inf)
    print(f"\nDid Config C or D beat Config B (XGBoost AUROC)? "
          f"C: {auroc_c - auroc_b:+.4f}, D: {auroc_d - auroc_b:+.4f}. "
          f"Best of the two: Config {winner[0]} ({winner[1] - auroc_b:+.4f} vs. B).")

    if shap_imp is not None:
        print(f"\n--- SHAP ranking (Config {best_config}) ---")
        print(shap_imp[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

    print("\n--- Interpretation ---")
    print(
        f"With n=91 total observations (train={len(train)}, test={len(test)}), this is a small-sample "
        f"analysis and every number above should be read with that in mind. The bootstrap 95% CI on the "
        f"Config C vs. B AUROC difference is "
        f"{'wide and includes 0' if b_vs_c['ci_low'] <= 0 <= b_vs_c['ci_high'] else 'excludes 0'}, "
        f"meaning the data here {'cannot rule out that evasion score adds nothing beyond accounting and LM sentiment' if b_vs_c['ci_low'] <= 0 <= b_vs_c['ci_high'] else 'is suggestive of evasion score adding predictive value'}. "
        f"The regression task's R^2 values"
        f"{' include negative values (worse than predicting the mean), which is a real and expected outcome at this sample size, not a bug' if (results_df[results_df.task=='regression']['r_squared'] < 0).any() else ' are all non-negative'}. "
        f"Taken together, this dataset is large enough to run the full pipeline end-to-end and to see whether "
        f"there is a hint of signal in the SHAP ranking, but it is not large enough to make a confident, "
        f"publishable claim about evasion score's incremental predictive value on its own -- that would need "
        f"a substantially larger sample of earnings calls before the confidence intervals narrow enough to be "
        f"conclusive."
    )


if __name__ == "__main__":
    main()
