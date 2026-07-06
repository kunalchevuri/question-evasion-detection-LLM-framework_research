"""
Regenerate the three paper figures from the CURRENT verified data
(data/features/master_panel.csv, results/final_model_results.csv), replacing
the stale pre-expansion PNGs. Reuses the exact color constants from models.py
(already validated against the dataviz skill's reference palette) so these
figures match the established visual style -- only the titles, DPI, output
paths, and underlying data source change per this request.

Figure 1 (evasion_vs_car_final.png) and Figure 2 (config_auroc_comparison_
final.png) are read directly from already-verified files (master_panel.csv,
final_model_results.csv) with no re-derivation, so there is zero risk of a
mismatch against those files. Figure 3 (shap_importance_final.png) has no
precomputed source to read from -- Day 6's original SHAP analysis was never
rerun on the expanded panel -- so this script trains Config A XGBoost fresh
(same chronological_split + impute_accounting_features methodology already
established) and computes SHAP directly on n=155.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    CONFIGS, COLOR_LR, COLOR_NEG, COLOR_POS, COLOR_SHAP, COLOR_TREND, COLOR_XGB,
    chronological_split, impute_accounting_features, train_xgb_classifier,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MASTER_PANEL_CSV = DATA_DIR / "features" / "master_panel.csv"
MODEL_RESULTS_CSV = RESULTS_DIR / "final_model_results.csv"
CORR_MATRIX_CSV = RESULTS_DIR / "final_correlation_matrix.csv"

FIG1_OUT = RESULTS_DIR / "evasion_vs_car_final.png"
FIG2_OUT = RESULTS_DIR / "config_auroc_comparison_final.png"
FIG3_OUT = RESULTS_DIR / "shap_importance_final.png"

DPI = 300


def figure1_evasion_vs_car(mp):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for direction, color, label in [(1, COLOR_POS, "Positive CAR"), (0, COLOR_NEG, "Negative CAR")]:
        sub = mp[mp["car_direction"] == direction]
        ax.scatter(sub["mean_evasion_score"], sub["car_3day"], color=color, label=label,
                   s=28, alpha=0.85, edgecolors="white", linewidths=0.5)

    x = mp["mean_evasion_score"].to_numpy()
    y = mp["car_3day"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=COLOR_TREND, linewidth=2,
            linestyle="--", label="Linear trend")

    ax.axhline(0, color="#9a9a94", linewidth=1, zorder=0)
    ax.set_xlabel("Mean evasion score (transcript-level, 0-100)")
    ax.set_ylabel("car_3day (next-quarter 3-day CAR)")
    ax.set_title("Evasion Score vs. Subsequent-Quarter Market Reaction")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e5e4df", linewidth=0.8, zorder=-1)
    fig.tight_layout()
    fig.savefig(FIG1_OUT, dpi=DPI)
    plt.close(fig)

    r, p = pearsonr(x, y)
    print(f"FIGURE 1 saved -> {FIG1_OUT}")
    print(f"  Pearson r (mean_evasion_score vs car_3day), n={len(mp)}: r={r:+.4f}  p={p:.4f}")
    return r, p


def figure2_config_auroc(model_results):
    configs = list(CONFIGS.keys())  # ["A", "B", "C", "D"]
    lr_auroc = [model_results[(model_results.config == c) &
                               (model_results.model_type == "LogisticRegression")]["auroc"].iloc[0]
                for c in configs]
    xgb_auroc = [model_results[(model_results.config == c) &
                                (model_results.model_type == "XGBoost")]["auroc"].iloc[0]
                 for c in configs]

    print("FIGURE 2 bar values (from final_model_results.csv):")
    for c, lr, xgb in zip(configs, lr_auroc, xgb_auroc):
        print(f"  Config {c}: LR={lr:.4f}  XGBoost={xgb:.4f}")

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
    ax.set_title("Out-of-Sample AUROC by Feature Configuration")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=-1)
    fig.tight_layout()
    fig.savefig(FIG2_OUT, dpi=DPI)
    plt.close(fig)
    print(f"FIGURE 2 saved -> {FIG2_OUT}")


def figure3_shap_config_a(mp):
    train, test = chronological_split(mp)
    train, test = impute_accounting_features(train, test)

    feature_cols = CONFIGS["A"]
    X_train = train[feature_cols]
    y_train = train["car_direction"]
    X_test = test[feature_cols]
    y_test = test["car_direction"]

    model, y_pred, y_proba = train_xgb_classifier(X_train, y_train, X_test)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y_test, y_proba)
    print(f"FIGURE 3: retrained Config A XGBoost fresh on current panel -- test AUROC={auroc:.4f} "
          f"(cross-check vs final_model_results.csv Config A XGBoost={0.5123:.4f})")

    X_full = pd.concat([X_train, X_test], axis=0)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_full)
    values = sv.values
    if values.ndim == 3:
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1

    print("FIGURE 3 SHAP importance (Config A, XGBoost):")
    print(imp[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

    imp_sorted = imp.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(imp_sorted) + 1)))
    ax.barh(imp_sorted["feature"], imp_sorted["mean_abs_shap"], color=COLOR_SHAP)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance -- Best XGBoost Model (Config A)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.8, zorder=-1)
    fig.tight_layout()
    fig.savefig(FIG3_OUT, dpi=DPI)
    plt.close(fig)
    print(f"FIGURE 3 saved -> {FIG3_OUT}")
    return imp


def main():
    mp = pd.read_csv(MASTER_PANEL_CSV, dtype={"transcript_id": str})
    print(f"Loaded master_panel.csv: {len(mp)} rows, {mp['company_ticker'].nunique()} companies\n")

    model_results = pd.read_csv(MODEL_RESULTS_CSV)

    print("=" * 70)
    r, p = figure1_evasion_vs_car(mp)
    corr_matrix = pd.read_csv(CORR_MATRIX_CSV, index_col=0)
    ref_r = corr_matrix.loc["mean_evasion_score", "car_3day"]
    match = "MATCH" if abs(r - ref_r) < 1e-6 else "MISMATCH"
    print(f"  Cross-check vs final_correlation_matrix.csv: stored r={ref_r:+.4f} -> {match}")

    print("\n" + "=" * 70)
    figure2_config_auroc(model_results)

    print("\n" + "=" * 70)
    figure3_shap_config_a(mp)

    print("\n" + "=" * 70)
    print("All three figures saved. Not committing -- awaiting review.")


if __name__ == "__main__":
    main()
