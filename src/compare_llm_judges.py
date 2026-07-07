"""
Cross-model validation comparison: merge Claude's original judge scores, any
number of additional LLM judge score files (Groq llama-3.3-70b-versatile,
OpenAI gpt-4o, ...), and both human scores for the same 75 validated pairs,
then compute inter-rater agreement across every pair of raters.

Merge safety: every score set is joined on pair_id, which IS
validation/human_annotation.csv's own row index (assigned once, in
multi_llm_judge.py, by iterating that file directly and never reconstructed
via a groupby/cumcount join against a separately-ordered file -- see that
script's docstring and sample_validation.py's docstring for the bug this
avoids). Reuses compute_kappa.py's exact bin_codes/BIN_LABELS/BIN_EDGES
rather than reimplementing the binning logic.
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_kappa import BIN_LABELS, bin_codes, interpret_kappa  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / "validation"
RESULTS_DIR = BASE_DIR / "results"
HUMAN_ANNOTATION_CSV = VALIDATION_DIR / "human_annotation.csv"
OUTPUT_CSV = RESULTS_DIR / "multi_llm_comparison.csv"

# Each entry: (rater_key, display_label, score_csv_or_None, scale_divisor)
# score_csv_or_None: None means the score lives directly in human_annotation.csv
# (Claude's original judge scores and the two human raters). scale_divisor
# converts each rater's native scale to a common 0-10 scale for Pearson.
RATER_SOURCES = [
    ("claude", "Claude (claude-sonnet-4-6)", None, "evasion_score", 10.0),
    ("human1", "Human 1", None, "human1_evasion_score", 1.0),
    ("human2", "Human 2", None, "human2_evasion_score", 1.0),
    ("groq_llama", "Groq (llama-3.3-70b-versatile)", VALIDATION_DIR / "second_llm_scores.csv", "evasion_score", 10.0),
    ("gpt4o", "OpenAI (gpt-4o)", VALIDATION_DIR / "third_llm_scores.csv", "evasion_score", 10.0),
]


def load_merged():
    df = pd.read_csv(HUMAN_ANNOTATION_CSV)
    h1_filled = df["human1_evasion_score"].notna() & (df["human1_evasion_score"].astype(str).str.strip() != "")
    h2_filled = df["human2_evasion_score"].notna() & (df["human2_evasion_score"].astype(str).str.strip() != "")
    both_filled = h1_filled & h2_filled
    subset = df.loc[both_filled].copy()
    print(f"human_annotation.csv: {len(df)} total rows, {len(subset)} with both human scores present.")

    subset["human1_evasion_score"] = pd.to_numeric(subset["human1_evasion_score"], errors="coerce")
    subset["human2_evasion_score"] = pd.to_numeric(subset["human2_evasion_score"], errors="coerce")
    subset["evasion_score"] = pd.to_numeric(subset["evasion_score"], errors="coerce")
    subset = subset.dropna(subset=["human1_evasion_score", "human2_evasion_score", "evasion_score"])
    subset = subset.rename(columns={"evasion_score": "claude_score"})

    merged = subset.rename(columns={"human1_evasion_score": "human1_score", "human2_evasion_score": "human2_score"})
    active_raters = [("claude", "Claude (claude-sonnet-4-6)", 10.0),
                      ("human1", "Human 1", 1.0),
                      ("human2", "Human 2", 1.0)]

    for key, label, csv_path, score_col, scale in RATER_SOURCES:
        if csv_path is None:
            continue
        if not csv_path.exists():
            print(f"  (skipping {label}: {csv_path} not found)")
            continue
        scores = pd.read_csv(csv_path)
        print(f"{csv_path.name}: {len(scores)} rows, model_name={scores['model_name'].unique().tolist()}")
        scores = scores.set_index("pair_id")
        merged = merged.join(scores[[score_col]].rename(columns={score_col: f"{key}_score"}), how="inner")
        active_raters.append((key, label, scale))
        n_dropped = len(subset) - len(merged)
        if n_dropped:
            print(f"  WARNING: {n_dropped} row(s) dropped after joining {label} "
                  f"-- no matching pair_id (likely a failed call, check the errors CSV)")

    print(f"\nActive raters ({len(active_raters)}): {[label for _, label, _ in active_raters]}")
    print(f"Merged (all active raters present): n={len(merged)}\n")
    return merged, active_raters


def main():
    merged, active_raters = load_merged()
    scale = {key: s for key, _, s in active_raters}
    labels = {key: label for key, label, _ in active_raters}

    for key in scale:
        merged[f"{key}_bin"] = bin_codes(merged[f"{key}_score"] / scale[key])

    bin_range = list(range(len(BIN_LABELS)))
    pairs = list(combinations([k for k, _, _ in active_raters], 2))

    rows = []
    print("=" * 90)
    print(f"CROSS-MODEL VALIDATION -- {len(pairs)} rater-pair comparisons (n={len(merged)})")
    print("=" * 90)
    for a, b in pairs:
        kappa = cohen_kappa_score(merged[f"{a}_bin"], merged[f"{b}_bin"], labels=bin_range)
        qwk = cohen_kappa_score(merged[f"{a}_bin"], merged[f"{b}_bin"], labels=bin_range, weights="quadratic")
        r, p = pearsonr(merged[f"{a}_score"] / scale[a], merged[f"{b}_score"] / scale[b])

        label = f"{labels[a]}  vs  {labels[b]}"
        print(f"\n{label}")
        print(f"  Unweighted kappa         = {kappa:.4f}  ({interpret_kappa(kappa)})")
        print(f"  Quadratic weighted kappa = {qwk:.4f}  ({interpret_kappa(qwk)})")
        print(f"  Pearson r                = {r:+.4f}  (p={p:.4f})")

        rows.append({
            "comparison": f"{a}_vs_{b}", "rater_a": a, "rater_b": b,
            "kappa": kappa, "quadratic_kappa": qwk, "pearson_r": r, "pearson_p": p,
            "n_rows": len(merged), "interpretation": interpret_kappa(kappa),
        })

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(results_df[["comparison", "kappa", "quadratic_kappa", "pearson_r", "pearson_p", "n_rows"]]
              .to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved -> {OUTPUT_CSV}")

    # ── Plain-language interpretation ──────────────────────────────────────
    r = {row["comparison"]: row for row in rows}
    llm_keys = [k for k, _, _ in active_raters if k not in ("human1", "human2")]
    human_keys = ["human1", "human2"]

    llm_llm_pairs = [c for c in pairs if c[0] in llm_keys and c[1] in llm_keys]
    llm_human_pairs = [c for c in pairs if (c[0] in llm_keys) != (c[1] in llm_keys)]
    human_human_pairs = [c for c in pairs if c[0] in human_keys and c[1] in human_keys]

    def avg(field, comp_list):
        vals = [r[f"{a}_vs_{b}"][field] for a, b in comp_list]
        return sum(vals) / len(vals) if vals else float("nan")

    mean_llm_llm_qwk = avg("quadratic_kappa", llm_llm_pairs)
    mean_llm_llm_r = avg("pearson_r", llm_llm_pairs)
    mean_llm_human_qwk = avg("quadratic_kappa", llm_human_pairs)
    mean_llm_human_r = avg("pearson_r", llm_human_pairs)
    human_human_qwk = avg("quadratic_kappa", human_human_pairs)
    human_human_r = avg("pearson_r", human_human_pairs)

    print("\n" + "=" * 90)
    print("PLAIN-LANGUAGE INTERPRETATION")
    print("=" * 90)
    print(f"LLMs in this comparison: {[labels[k] for k in llm_keys]}")
    print(f"Mean LLM-vs-LLM agreement:    QWK={mean_llm_llm_qwk:.4f}, r={mean_llm_llm_r:+.4f}  "
          f"(over {len(llm_llm_pairs)} pair(s): {llm_llm_pairs})")
    print(f"Mean LLM-vs-human agreement:  QWK={mean_llm_human_qwk:.4f}, r={mean_llm_human_r:+.4f}  "
          f"(over {len(llm_human_pairs)} pair(s))")
    print(f"Human1 vs Human2 (baseline):  QWK={human_human_qwk:.4f}, r={human_human_r:+.4f}")

    if mean_llm_llm_qwk > mean_llm_human_qwk and mean_llm_llm_r > mean_llm_human_r:
        verdict = (
            "The LLMs agree with EACH OTHER more than any of them agrees with humans, on both metrics. "
            "This is a mixed signal for construct validity: it shows the evasion-scoring task is "
            "reproducible ACROSS DIFFERENT MODEL ARCHITECTURES (not an artifact of one model's specific "
            "quirks), which is a genuine strength. But it also raises the possibility that the models "
            "share a common bias or blind spot relative to human judgment -- convergence between models "
            "is necessary but not sufficient evidence that the score captures what human readers would "
            "call 'evasiveness'; it could also mean the models key off similar surface features "
            "(response length, hedge words, disclaimer boilerplate) that diverge from human intuition in "
            "the same direction."
        )
    elif mean_llm_llm_qwk < mean_llm_human_qwk and mean_llm_llm_r < mean_llm_human_r:
        verdict = (
            "Each LLM agrees with humans about as much or more than the LLMs agree with each other. "
            "This is a stronger result for construct validity: it suggests the models are not simply "
            "converging on a shared model-specific artifact, and that each is independently tracking "
            "something closer to the human-perceived construct of evasiveness than to the other models' "
            "idiosyncrasies."
        )
    else:
        verdict = (
            "LLM-LLM agreement and average LLM-human agreement are close, with no clear direction on both "
            "metrics simultaneously -- read the two metrics individually above rather than a single "
            "verdict; this pattern does not cleanly support or undermine model-independence."
        )
    print(f"\n{verdict}")

    if human_human_qwk > 0:
        print(
            f"\nRelative to the original human1-vs-human2 baseline (QWK={human_human_qwk:.4f}), the "
            f"cross-model numbers above are the right context for judging whether this measurement "
            f"replicates: agreement levels in a similar range to human-human agreement would suggest the "
            f"scoring task has inherent ceiling-level noise shared by every rater type (human or model), "
            f"rather than any one LLM judge specifically being unreliable."
        )


if __name__ == "__main__":
    main()
