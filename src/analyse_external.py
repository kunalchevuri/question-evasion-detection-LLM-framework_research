"""
Analyse the judge's scores against external corpus labels.

Task A2 of the IEEE BigData sprint, applied to both external corpora.

Reports, per corpus:

  Threshold-free (report these first -- they do not depend on any cut point)
    * Spearman rho between the continuous evasion score and the ordinal label
    * one-vs-rest AUROC for the most-evasive class, with a bootstrap 95% CI

  Zero-shot ordinal (pre-registered, no fitting)
    Cut points fixed from OUR OWN corpus distribution -- the tertiles of the
    3,350-pair earnings-call score distribution -- and applied unchanged. No
    external label is consulted in choosing them, so nothing is fitted.

  Threshold-fit (fitted, honestly labelled as such)
    A 30/70 split; two cut points chosen on the 30% by maximising Macro-F1;
    Macro-F1 and quadratic-weighted kappa reported on the held-out 70%.

Both mapping approaches are reported together, as the advisor asked, because
the pairing is what makes the result interpretable: the fitted version shows
the ceiling, the zero-shot version shows what the measure does untuned.

Output: results/external_benchmark_{corpus}.txt
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OWN_SCORES = BASE_DIR / "data" / "parsed_qa" / "evasion_scores.csv"

# Ordinal label orders, least to most evasive.
ORDER = {
    "evasionbench": ["direct", "intermediate", "fully_evasive"],
    "qevasion": ["Clear Reply", "Ambivalent", "Clear Non-Reply"],
}
HUMAN = {"evasionbench": False, "qevasion": True}
RNG = np.random.default_rng(20260819)
N_SPLITS = 200


def own_tertiles():
    """Pre-registered cut points: tertiles of our own corpus, label-blind."""
    s = pd.read_csv(OWN_SCORES)["evasion_score"].astype(float)
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


def macro_f1_for_cuts(score, y, c1, c2):
    pred = np.digitize(score, [c1, c2])
    return f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)


def best_cuts(score, y, grid):
    best, argbest = -1.0, (grid[0], grid[1])
    for c1, c2 in combinations(grid, 2):
        f = macro_f1_for_cuts(score, y, c1, c2)
        if f > best:
            best, argbest = f, (c1, c2)
    return argbest


def auroc_ci(y_bin, score, n_boot=2000, seed=20260819):
    """Bootstrap CI with its own generator.

    Previously this drew from the module-level RNG, which made the interval
    depend on how much randomness earlier code had already consumed -- editing
    an unrelated analysis silently shifted the published CI. It now seeds
    locally so the interval is a function of the data alone.
    """
    if len(np.unique(y_bin)) < 2:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y_bin, score)
    idx = np.arange(len(y_bin))
    boots = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_bin[b])) < 2:
            continue
        boots.append(roc_auc_score(y_bin[b], score[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return point, (lo, hi)


def analyse(corpus):
    path = RESULTS_DIR / f"external_scores_{corpus}.csv"
    if not path.exists():
        print(f"[skip] {path.name} not found")
        return

    df = pd.read_csv(path)
    order = ORDER[corpus]
    df = df[df["ext_label"].isin(order)].copy()
    df["y"] = df["ext_label"].map({lab: i for i, lab in enumerate(order)})
    score = df["evasion_score"].to_numpy(float)
    y = df["y"].to_numpy(int)

    L = []
    def log(s=""):
        L.append(s)
        print(s)

    log(f"External benchmark: {corpus}")
    log("=" * 68)
    log(f"n scored                 : {len(df)}")
    log(f"label source             : "
        f"{'HUMAN annotation' if HUMAN[corpus] else 'MODEL-GENERATED (Eva-4B-V2)'}")
    if not HUMAN[corpus]:
        log("  NOTE: agreement below is cross-system agreement with a fine-tuned")
        log("        specialist model. It is NOT external human validation.")
    log(f"class counts             : {df['ext_label'].value_counts().reindex(order).to_dict()}")
    log(f"judge score mean (SD)    : {score.mean():.2f} ({score.std(ddof=1):.2f})")
    log("")
    log("mean judge score by external label (monotone = the measure orders correctly)")
    for lab in order:
        sub = score[df["ext_label"].to_numpy() == lab]
        log(f"  {lab:<18} n={len(sub):<5} mean={sub.mean():7.2f}  SD={sub.std(ddof=1):6.2f}")
    log("")

    if corpus == "qevasion":
        import itertools
        from sklearn.metrics import cohen_kappa_score as _ck
        src = BASE_DIR / "data" / "external" / "qevasion_test.csv"
        if src.exists():
            ann = pd.read_csv(src)[["annotator1", "annotator2", "annotator3"]].dropna()
            labs = sorted(set(ann.to_numpy().ravel()))
            ks = [_ck(ann[a], ann[b], labels=labs)
                  for a, b in itertools.combinations(ann.columns, 2)]
            log("-" * 68)
            log("HUMAN CEILING  (their own annotators, on their own labels)")
            log("-" * 68)
            log(f"  n with all three annotators   : {len(ann)}")
            for (a, b), k in zip(itertools.combinations(ann.columns, 2), ks):
                log(f"  {a} vs {b:<12}: Cohen kappa = {k:.3f}")
            log(f"  MEAN pairwise kappa           : {np.mean(ks):.3f}")
            log(f"  all three identical           : "
                f"{(ann.nunique(axis=1) == 1).mean()*100:.1f}%")
            log("")

    log("-" * 68)
    log("THRESHOLD-FREE  (no cut points -- the safest numbers)")
    log("-" * 68)
    rho, p_rho = spearmanr(score, y)
    log(f"  Spearman rho vs ordinal label : {rho:+.3f}  (p = {p_rho:.3g})")
    top = (y == len(order) - 1).astype(int)
    a, (lo, hi) = auroc_ci(top, score)
    log(f"  AUROC, '{order[-1]}' vs rest  : {a:.3f}  "
        f"[95% CI {lo:.3f}, {hi:.3f}]   ({top.sum()} positives)")
    nontop = (y >= 1).astype(int)
    a2, (lo2, hi2) = auroc_ci(nontop, score)
    log(f"  AUROC, any-evasion vs direct   : {a2:.3f}  [95% CI {lo2:.3f}, {hi2:.3f}]")
    log("")

    log("-" * 68)
    log("ZERO-SHOT ORDINAL  (cut points pre-registered from our own corpus)")
    log("-" * 68)
    c1, c2 = own_tertiles()
    log(f"  cut points (our corpus tertiles): {c1:.2f}, {c2:.2f}")
    log("  chosen without reference to any external label -- nothing fitted")
    pred = np.digitize(score, [c1, c2])
    log(f"  Macro-F1                        : "
        f"{f1_score(y, pred, average='macro', labels=[0,1,2], zero_division=0):.3f}")
    log(f"  quadratic-weighted kappa        : "
        f"{cohen_kappa_score(y, pred, labels=[0,1,2], weights='quadratic'):.3f}")
    log(f"  accuracy                        : {(pred == y).mean():.3f}")
    log("")

    log("-" * 68)
    log("THRESHOLD-FIT  (30% fit / 70% held out -- fitted, and labelled as such)")
    log("-" * 68)
    log(f"Repeated over {N_SPLITS} random splits rather than reported from one.")
    log("A single split of this size carries roughly 0.03 SD in held-out Macro-F1,")
    log("enough that one draw can look materially better or worse than the method")
    log("actually is. The candidate cut-point grid is built from the FIT half only,")
    log("so the held-out scores inform nothing.")
    log("")
    f1s, qwks, accs, cuts = [], [], [], []
    for si in range(N_SPLITS):
        rng = np.random.default_rng(1000 + si)
        idx = rng.permutation(len(df))
        k = int(round(0.30 * len(df)))
        fit_i, test_i = idx[:k], idx[k:]
        grid = np.unique(np.percentile(score[fit_i], np.arange(5, 100, 2.5)))
        c1f, c2f = best_cuts(score[fit_i], y[fit_i], grid)
        pred_t = np.digitize(score[test_i], [c1f, c2f])
        f1s.append(f1_score(y[test_i], pred_t, average="macro", labels=[0, 1, 2],
                            zero_division=0))
        qwks.append(cohen_kappa_score(y[test_i], pred_t, labels=[0, 1, 2],
                                      weights="quadratic"))
        accs.append((pred_t == y[test_i]).mean())
        cuts.append((c1f, c2f))
    f1s, qwks, accs = np.array(f1s), np.array(qwks), np.array(accs)
    k = int(round(0.30 * len(df)))
    log(f"  fit on n={k}, tested on n={len(df)-k}, x{N_SPLITS} splits")
    log(f"  median fitted cut points        : "
        f"{np.median([c[0] for c in cuts]):.2f}, {np.median([c[1] for c in cuts]):.2f}")
    log(f"  Macro-F1  (held out)            : {f1s.mean():.3f} "
        f"(SD {f1s.std():.3f}) [5th {np.percentile(f1s,5):.3f}, "
        f"95th {np.percentile(f1s,95):.3f}]")
    log(f"  quadratic-weighted kappa        : {qwks.mean():.3f} (SD {qwks.std():.3f})")
    log(f"  accuracy  (held out)            : {accs.mean():.3f} (SD {accs.std():.3f})")
    log("")

    out = RESULTS_DIR / f"external_benchmark_{corpus}.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}\n")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["qevasion", "evasionbench"]
    for c in targets:
        analyse(c)
