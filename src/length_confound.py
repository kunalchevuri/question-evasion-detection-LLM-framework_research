"""
Is the judge just measuring response length?

The first objection a benchmarking reviewer raises about any continuous
evasiveness score is that it might be a proxy for verbosity: short answers get
called evasive, long ones substantive. If that were true, the external
validation in Section V would be measuring nothing but word count.

This checks it the direct way. For each external corpus we correlate raw
response length (word count) with the external ordinal label, and compare that
against the correlation the judge's score achieves on the same items. If length
alone tracked the label as well as the judge does, the judge would be adding
nothing.

Requires the raw external corpora in data/external/, which are NOT
redistributed here (see README): the committed score files deliberately carry
no source text. Re-download them with the commands in the README to reproduce.

Output: results/length_confound.txt
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
EXTERNAL = BASE / "data" / "external"
RESULTS = BASE / "results"
OUT = RESULTS / "length_confound.txt"

# Same source specs as score_external.py, plus the ordinal label order.
SPECS = {
    "evasionbench": {
        "path": EXTERNAL / "evasionbench_full.parquet",
        "id": "uid",
        "answer": "answer",
        "order": ["direct", "intermediate", "fully_evasive"],
        "pretty": "EvasionBench",
    },
    "qevasion": {
        "path": EXTERNAL / "qevasion_test.csv",
        "id": "index",
        "answer": "interview_answer",
        "order": ["Clear Reply", "Ambivalent", "Clear Non-Reply"],
        "pretty": "CLARITY / QEvasion",
    },
}


def main():
    lines = []

    def log(s=""):
        lines.append(s)
        print(s)

    log("Length as a confound: does the judge just count words?")
    log("=" * 68)
    log("Spearman rho against the external ordinal label, for raw response")
    log("length and for the judge's evasion score, on identical items.")
    log("")

    for name, spec in SPECS.items():
        scored = pd.read_csv(RESULTS / f"external_scores_{name}.csv")
        if not spec["path"].exists():
            log(f"{spec['pretty']}: SKIPPED -- {spec['path'].name} not present.")
            log("  The raw corpora are not redistributed; see README to re-download.")
            log("")
            continue

        raw = (pd.read_parquet(spec["path"]) if spec["path"].suffix == ".parquet"
               else pd.read_csv(spec["path"]))
        raw = raw[[spec["id"], spec["answer"]]].copy()
        raw.columns = ["ext_id", "response_text"]

        # ext_id round-trips through CSV as a string on one corpus and an int on
        # the other; match on the string form so the join cannot silently empty.
        raw["ext_id"] = raw["ext_id"].astype(str)
        scored = scored.copy()
        scored["ext_id"] = scored["ext_id"].astype(str)

        df = scored.merge(raw, on="ext_id", how="inner")
        if len(df) != len(scored):
            raise SystemExit(
                f"{name}: joined {len(df)} of {len(scored)} scored rows -- "
                "id mismatch, refusing to report a partial correlation.")

        df["response_words"] = df["response_text"].str.split().str.len()
        df["label_ord"] = df["ext_label"].map(
            {v: i for i, v in enumerate(spec["order"])})
        df = df.dropna(subset=["response_words", "label_ord", "evasion_score"])

        r_len, p_len = spearmanr(df["response_words"], df["label_ord"])
        r_jud, p_jud = spearmanr(df["evasion_score"], df["label_ord"])
        r_cross, p_cross = spearmanr(df["response_words"], df["evasion_score"])

        log("-" * 68)
        log(f"{spec['pretty']}  (n = {len(df)})")
        log("-" * 68)
        log(f"  median response length          : {df['response_words'].median():.0f} words")
        log(f"  length      vs. external label  : rho = {r_len:+.3f}  (p = {p_len:.3g})")
        log(f"  judge score vs. external label  : rho = {r_jud:+.3f}  (p = {p_jud:.3g})")
        log(f"  length      vs. judge score     : rho = {r_cross:+.3f}  (p = {p_cross:.3g})")
        log("")

    log("=" * 68)
    log("READ: where length carries essentially no rank information about the")
    log("label and the judge carries a great deal, the score is not a verbosity")
    log("proxy. Any residual length-score correlation is expected and benign --")
    log("terse non-answers are genuinely shorter than substantive ones.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
