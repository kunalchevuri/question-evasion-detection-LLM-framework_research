"""
Prompt-robustness check: re-score the same 75 human-validated pairs with two
reworded variants of judge.py's rubric prompt, to test whether evasion_score
reflects the underlying construct or a specific prompt's quirks.

Variant A: same four dimensions/1-5 scale/output schema, each dimension's
description reworded to different phrasing.
Variant B: original wording verbatim, but the four dimensions presented in a
different order (hedging, deflection, vagueness, non_responsiveness) in both
the rubric list and the JSON schema, to test order sensitivity specifically.

Reads pairs from validation/human_annotation.csv (same 75-pair source used
throughout this validation work) and compares each variant's evasion_score
against that file's own "evasion_score" column (the ORIGINAL judge.py-prompt
Claude scores), matched by that file's own row index -- no cross-file join.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import MODEL, TEMPERATURE, MAX_TOKENS, MAX_RETRIES, USER_PROMPT_TMPL  # noqa: E402
from compute_kappa import BIN_LABELS, bin_codes, interpret_kappa  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / "validation"
RESULTS_DIR = BASE_DIR / "results"
HUMAN_ANNOTATION_CSV = VALIDATION_DIR / "human_annotation.csv"
OUTPUT_TXT = RESULTS_DIR / "prompt_robustness_check.txt"

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

PROMPT_A = """You are an expert analyst of corporate earnings call communications, evaluating whether a management response evades or directly answers an analyst's question.

Score the response on four dimensions, each an integer 1-5 (5 = most evasive):
1. non_responsiveness: Does the answer sidestep the actual question that was asked, rather than engaging with it? (1 = directly engages with what was asked, 5 = never engages with the actual question)
2. vagueness: Is the answer light on concrete detail -- hard numbers, dates, or named plans -- relying instead on broad, generic phrasing? (1 = concrete and detailed, 5 = purely generic and abstract)
3. deflection: Does the reply steer toward a different subject, an unsolicited talking point, or a tangent rather than staying on the question that was asked? (1 = remains focused on the topic, 5 = substantially redirected elsewhere)
4. hedging: Does the speaker openly decline to commit to an answer, pile on disclaimers, or push the matter off to a later time (e.g. "we won't comment on that," "let's follow up separately")? (1 = gives a direct, committed response, 5 = a clear refusal or non-answer)

Also pick a primary_evasion_type: one of "none", "non_responsive", "vague", "deflection", "hedging" (the dominant pattern, or "none" if the response is not evasive).

Respond with ONLY a single-line JSON object, no markdown formatting, no commentary before or after:
{"non_responsiveness": <1-5>, "vagueness": <1-5>, "deflection": <1-5>, "hedging": <1-5>, "primary_evasion_type": "<type>", "rationale": "<one sentence>"}"""

PROMPT_B = """You are an expert analyst of corporate earnings call communications, evaluating whether a management response evades or directly answers an analyst's question.

Score the response on four dimensions, each an integer 1-5 (5 = most evasive):
1. hedging: Does the response explicitly decline to answer, hedge with disclaimers, or defer ("we don't guide on that," "let's take this offline")? (1 = a clear, committed answer, 5 = explicit non-answer/refusal)
2. deflection: Does the response pivot to unrelated talking points, unprompted narratives, or a different topic than what was asked? (1 = stays on topic, 5 = heavy redirection)
3. vagueness: Does the response avoid concrete, verifiable specifics (numbers, dates, named initiatives) in favor of generic language? (1 = highly specific and concrete, 5 = entirely vague/generic)
4. non_responsiveness: Does the response fail to address what was specifically asked? (1 = fully addresses the question, 5 = completely ignores/talks past it)

Also pick a primary_evasion_type: one of "none", "non_responsive", "vague", "deflection", "hedging" (the dominant pattern, or "none" if the response is not evasive).

Respond with ONLY a single-line JSON object, no markdown formatting, no commentary before or after:
{"hedging": <1-5>, "deflection": <1-5>, "vagueness": <1-5>, "non_responsiveness": <1-5>, "primary_evasion_type": "<type>", "rationale": "<one sentence>"}"""

VARIANTS = {"variant_a_reworded": PROMPT_A, "variant_b_reordered": PROMPT_B}

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def call_judge(client, system_prompt, question, response_text):
    user_prompt = USER_PROMPT_TMPL.format(question=question, response=response_text)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = msg.content[0].text.strip()
            m = JSON_RE.search(raw)
            if not m:
                raise ValueError(f"No JSON found: {raw[:200]}")
            data = json.loads(m.group(0))
            for k in ("non_responsiveness", "vagueness", "deflection", "hedging"):
                data[k] = int(data[k])
            return data
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")


def load_75():
    df = pd.read_csv(HUMAN_ANNOTATION_CSV)
    h1 = df["human1_evasion_score"].notna() & (df["human1_evasion_score"].astype(str).str.strip() != "")
    h2 = df["human2_evasion_score"].notna() & (df["human2_evasion_score"].astype(str).str.strip() != "")
    subset = df.loc[h1 & h2].copy()
    subset["evasion_score"] = pd.to_numeric(subset["evasion_score"], errors="coerce")
    subset = subset.dropna(subset=["evasion_score"])
    return subset


def score_variant(client, subset, variant_name, system_prompt):
    log(f"\nScoring {variant_name} ({len(subset)} pairs)...")
    scores = {}
    n_fail = 0
    for i, (idx, r) in enumerate(subset.iterrows(), 1):
        try:
            d = call_judge(client, system_prompt, r["question_text"], r["response_text"])
            evasion_score = (
                (d["non_responsiveness"] + d["vagueness"] + d["deflection"] + d["hedging"]) / 4 - 1
            ) / 4 * 100
            scores[idx] = round(evasion_score, 2)
        except Exception as exc:
            n_fail += 1
            log(f"  [{i}/{len(subset)}] FAILED pair_id={idx}: {exc}")
        time.sleep(0.5)
        if i % 20 == 0:
            log(f"  [{i}/{len(subset)}] scored: {len(scores)}, failed: {n_fail}")
    log(f"Done: {len(scores)}/{len(subset)} scored, {n_fail} failed")
    return scores


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    subset = load_75()
    log(f"Loaded {len(subset)} validated pairs from {HUMAN_ANNOTATION_CSV}")
    log(f"Model: {MODEL}, temperature={TEMPERATURE}\n")

    original = subset["evasion_score"]

    results_summary = []
    for variant_name, prompt in VARIANTS.items():
        hr_title = f"{'=' * 70}\n{variant_name}\n{'=' * 70}"
        log("\n" + hr_title)
        scores = score_variant(client, subset, variant_name, prompt)

        common_idx = [i for i in subset.index if i in scores]
        variant_scores = pd.Series({i: scores[i] for i in common_idx})
        orig_scores = original.loc[common_idx]

        r, p = pearsonr(variant_scores, orig_scores)

        variant_bins = bin_codes(variant_scores / 10.0)
        orig_bins = bin_codes(orig_scores / 10.0)
        bin_range = list(range(len(BIN_LABELS)))
        qwk = cohen_kappa_score(variant_bins, orig_bins, labels=bin_range, weights="quadratic")

        log(f"\n--- {variant_name} vs ORIGINAL prompt (n={len(common_idx)}) ---")
        log(f"Pearson r = {r:+.4f}  (p={p:.4e})")
        log(f"Quadratic-weighted kappa = {qwk:.4f}  ({interpret_kappa(qwk)})")

        results_summary.append({"variant": variant_name, "n": len(common_idx), "pearson_r": r,
                                 "pearson_p": p, "quadratic_kappa": qwk})

    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)
    summary_df = pd.DataFrame(results_summary)
    log(summary_df.to_string(index=False))

    log("\n" + "=" * 70)
    log("INTERPRETATION")
    log("=" * 70)
    mean_r = summary_df["pearson_r"].mean()
    mean_qwk = summary_df["quadratic_kappa"].mean()
    log(f"Mean correlation with original across both variants: r={mean_r:+.4f}")
    log(f"Mean quadratic-weighted kappa with original across both variants: {mean_qwk:.4f}")
    log(
        f"\n{'The scores are STABLE across reasonable prompt rewording -- both variants correlate strongly with the original (r > 0.7), supporting the claim that evasion_score reflects a real, prompt-independent signal rather than one specific phrasing.' if mean_r > 0.7 else 'The scores show MODERATE stability across prompt rewording -- correlations with the original are positive but not uniformly strong, suggesting some sensitivity to exact prompt wording that should be acknowledged as a limitation.' if mean_r > 0.4 else 'The scores are NOT stable across prompt rewording -- low correlation with the original suggests meaningful sensitivity to prompt phrasing, which is a real limitation for the construct validity claim.'}"
    )

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")
    log(f"\nSaved full log -> {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
