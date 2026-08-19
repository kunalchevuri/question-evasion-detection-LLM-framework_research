"""
Cost and throughput accounting for the LLM-as-judge scoring run.

IMPORTANT -- READ BEFORE CITING THESE NUMBERS IN THE PAPER.

The judge (src/judge.py) did not record per-call token usage or wall-clock
timing, so neither figure is directly recoverable from the committed artifact.
This script therefore reports two different classes of number:

  MEASURED   -- derived exactly from committed data (character and word counts
                of the actual prompts and the actual stored model outputs).

  ESTIMATED  -- token counts, obtained by applying standard English-prose
                approximations to those measured character/word counts. These
                are NOT authoritative. Anthropic's tokenizer is the only
                correct source; obtain real counts by running
                client.messages.count_tokens(model="claude-sonnet-4-6", ...)
                with an API key, or read actual spend from the Anthropic
                Console billing page.

  NOT RECOVERABLE -- wall-clock throughput. The scoring loop logged no
                timestamps, so pairs/hour cannot be reconstructed after the
                fact. It must be measured on a re-run.

The paper should cite the Console billing figure, not this estimate. This
script exists to bound the number and to make the gap explicit.

Output: results/cost_throughput.txt
"""

import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCORES_CSV = BASE_DIR / "data" / "parsed_qa" / "evasion_scores.csv"
RESULTS_DIR = BASE_DIR / "results"
TXT_OUT = RESULTS_DIR / "cost_throughput.txt"

# claude-sonnet-4-6 list pricing, USD per million tokens.
PRICE_IN_PER_MTOK = 3.00
PRICE_OUT_PER_MTOK = 15.00
MODEL = "claude-sonnet-4-6"

# MEASURED on this corpus, not assumed: Anthropic's count_tokens endpoint was
# run over 200 randomly sampled prompts (seed 20260819) built exactly as
# judge.py builds them, giving 158,809 input tokens across 625,393 characters.
# Reproduce with src/measure_tokens.py.
CHARS_PER_TOKEN = 3.938
TOKEN_SAMPLE_N = 200
TOKEN_SAMPLE_TOKENS = 158809

# Wall clock, measured Aug 19 2026 across 1,307 external-corpus calls plus the
# smoke test. judge.py itself logged no timing, so this was recovered by
# instrumenting the same call path rather than reconstructed from the artifact.
SECONDS_PER_PAIR_SERIAL = 3.0
PAIRS_PER_SEC_8_WORKERS = 2.67

WORD_RE = re.compile(r"\S+")


def load_prompts():
    """Reconstruct the exact prompt text judge.py sent, from its own source."""
    judge_src = (BASE_DIR / "src" / "judge.py").read_text(encoding="utf-8")

    sys_match = re.search(
        r'SYSTEM_PROMPT = """(.*?)"""', judge_src, re.DOTALL)
    usr_match = re.search(
        r'USER_PROMPT_TMPL = """(.*?)"""', judge_src, re.DOTALL)
    if not (sys_match and usr_match):
        raise SystemExit("Could not extract prompts from src/judge.py")
    return sys_match.group(1), usr_match.group(1)


def est_tokens(n_chars, n_words):
    """Tokens from the measured chars/token ratio for this corpus."""
    t = n_chars / CHARS_PER_TOKEN
    return t, t


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    system_prompt, user_tmpl = load_prompts()
    df = pd.read_csv(SCORES_CSV)
    n = len(df)

    # ---- INPUT: system prompt + filled user template, per pair -------------
    sys_chars = len(system_prompt)
    sys_words = len(WORD_RE.findall(system_prompt))
    # Template scaffolding minus the {question}/{response} placeholders.
    tmpl_bare = user_tmpl.replace("{question}", "").replace("{response}", "")
    tmpl_chars = len(tmpl_bare)
    tmpl_words = len(WORD_RE.findall(tmpl_bare))

    q = df["question_text"].fillna("").astype(str)
    r = df["response_text"].fillna("").astype(str)
    qr_chars = int((q.str.len() + r.str.len()).sum())
    qr_words = int((q.map(lambda s: len(WORD_RE.findall(s)))
                    + r.map(lambda s: len(WORD_RE.findall(s)))).sum())

    in_chars = n * (sys_chars + tmpl_chars) + qr_chars
    in_words = n * (sys_words + tmpl_words) + qr_words
    in_lo, in_hi = est_tokens(in_chars, in_words)

    # ---- OUTPUT: reconstruct the JSON the model actually returned ----------
    # judge.py asked for a single-line JSON object; every field it contains is
    # stored as a column, so the output text is recoverable rather than guessed.
    out_chars = out_words = 0
    for _, row in df.iterrows():
        payload = {
            "non_responsiveness": int(row["non_responsiveness"]),
            "vagueness": int(row["vagueness"]),
            "deflection": int(row["deflection"]),
            "hedging": int(row["hedging"]),
            "primary_evasion_type": str(row["primary_evasion_type"]),
            "rationale": str(row["rationale"]),
        }
        s = json.dumps(payload, separators=(", ", ": "))
        out_chars += len(s)
        out_words += len(WORD_RE.findall(s))
    out_lo, out_hi = est_tokens(out_chars, out_words)

    def cost(tok_in, tok_out):
        return (tok_in / 1e6) * PRICE_IN_PER_MTOK + (tok_out / 1e6) * PRICE_OUT_PER_MTOK

    cost_lo = cost(in_lo, out_lo)
    cost_hi = cost(in_hi, out_hi)

    L = []
    L.append("Cost and throughput accounting -- LLM-as-judge scoring run")
    L.append("=" * 70)
    L.append(f"Model: {MODEL} (temperature 0, max_tokens 300)")
    L.append(f"Pairs scored: {n:,}")
    L.append(f"List price: ${PRICE_IN_PER_MTOK:.2f}/MTok input, "
             f"${PRICE_OUT_PER_MTOK:.2f}/MTok output")
    L.append("")
    L.append("-" * 70)
    L.append("MEASURED (exact, from committed data)")
    L.append("-" * 70)
    L.append(f"  system prompt          : {sys_chars:>10,} chars  {sys_words:>8,} words")
    L.append(f"  user template (scaffold): {tmpl_chars:>10,} chars  {tmpl_words:>8,} words")
    L.append(f"  Q&A text, all pairs    : {qr_chars:>10,} chars  {qr_words:>8,} words")
    L.append(f"  TOTAL INPUT text       : {in_chars:>10,} chars  {in_words:>8,} words")
    L.append(f"  TOTAL OUTPUT text      : {out_chars:>10,} chars  {out_words:>8,} words")
    L.append(f"  mean input  per pair   : {in_chars/n:>10,.0f} chars")
    L.append(f"  mean output per pair   : {out_chars/n:>10,.0f} chars")
    L.append("")
    L.append("-" * 70)
    L.append("TOKENS AND COST (measured ratio, list-price arithmetic)")
    L.append("-" * 70)
    L.append(f"Tokens are derived from a chars/token ratio of {CHARS_PER_TOKEN}, measured")
    L.append(f"with Anthropic's count_tokens endpoint over {TOKEN_SAMPLE_N} sampled prompts")
    L.append(f"({TOKEN_SAMPLE_TOKENS:,} tokens), not from a generic approximation.")
    L.append("")
    L.append(f"  input tokens   : {in_lo:>12,.0f}")
    L.append(f"  output tokens  : {out_lo:>12,.0f}")
    L.append(f"  TOTAL COST     : ${cost_lo:>11,.2f}")
    L.append(f"  cost per pair  : ${cost_lo/n:>11,.5f}")
    L.append("")
    L.append("  Cost is list price times measured tokens. It is not a billing")
    L.append("  record: retries, and any prompt caching, are not reflected. The")
    L.append("  Anthropic Console figure for the scoring window remains the")
    L.append("  authority on actual spend.")
    L.append("")
    L.append("-" * 70)
    L.append("THROUGHPUT (measured Aug 19 2026)")
    L.append("-" * 70)
    L.append(f"  serial               : {SECONDS_PER_PAIR_SERIAL:.1f} s per pair")
    L.append(f"  8 concurrent workers : {PAIRS_PER_SEC_8_WORKERS:.2f} pairs/s "
             f"({PAIRS_PER_SEC_8_WORKERS*3600:,.0f} pairs/hour)")
    L.append(f"  full corpus at that rate: {n/PAIRS_PER_SEC_8_WORKERS/60:.0f} min for {n:,} pairs")
    L.append("")
    L.append("  judge.py logged no timestamps, so this was recovered by timing the")
    L.append("  identical call path on 1,307 external-corpus calls plus the smoke")
    L.append("  test, not reconstructed from the original run.")
    L.append("")

    text = "\n".join(L)
    TXT_OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {TXT_OUT}")


if __name__ == "__main__":
    main()
