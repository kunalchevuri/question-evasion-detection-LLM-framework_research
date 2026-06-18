"""
LLM-as-Judge: scores every analyst question / management response pair in
data/parsed_qa/all_qa_pairs.csv on a four-dimension evasion rubric.

Rubric (each 1-5, 5 = most evasive):
  non_responsiveness - does the response fail to address what was specifically asked?
  vagueness          - does it avoid concrete, verifiable specifics in favor of generic language?
  deflection          - does it pivot to unrelated talking points / a different topic?
  hedging             - does it explicitly decline to answer, hedge, or defer?

evasion_score = mean of the four dimensions, rescaled to 0-100.

Outputs:
  data/parsed_qa/evasion_scores.csv      - one row per Q&A pair
  data/parsed_qa/transcript_evasion.csv  - aggregated per transcript

Resumable: if evasion_scores.csv already exists, already-scored pairs are
skipped (keyed by transcript_id + question_text), and the run only scores
what's missing. Results are checkpointed to disk every 50 pairs so a crash
mid-run doesn't lose completed work.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import anthropic
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
QA_CSV = BASE_DIR / "data" / "parsed_qa" / "all_qa_pairs.csv"
SCORES_CSV = BASE_DIR / "data" / "parsed_qa" / "evasion_scores.csv"
TRANSCRIPT_CSV = BASE_DIR / "data" / "parsed_qa" / "transcript_evasion.csv"

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 300
CHECKPOINT_EVERY = 50
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are an expert analyst of corporate earnings call communications, evaluating whether a management response evades or directly answers an analyst's question.

Score the response on four dimensions, each an integer 1-5 (5 = most evasive):
1. non_responsiveness: Does the response fail to address what was specifically asked? (1 = fully addresses the question, 5 = completely ignores/talks past it)
2. vagueness: Does the response avoid concrete, verifiable specifics (numbers, dates, named initiatives) in favor of generic language? (1 = highly specific and concrete, 5 = entirely vague/generic)
3. deflection: Does the response pivot to unrelated talking points, unprompted narratives, or a different topic than what was asked? (1 = stays on topic, 5 = heavy redirection)
4. hedging: Does the response explicitly decline to answer, hedge with disclaimers, or defer ("we don't guide on that," "let's take this offline")? (1 = a clear, committed answer, 5 = explicit non-answer/refusal)

Also pick a primary_evasion_type: one of "none", "non_responsive", "vague", "deflection", "hedging" (the dominant pattern, or "none" if the response is not evasive).

Respond with ONLY a single-line JSON object, no markdown formatting, no commentary before or after:
{"non_responsiveness": <1-5>, "vagueness": <1-5>, "deflection": <1-5>, "hedging": <1-5>, "primary_evasion_type": "<type>", "rationale": "<one sentence>"}"""

USER_PROMPT_TMPL = """ANALYST QUESTION:
{question}

MANAGEMENT RESPONSE:
{response}"""

JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def call_judge(client, question, response_text):
    user_prompt = USER_PROMPT_TMPL.format(question=question, response=response_text)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = msg.content[0].text.strip()
            m = JSON_RE.search(raw)
            if not m:
                raise ValueError(f"No JSON found in response: {raw[:200]}")
            data = json.loads(m.group(0))
            for k in ("non_responsiveness", "vagueness", "deflection", "hedging"):
                data[k] = int(data[k])
            return data
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            last_err = exc
            time.sleep(1)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")


def load_done_keys():
    if not SCORES_CSV.exists():
        return [], set()
    existing = pd.read_csv(SCORES_CSV, dtype=str)
    rows = existing.to_dict("records")
    keys = {(r["transcript_id"], r["question_text"]) for r in rows}
    return rows, keys


def save(rows, fieldnames):
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_transcript_aggregate(rows):
    df = pd.DataFrame(rows)
    for col in ("non_responsiveness", "vagueness", "deflection", "hedging", "evasion_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = df.groupby("transcript_id").agg(
        company_ticker=("company_ticker", "first"),
        filing_date=("filing_date", "first"),
        n_pairs=("evasion_score", "count"),
        mean_evasion_score=("evasion_score", "mean"),
        mean_non_responsiveness=("non_responsiveness", "mean"),
        mean_vagueness=("vagueness", "mean"),
        mean_deflection=("deflection", "mean"),
        mean_hedging=("hedging", "mean"),
        pct_high_evasion=("evasion_score", lambda s: (s >= 60).mean() * 100),
    ).reset_index().sort_values("transcript_id")

    agg.to_csv(TRANSCRIPT_CSV, index=False)
    print(f"Wrote {TRANSCRIPT_CSV} ({len(agg)} transcripts)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N unscored pairs (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(QA_CSV, dtype=str)
    print(f"Loaded {len(df)} Q&A pairs from {QA_CSV}")

    fieldnames = list(df.columns) + [
        "non_responsiveness", "vagueness", "deflection", "hedging",
        "evasion_score", "primary_evasion_type", "rationale", "judge_model",
    ]

    rows, done_keys = load_done_keys()
    if done_keys:
        print(f"Resuming: {len(done_keys)} pairs already scored in {SCORES_CSV}")

    todo = [r for _, r in df.iterrows() if (r["transcript_id"], r["question_text"]) not in done_keys]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Scoring {len(todo)} pairs (model={MODEL}, temperature={TEMPERATURE}, max_tokens={MAX_TOKENS})\n")

    n_done = 0
    n_failed = 0
    for i, r in enumerate(todo, 1):
        try:
            scores = call_judge(client, r["question_text"], r["response_text"])
            evasion_score = (
                (scores["non_responsiveness"] + scores["vagueness"] + scores["deflection"] + scores["hedging"]) / 4
                - 1
            ) / 4 * 100  # rescale mean(1-5) -> 0-100

            row = dict(r)
            row.update(
                non_responsiveness=scores["non_responsiveness"],
                vagueness=scores["vagueness"],
                deflection=scores["deflection"],
                hedging=scores["hedging"],
                evasion_score=round(evasion_score, 2),
                primary_evasion_type=scores["primary_evasion_type"],
                rationale=scores["rationale"],
                judge_model=MODEL,
            )
            rows.append(row)
            n_done += 1
        except Exception as exc:
            n_failed += 1
            print(f"  [{i}/{len(todo)}] FAILED transcript={r['transcript_id']}: {exc}")
            continue

        if i % CHECKPOINT_EVERY == 0:
            save(rows, fieldnames)
            avg = sum(float(x["evasion_score"]) for x in rows) / len(rows)
            print(f"  [{i}/{len(todo)}] checkpointed {len(rows)} total rows | running mean evasion_score={avg:.1f}")

    save(rows, fieldnames)
    print(f"\n{'='*60}")
    print(f"Scored this run : {n_done}")
    print(f"Failed this run  : {n_failed}")
    print(f"Total in file    : {len(rows)}")
    print(f"Output           : {SCORES_CSV}")
    print('='*60)

    if rows:
        write_transcript_aggregate(rows)


if __name__ == "__main__":
    main()
