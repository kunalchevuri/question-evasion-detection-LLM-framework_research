"""
Cross-model validation: score the same 75 human-validated Q&A pairs
(validation/human_annotation.csv) with a second LLM, reusing the EXACT rubric
prompt from judge.py so this is a fair comparison against the original
Claude judge. Supports multiple providers (--provider groq|openai) so
results from different models can be collected side by side without
overwriting each other -- each run writes to its own --output path.

Reads ONLY validation/human_annotation.csv -- the same file already used by
compute_kappa.py, which bundles question_text/response_text/Claude scores/
human scores together in a single row per pair. pair_id is that file's own
row index, preserved through filtering. This is deliberate: sample_
validation.py's docstring documents a prior bug where reconstructing pair
alignment via groupby('transcript_id').cumcount() across separately-ordered
files silently scrambled rows. Keying everything off human_annotation.csv's
own row index (never rebuilt, never joined against a differently-ordered
file) avoids that failure mode entirely.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import SYSTEM_PROMPT, USER_PROMPT_TMPL  # noqa: E402 -- exact prompt reuse

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / "validation"
HUMAN_ANNOTATION_CSV = VALIDATION_DIR / "human_annotation.csv"

MAX_TOKENS = 300
MAX_RETRIES = 5
TEMPERATURE = 0

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# $/1M tokens, (input, output). Verified current as of this run.
PRICING = {
    "groq": {"llama-3.3-70b-versatile": (0.59, 0.79)},  # Groq published rate
    "openai": {"gpt-4o": (2.50, 10.00)},
}


def build_client(provider):
    if provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("GROQ_API_KEY environment variable is not set.")
        return Groq(api_key=api_key)
    elif provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key)
    else:
        raise SystemExit(f"Unknown provider: {provider}")


def call_judge(client, model, question, response_text):
    user_prompt = USER_PROMPT_TMPL.format(question=question, response=response_text)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = msg.choices[0].message.content.strip()
            m = JSON_RE.search(raw)
            if not m:
                raise ValueError(f"No JSON found in response: {raw[:200]}")
            data = json.loads(m.group(0))
            for k in ("non_responsiveness", "vagueness", "deflection", "hedging"):
                data[k] = int(data[k])
            usage = getattr(msg, "usage", None)
            data["_prompt_tokens"] = getattr(usage, "prompt_tokens", None) if usage else None
            data["_completion_tokens"] = getattr(usage, "completion_tokens", None) if usage else None
            return data
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")


def load_75_validated_pairs():
    df = pd.read_csv(HUMAN_ANNOTATION_CSV)
    h1_filled = df["human1_evasion_score"].notna() & (df["human1_evasion_score"].astype(str).str.strip() != "")
    h2_filled = df["human2_evasion_score"].notna() & (df["human2_evasion_score"].astype(str).str.strip() != "")
    both_filled = h1_filled & h2_filled
    subset = df.loc[both_filled].copy()
    print(f"Loaded {len(df)} total rows from {HUMAN_ANNOTATION_CSV}; "
          f"{len(subset)} have both human1 and human2 scores (the validated 75).")
    return subset


def estimate_cost(subset, provider, model, sleep_seconds):
    avg_q_chars = subset["question_text"].str.len().mean()
    avg_r_chars = subset["response_text"].str.len().mean()
    approx_input_tokens_per_call = (avg_q_chars + avg_r_chars) / 4 + 350  # + system prompt/template overhead
    approx_output_tokens_per_call = 100  # short JSON completion

    n = len(subset)
    total_input_tokens = approx_input_tokens_per_call * n
    total_output_tokens = approx_output_tokens_per_call * n

    price_in, price_out = PRICING[provider][model]
    cost_in = total_input_tokens / 1e6 * price_in
    cost_out = total_output_tokens / 1e6 * price_out
    total_cost = cost_in + cost_out

    est_seconds = n * sleep_seconds
    print(f"\nCost estimate ({provider}/{model}, n={n} pairs):")
    print(f"  Approx input tokens/call:  {approx_input_tokens_per_call:.0f}")
    print(f"  Approx output tokens/call: {approx_output_tokens_per_call:.0f}")
    print(f"  Est. total input tokens:   {total_input_tokens:,.0f}  (${cost_in:.4f})")
    print(f"  Est. total output tokens:  {total_output_tokens:,.0f}  (${cost_out:.4f})")
    print(f"  ESTIMATED TOTAL COST:      ${total_cost:.4f}")
    print(f"  Est. wall-clock (sleep only): ~{est_seconds:.0f}s")
    return total_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    parser.add_argument("--errors-output", required=True, help="Path to write failure log CSV")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    client = build_client(args.provider)
    print(f"Using model: {args.model} ({args.provider} API)\n")

    subset = load_75_validated_pairs()

    if args.provider in PRICING and args.model in PRICING[args.provider]:
        estimate_cost(subset, args.provider, args.model, args.sleep)
    else:
        print(f"\nNo cached pricing for {args.provider}/{args.model} -- skipping pre-run cost estimate.")

    rows = []
    errors = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, (idx, r) in enumerate(subset.iterrows(), 1):
        try:
            scores = call_judge(client, args.model, r["question_text"], r["response_text"])
            evasion_score = (
                (scores["non_responsiveness"] + scores["vagueness"] + scores["deflection"] + scores["hedging"]) / 4
                - 1
            ) / 4 * 100
            if scores.get("_prompt_tokens"):
                total_prompt_tokens += scores["_prompt_tokens"]
            if scores.get("_completion_tokens"):
                total_completion_tokens += scores["_completion_tokens"]
            rows.append({
                "transcript_id": r["transcript_id"],
                "pair_id": idx,
                "model_name": args.model,
                "non_responsiveness": scores["non_responsiveness"],
                "vagueness": scores["vagueness"],
                "deflection": scores["deflection"],
                "hedging": scores["hedging"],
                "evasion_score": round(evasion_score, 2),
            })
        except Exception as exc:
            errors.append({"transcript_id": r["transcript_id"], "pair_id": idx, "error": str(exc)})
            print(f"  [{i}/{len(subset)}] FAILED pair_id={idx} transcript={r['transcript_id']}: {exc}")

        time.sleep(args.sleep)

        if i % args.progress_every == 0:
            print(f"  [{i}/{len(subset)}] scored so far: {len(rows)}, failed: {len(errors)}")

    out_path = Path(args.output)
    err_path = Path(args.errors_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["transcript_id", "pair_id", "model_name", "non_responsiveness",
                  "vagueness", "deflection", "hedging", "evasion_score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} scored pairs -> {out_path}")

    if errors:
        with open(err_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["transcript_id", "pair_id", "error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"Logged {len(errors)} failure(s) -> {err_path}")
    else:
        print("No failures.")

    print(f"\n{'=' * 60}")
    print(f"Scored : {len(rows)} / {len(subset)}")
    print(f"Failed : {len(errors)} / {len(subset)}")
    if args.provider in PRICING and args.model in PRICING[args.provider] and total_prompt_tokens:
        price_in, price_out = PRICING[args.provider][args.model]
        actual_cost = (total_prompt_tokens / 1e6 * price_in) + (total_completion_tokens / 1e6 * price_out)
        print(f"Actual prompt tokens:     {total_prompt_tokens:,}")
        print(f"Actual completion tokens: {total_completion_tokens:,}")
        print(f"ACTUAL COST:              ${actual_cost:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
