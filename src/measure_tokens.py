"""
Measure the real chars-per-token ratio for this corpus.

cost_throughput.py previously applied a generic ~4 chars/token approximation
and reported a range, because Anthropic's tokenizer is the only correct source
and no API key was available. This script calls that tokenizer
(client.messages.count_tokens) on a random sample of prompts built exactly the
way judge.py builds them, and reports the measured ratio that
cost_throughput.py then applies to the full corpus.

count_tokens does not run inference and is not billed as generation, so this is
cheap to re-run.

Usage:
    python src/measure_tokens.py --n 200
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
SCORES_CSV = BASE_DIR / "data" / "parsed_qa" / "evasion_scores.csv"


def load_dotenv():
    p = BASE_DIR / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (checked environment and .env)")

    import anthropic
    import judge

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    df = pd.read_csv(SCORES_CSV)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    toks, chars = [], []
    for _, r in sample.iterrows():
        user = judge.USER_PROMPT_TMPL.format(question=r["question_text"],
                                             response=r["response_text"])
        res = client.messages.count_tokens(
            model=judge.MODEL,
            system=judge.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        toks.append(res.input_tokens)
        chars.append(len(judge.SYSTEM_PROMPT) + len(user))

    toks, chars = np.array(toks), np.array(chars)
    ratio = chars.sum() / toks.sum()

    print(f"model                : {judge.MODEL}")
    print(f"prompts sampled      : {len(toks)} (seed {args.seed})")
    print(f"input tokens         : mean {toks.mean():.1f}, median "
          f"{np.median(toks):.0f}, total {toks.sum():,}")
    print(f"characters           : total {chars.sum():,}")
    print(f"MEASURED chars/token : {ratio:.3f}")
    print()
    print("Set CHARS_PER_TOKEN in src/cost_throughput.py to this value.")


if __name__ == "__main__":
    main()
