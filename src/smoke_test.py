"""
Pipeline smoke test -- Task 0.2 of the IEEE BigData sprint.

Re-scores a small random sample of pairs that were ALREADY scored in the
committed run, using judge.py's own prompt, model, and parsing code, then
compares the fresh scores against the committed ones.

This is the right test to run before any new scoring work. The question is not
"does the API respond" but "does claude-sonnet-4-6 still behave the way it did
when the paper's numbers were produced". If the model drifted, every downstream
external-benchmark number would be measured on a different instrument than the
one the paper describes, and we need to know that before spending API budget.

Nothing is written to data/ -- the committed evasion_scores.csv is read only.
The console transcript is mirrored to results/smoke_test.txt so the go/no-go
gate leaves a durable artifact rather than living only in a terminal buffer.

Usage:
    python src/smoke_test.py            # 10 pairs
    python src/smoke_test.py --n 25
"""

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

SCORES_CSV = BASE_DIR / "data" / "parsed_qa" / "evasion_scores.csv"
OUT = BASE_DIR / "results" / "smoke_test.txt"
DIMS = ["non_responsiveness", "vagueness", "deflection", "hedging"]


def load_dotenv():
    """Read .env into os.environ. .env is gitignored; never commit a key."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="pairs to re-score")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (checked environment and .env)")

    import anthropic
    import judge  # reuse the exact prompt/model/parse path the paper used

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    df = pd.read_csv(SCORES_CSV)
    sample = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    print(f"Smoke test: re-scoring {len(sample)} previously-scored pairs")
    print(f"model={judge.MODEL}  temperature={judge.TEMPERATURE}  "
          f"max_tokens={judge.MAX_TOKENS}  seed={args.seed}\n")

    recs = []
    t0 = time.time()
    for i, r in sample.iterrows():
        try:
            s = judge.call_judge(client, r["question_text"], r["response_text"])
        except Exception as exc:
            print(f"  [{i+1}/{len(sample)}] FAILED: {exc}")
            recs.append({"ok": False})
            continue

        new_score = ((sum(s[d] for d in DIMS) / 4) - 1) / 4 * 100
        rec = {"ok": True, "old_score": float(r["evasion_score"]),
               "new_score": round(new_score, 2)}
        for d in DIMS:
            rec[f"old_{d}"] = int(r[d])
            rec[f"new_{d}"] = int(s[d])
        rec["old_type"] = str(r["primary_evasion_type"])
        rec["new_type"] = str(s["primary_evasion_type"])
        recs.append(rec)
        print(f"  [{i+1}/{len(sample)}] committed={rec['old_score']:6.2f}  "
              f"fresh={rec['new_score']:6.2f}  delta={rec['new_score']-rec['old_score']:+6.2f}")

    elapsed = time.time() - t0
    res = pd.DataFrame([r for r in recs if r.get("ok")])
    n_fail = len(recs) - len(res)

    print("\n" + "=" * 62)
    print("SMOKE TEST RESULT")
    print("=" * 62)
    if res.empty:
        raise SystemExit("All calls failed -- pipeline is NOT clear. Do not proceed.")

    delta = res["new_score"] - res["old_score"]
    exact = (delta.abs() < 1e-9).sum()
    dim_exact = sum(int((res[f"new_{d}"] == res[f"old_{d}"]).sum()) for d in DIMS)
    dim_total = len(res) * len(DIMS)
    type_match = (res["new_type"] == res["old_type"]).sum()

    print(f"  scored / failed        : {len(res)} / {n_fail}")
    print(f"  identical evasion_score: {exact}/{len(res)}")
    print(f"  mean |delta|           : {delta.abs().mean():.2f} points (0-100 scale)")
    print(f"  max  |delta|           : {delta.abs().max():.2f}")
    print(f"  dimension-level exact  : {dim_exact}/{dim_total} "
          f"({100*dim_exact/dim_total:.0f}%)")
    print(f"  primary_type agreement : {type_match}/{len(res)}")
    if len(res) > 1:
        print(f"  correlation old vs new : r = {res['old_score'].corr(res['new_score']):.3f}")
    print(f"  wall clock             : {elapsed:.1f}s "
          f"({elapsed/max(len(res),1):.1f}s per pair)")

    print("\n  VERDICT: ", end="")
    if n_fail:
        print("PARTIAL -- some calls failed; investigate before scoring at scale.")
    elif delta.abs().mean() < 3.0:
        print("CLEAR -- model behavior consistent with the committed run.")
    else:
        print("DRIFT -- scores differ materially. Do NOT treat new external\n"
              "           benchmark numbers as comparable to the paper's without\n"
              "           reporting this shift explicitly.")
    print("=" * 62)


class _Tee:
    """Write to the real stdout and to a buffer, so the run is both watchable
    live and recoverable afterwards."""

    def __init__(self, stream, buf):
        self._stream, self._buf = stream, buf

    def write(self, s):
        self._stream.write(s)
        self._buf.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()


if __name__ == "__main__":
    buf = io.StringIO()
    status = 0
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            main()
    except SystemExit as exc:
        buf.write(chr(10) + "SystemExit: %s" % exc + chr(10))
        status = 1
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Wrote {OUT}")
    sys.exit(status)
