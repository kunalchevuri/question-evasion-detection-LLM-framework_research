# question-evasion-detection-LLM-framework_research

LLM-as-Judge framework for detecting institutional question evasion in earnings call Q&A, and testing whether it predicts subsequent market reaction.

## Research question

Do management teams that evade analyst questions on earnings calls see a different market reaction (3-day cumulative abnormal return around the next earnings announcement) than teams that answer directly? An LLM judge scores every analyst question / management response pair in a corpus of SEC-filed earnings call transcripts on four dimensions of evasiveness (non-responsiveness, vagueness, deflection, hedging), and the resulting evasion measures are tested against forward market reaction, accounting fundamentals, and linguistic sentiment.

## Headline result

**Null.** On the final analytical panel (155 firm-quarter observations, 32 companies, filing years 2015-2023), none of the three evasion measures (mean, variance, max evasion score) correlate significantly with 3-day CAR (all \|r\| < 0.05, all p > 0.55), and adding evasion score to a classification model (accounting + sentiment features → CAR direction) does not improve out-of-sample AUROC beyond what a bootstrap 95% CI attributes to chance. This null result is stable under 5-fold cross-validation and under excluding the 2020 (COVID) filing year. See `results/final_robustness_checks.txt` and `results/final_correlation_matrix.csv` for the full analysis.

## Pipeline

1. **Scrape** (`scraper.py`, `fmp_scraper.py`) — retrieve earnings call transcripts filed with the SEC.
2. **Parse** (`parser.py`, `parse_new_transcripts.py`, `backfill_tickers.py`) — isolate the Q&A section, pair analyst questions with management responses, resolve company tickers.
3. **Filter** (`filter_before_scoring.py`, `prune_and_confirm.py`) — exclude REITs/SPACs and single-transcript companies before spending on LLM scoring.
4. **Score** (`judge.py`) — LLM-as-judge (Claude) rates every Q&A pair on 4 dimensions (1-5 scale each), rescaled to a 0-100 evasion score.
5. **Build features** (`features.py`, `build_features_new_transcripts.py`) — merge transcript-level evasion measures with forward CAR (yfinance), accounting fundamentals (SEC XBRL, prior-quarter values only), and Loughran-McDonald sentiment into `data/features/master_panel.csv`.
6. **Model** (`models.py`) — nested logistic regression / XGBoost configs (accounting → + sentiment → + evasion) on a chronological train/test split, with bootstrap CI on the incremental value of evasion score.
7. **Robustness** (`robustness_checks.py`, `final_verification.py`) — 5-fold CV, COVID-exclusion sensitivity, power analysis, company concentration, sector composition.
8. **Human validation** (`sample_validation.py`, `merge_annotations.py`, `compute_kappa.py`) — two independent human raters score a stratified sample of 75 pairs; Cohen's kappa (unweighted + quadratic-weighted) and Pearson r against the LLM judge.
9. **Cross-model validation** (`multi_llm_judge.py`, `compare_llm_judges.py`, `plot_cross_rater_agreement.py`) — the same 75 pairs re-scored by a second LLM (Groq-hosted `llama-3.3-70b-versatile`) to test whether the evasion construct generalizes across model architectures rather than being specific to the primary judge model.

`final_verification.py` re-derives every headline number directly from the current files on disk with no reliance on cached output — the intended entry point for confirming the analysis before any figure or number goes into the paper.

## Repository structure

```
src/            all pipeline scripts (see Pipeline above)
data/
  raw_transcripts/    scraped .htm filings
  parsed_qa/          parsed Q&A pairs + LLM evasion scores
  features/           master_panel.csv (final modeling panel)
validation/     human-annotated samples + second-LLM scores for cross-validation
results/        all figures, tables, and text output from modeling/robustness/verification scripts
```

## Data sources

- Earnings call transcripts: SEC EDGAR filings
- Market data: Yahoo Finance (`yfinance`)
- Accounting fundamentals: SEC XBRL structured data (`data.sec.gov/api/xbrl`)
- Sentiment dictionary: Loughran-McDonald financial sentiment word lists
- LLM judges: Claude (Anthropic), Llama 3.3 70B (Groq)

## Reproduction Steps

1. Clone the repo and install dependencies:
   ```
   git clone <repo-url>
   cd question-evasion-detection-LLM-framework_research
   pip install -r requirements.txt
   ```
2. If re-running any LLM scoring step (`judge.py`, `multi_llm_judge.py`), set the relevant API key(s) as environment variables:
   ```
   export ANTHROPIC_API_KEY=...
   export GROQ_API_KEY=...   # only needed for the cross-model validation step
   ```
   All data already produced by these steps (parsed Q&A pairs, evasion scores, human/cross-LLM annotations) is committed to the repo, so this step is optional unless you want to regenerate it from scratch.
3. Reproduce every headline number in the paper from the files currently on disk:
   ```
   python src/final_verification.py
   ```
   This is the canonical entry point — it re-derives provenance counts, descriptive statistics, the correlation matrix, classification results (including the Table 2 bootstrap CI), robustness checks (5-fold CV, COVID-year exclusion, power analysis), and the human-validation overlap directly from `data/` and `results/`, with no reliance on cached or previously printed output.

## License

MIT — see `LICENSE`.
