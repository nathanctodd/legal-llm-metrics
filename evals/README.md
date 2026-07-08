# Eval scripts

One standalone script per scorecard metric. Each reads from `../data/`, writes
raw outputs + a summary row to `../results/`. Claim one by putting your name here.

| Script | Metric | Owner | Status |
|---|---|---|---|
| `citation_accuracy.py` | Citation accuracy (LLM-as-judge) | TBD | not started |
| `factual_consistency.py` | Factual consistency, multi-turn (LLM-as-judge) | TBD | not started |
| `sycophancy.py` | Flip rate under user pushback | TBD | not started |
| `crosslingual_calibration.py` | Cross-lingual shift + calibration (one script — same logit runs produce both) | Hudson | not started |
| `judge_comparison.py` | Required by the info sheet: run 2 extra judge models alongside ShieldGemma, compare scores | TBD | not started |

Conventions (so results merge cleanly):
- Load questions from `data/subset_100_{lang}.json`; key everything on `id`.
- For multiple-choice metrics, score via A/B/C/D answer-token logits (one forward
  pass per item) rather than parsing generated text — gives accuracy and a
  confidence number from the same run. See `reference/llm_judge_boilerplate.py`
  for the yes/no-logit version of the trick.
- Save per-item rows (id, language, prediction, confidence, correct) as CSV in
  `results/`, plus print the summary numbers.
