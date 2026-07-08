# Legal LLM Safety Metrics

DS6051 hackathon project. We evaluate a general-purpose open-weight LLM
(Gemma 4 E2B-it) **as if it were deployed as a legal AI assistant**, and build a
safety scorecard for that use-case: what to measure, why it matters for legal
deployment, how to measure it, and what each measurement cannot tell you.

Team: <!-- TODO: names -->

## Scorecard

| Metric | Result | How it's measured |
|---|---|---|
| Citation accuracy | TBD | LLM-as-judge |
| Factual consistency | TBD | LLM-as-judge, multi-turn |
| Sycophancy | TBD | Automated flip-rate under user pushback |
| Calibration | TBD (ECE) | Automated, answer-token logits |
| Cross-lingual shift | TBD (EN vs ES vs EU accuracy) | Automated, paired translated dataset |

Full justifications, limitations, and the results discussion: see
[DELIVERABLE.md](DELIVERABLE.md) (filled in as runs complete).

## Data

`data/subset_100_{en,es,eu}.json` — 100 questions from MMLU `professional_law`
(test split, shuffled seed 42, first 100), in English plus Spanish and Basque
translations. Items are paired by `id`; `answer` indices are identical across
languages. Schema: `{id, question, choices[4], answer}`.

Basque is our low-resource language for the cross-lingual safety
distribution-shift metric. Translation methodology and caveats:
[data/TRANSLATION_NOTES.md](data/TRANSLATION_NOTES.md).

Regenerate the English subset:
```python
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "professional_law")["test"]
subset = ds.shuffle(seed=42).select(range(100))
```

## Repo layout

```
data/        evaluation datasets (EN/ES/EU) + translation notes
evals/       one script per metric (see evals/README.md)
results/     metric outputs + final results table
reference/   course-provided boilerplates (inference, LLM-as-judge)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # gemma + shieldgemma are gated; accept licenses on HF first
```

## How to run

Each metric is a standalone script (see `evals/README.md`); outputs land in
`results/`. <!-- TODO: exact commands once scripts exist -->
