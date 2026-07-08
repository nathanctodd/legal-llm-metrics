# Deliverable Plan — Legal AI Safety Scorecard (DS6051 Hackathon)

Submission: Devpost — project overview, GitHub repo, results table, written
discussion, ~5-minute video. Peer feedback due EOD July 10.

Framing: the provided general-purpose LLM is treated as if it were built as a
**legal AI assistant**. The deliverable is a safety scorecard: what we measure,
why it matters for legal deployment, how we measure it, and what each
measurement cannot tell us.

## Devpost checklist
- [ ] Project overview (short: use-case, model, the 5 metrics, headline findings)
- [ ] GitHub repo, structured so others can follow (see repo layout below)
- [ ] Results table (scorecard format below, one row per metric)
- [ ] Written discussion (per-metric: how evaluated, result, what it can/can't tell you)
- [ ] ~5-minute video
- [ ] Required by day-of flow step 4: judge pipeline compared across **three** judge models (ShieldGemma + two more) — confirm a teammate owns this
- [ ] Bonus (optional): propose one new safety metric for the legal domain

## Scorecard skeleton (required table format)

| Metric | Result | Why it's needed for a legal assistant | How measured | Limitations |
|---|---|---|---|---|
| Citation accuracy | TBD | Fabricated case citations are the canonical legal-AI failure (sanctioned attorneys have filed AI-invented cases). A legal assistant that cites confidently but falsely is worse than none. | LLM-as-judge: does each cited case/statute exist and support the claim? | Judge can only assess plausibility/consistency, not verify against a real legal database; judge shares training data with the target model. |
| Factual consistency | TBD | Legal advice that contradicts itself across a conversation destroys reliability; users act on it. | Multi-turn prompts; LLM-as-judge flags contradictions between turns. | Judge misses subtle doctrinal contradictions; consistency ≠ correctness (consistently wrong scores well). |
| Sycophancy | TBD | Clients push back ("are you sure? I think I CAN break the lease"). A legal assistant must not flip correct answers under social pressure. | Automated: MMLU professional_law items, re-ask with user disagreement, measure flip rate on initially-correct answers. | Flip rate on multiple-choice proxies real advisory pressure imperfectly; only tests one pressure phrasing. |
| Calibration | TBD (ECE + confidence-accuracy plot) | Deployment gate: the assistant must know when to defer to a human lawyer. Overconfident wrong answers are the danger mode. | Automated: A/B/C/D answer-token logits → confidence; compare to accuracy (ECE, per-language). | Token-logit confidence ≠ stated verbal confidence users actually see; MC format only. |
| **Cross-lingual shift (main piece)** | TBD (EN vs ES vs EU accuracy + per-language calibration) | Equal access: non-English speakers are heavy users of legal aid. If competence degrades in Spanish or collapses in Basque (low-resource, per doc recommendation) the same product is silently unsafe for those users. | Automated: same 100 MMLU professional_law items, professionally consistent translations paired by id (`data/subset_100_{en,es,eu}.json`), same logit pipeline per language. | Translation itself introduces variance (see `data/TRANSLATION_NOTES.md`: untranslatable common-law terms keep English in parentheses, which may understate the shift; translations not validated by native speakers). Accuracy shift ≠ full safety shift (we measure competence gap, not refusal gap). |

## Results table inputs needed (from eval runs)
- Per-language accuracy on the 100 paired items (EN/ES/EU) + per-question flips
- Per-language mean confidence + ECE
- Sycophancy flip rate (EN at minimum; ES/EU if time — cross-lingual sycophancy is a nice interaction finding)
- Judge-model agreement table (3 judges) for the judge-based metrics

## Repo layout (GitHub)
```
README.md            ← project overview + results table + how to run
data/                ← subset_100_{en,es,eu}.json, TRANSLATION_NOTES.md
evals/               ← one script per metric
results/             ← raw outputs + final results table (csv/md)
```

## ~5-minute video outline
1. (0:30) Use-case: general-purpose LLM deployed as a legal assistant; why that's high-stakes
2. (1:00) The scorecard: 5 metrics, one-line justification each (What/Why)
3. (1:00) How we measured: dataset, logit-based scoring, LLM-as-judge + 3-judge comparison (How)
4. (1:30) Results walk: table, then the cross-lingual shift chart as centerpiece (EN→ES→EU accuracy stairstep + a concrete question that flips in Basque)
5. (0:45) Limitations + what the scorecard cannot tell you; deployment recommendation
6. (0:15) Bonus metric proposal (if included)

## Discussion section skeleton (per metric, per rubric)
For each metric: what we measured → result → what it means for deploying a
legal assistant → what the measurement can and cannot tell you. The doc grades
on explicitly grappling with What/Why/How and architectural limitations — every
metric writeup should name at least one architecture-rooted limitation (e.g.
logit confidence exists because the model is a token-probability machine, which
is also why verbalized confidence and internal confidence diverge).

## Open items
- [ ] Confirm nobody is fine-tuning (doc: evaluation only, no training)
- [ ] Who owns the 3-judge comparison
- [ ] Who records/edits the video; who writes which discussion sections
- [ ] Devpost page owner
