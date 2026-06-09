# `data/` — human and rewritten reviews + judge scores

## Review conditions

| Condition | Description |
|---|---|
| `human_reviews` | Original human peer reviews (R) |
| `rewritten` | Meaning-preserving LLM rewrites (R′) — same judgments, new wording |

## `reviews/` — input JSONL

```
reviews/
├── human_reviews/{venue}{year}.jsonl
└── transformations/rewritten/{venue}{year}_{provider}_{model}.jsonl
```

Paper-level keys: `venue`, `year`, `paper_id`, `forum_id`, `title`, `decision`, `reviews`.

Human review fields: `full_review_text`, `rating`, `confidence`, `review_id`.

Rewritten review fields: `text` (scored), `original_text`, `transformation` (`rewrite`),
`idea_origin` (`human`), `text_origin` (`ai`), `rewrite_model`.

## `results/` — judge scores

```
results/
├── human_reviews/{venue}{year}_{judge}.json   # S_m(R)
└── rewritten/{venue}{year}/{judge}.json       # S_m(R′) + ScholarPeer
```

`{judge}` is `gpt5mini` or `gemini`.

### Point-wise scores (`llm_judge`)

Nested under `ReviewEval`, `REMOR`, `RottenReviews` (list per review):

**ReviewEval** (9): `existing_literature_comparison`, `methodological_scrutiny`,
`results_interpretation`, `theoretical_contributions`, `logical_gaps_identification`,
`overall_depth`, `total_insights`, `actionable_insights`, `actionability_score`

**REMOR** (8): `criticism`, `example`, `importance_and_relevance`,
`materials_and_methods`, `praise`, `presentation_and_reporting` *(excluded from analysis)*,
`results_and_discussion`, `suggestion_and_solution`

**RottenReviews** (10 per review): `comprehensiveness`, `usage_of_technical_terms`
*(excluded)*, `objectivity`, `fairness`, `actionability`, `constructiveness`,
`relevance_alignment`, `clarity_and_readability` *(excluded)*, `overall_quality`,
`overall_score_100`

### ScholarPeer (`ScholarPeer.results[]`)

Comparative AI-vs-human scores on 1–10 scale (5 = human baseline):
`technical_accuracy`, `constructive_value`, `analytical_depth`,
`novelty_and_significance`, `overall`.

## Analysis inputs

`pipeline/collect_pairs.py` computes paired differences:
- Point-wise metrics: `diff = S_m(R′) − S_m(R)`
- ScholarPeer: `diff = score − 5.0`

Three metrics are excluded a priori from the 29-metric analysis (style/presentation
by design): Presentation & Reporting, Usage of Technical Terms, Clarity & Readability.
