# Judging a Review by its Cover

This repository contains the code, data, and precomputed results for the paper
*Judging a Review by its Cover: A Reliability Analysis of LLM-based Peer
Review Evaluation Metrics*.

The paper (and this repo) tests whether automatic LLM-based metrics for
peer-review quality actually measure the **content** of a review or just
react to its **writing style**. The intuition: if a metric gives different
scores to an original human review and to a rewrite of the same review that
preserves the content but changes the wording, then the metric is being
fooled by surface form.

## How the analysis works

For every metric and every (original, rewrite) review pair:

1. Score the original human review **R** and its rewritten version **R′**
   with the metric.
2. Compute the per-pair difference `d = score(R′) − score(R)`.
3. Run two statistical tests over the population of differences:

   | Test | Question it answers | Verdict |
   |---|---|---|
   | **Wilcoxon signed-rank** | Does rewriting systematically shift scores? | If yes → metric is **Sensitive** to writing style |
   | **TOST equivalence** | Is the shift small enough to be practically negligible? | If yes → metric is **Robust** under rewriting |

Both tests use a Bonferroni-corrected threshold of `α = 0.05 / 29 ≈ 0.00172`.
A metric can end up Sensitive only, Robust only, both, or neither.

The pipeline runs this analysis over **29 content-oriented metrics** drawn
from four prior peer-review evaluation works (ReviewEval, REMOR,
RottenReviews, ScholarPeer), using **two judge LLMs** (GPT-5-mini and
Gemini-2.5-Flash) so the conclusions are not specific to one judge.

## Metrics

Drawn from four prior peer-review evaluation works:

| Source | Metrics analyzed |
|---|---|
| **ReviewEval** (9) | Literature Comparison, Methodological Scrutiny, Results Interpretation, Theoretical Contributions, Logical Gaps Identification, Overall Depth, Total Insights, Actionable Insights, Actionability Score |
| **REMOR** (7) | Criticism, Example, Importance & Relevance, Materials & Methods, Praise, Results & Discussion, Suggestion & Solution |
| **RottenReviews** (8) | Comprehensiveness, Objectivity, Fairness, Actionability, Constructiveness, Relevance Alignment, Overall Quality, Overall Score |
| **ScholarPeer** (5) | Technical Accuracy, Constructive Value, Analytical Depth, Novelty & Significance, Overall |

Three style/presentation metrics from these sources are excluded a priori
(Presentation & Reporting from REMOR; Usage of Technical Terms and Clarity
& Readability from RottenReviews), since they explicitly target writing
form rather than content.

The verbatim prompts used to score each metric are in `pipeline/prompts.py`,
and `pipeline/metric_registry.py` holds the canonical metric list, scales,
and exclusion set.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

API keys are only needed if you want to re-score reviews from scratch (last
section). The bundled scores in `data/results/` are enough to reproduce
the published results.

## Step 1 — Run the Wilcoxon + TOST analysis

Two commands, one per judge LLM:

```bash
python pipeline/run_analysis.py           # GPT-5-mini       → results/results_gpt5mini.csv
python pipeline/run_analysis.py --gemini  # Gemini-2.5-Flash → results/results_gemini.csv
```

Each run loads the cached per-review scores from `data/results/`, computes
paired differences, runs the Wilcoxon and TOST tests over the 29 metrics,
and writes the CSV described in Step 2.

## Step 2 — Read the per-metric CSV

Each CSV has 29 rows (one per metric). The columns are the *outputs of the
two tests* run on that metric's paired differences `d = score(R′) − score(R)`
— first the input statistics, then the Wilcoxon result, then the TOST
result, then the final classifications.

| Group | Column | Meaning |
|---|---|---|
| Identifier | `metric`, `source` | Metric name and its source paper |
| Test input | `n_obs` | Number of paired observations the tests were run on |
|  | `mean_delta`, `sd_delta` | Mean and SD of `d` — the summary of the population fed to both tests |
| **Wilcoxon test** (surface sensitivity) | `wilcoxon_p` | Two-sided signed-rank p-value: how likely the observed shift would be if rewriting had no effect |
|  | `sensitive` | `True` iff `wilcoxon_p < 0.05/29` — i.e. rewriting *systematically* changes the score |
| **TOST test** (robustness) | `delta_bound` | Equivalence bound `δ = 0.2 × sd_delta` — the per-metric "negligible-shift" threshold |
|  | `tost_p` | TOST p-value: how likely the true shift is *outside* the equivalence bound |
|  | `robust` | `True` iff `tost_p < 0.05/29` — i.e. the shift is positively shown to be practically negligible |

`run_analysis.py` also prints a region-by-region summary to stdout (how
many metrics fall into each cell of the Sensitive × Robust 2×2 table).

## Re-score reviews with a different judge LLM (optional)

Slow and API-bound. Skip if you only want to reproduce the bundled results.

```bash
export OPENAI_API_KEY=...    # or GEMINI_API_KEY for Gemini

# Score the original human reviews and the rewrites with the point-wise
# metric families (ReviewEval / REMOR / RottenReviews):
python pipeline/run_llm_judge.py pointwise --only-human --sample 0
python pipeline/run_llm_judge.py pointwise --only-rewritten --sample 0

# Score with ScholarPeer (compares the rewrite against the paired human on
# a 1–10 scale):
python pipeline/run_llm_judge.py scholarpeer --full --ttype rewritten --with-paper
```

Common flags: `--judge-model`, `--judge-base-url`, `--judge-api-key-env`,
`--model-filter`, `--sample`, `--skip-existing`.

New scores land in `data/results/`. Re-run `pipeline/run_analysis.py`
afterwards to pick them up.

## Repo layout

```
.
├── pipeline/
│   ├── run_analysis.py        # Wilcoxon + TOST tests → results/*.csv
│   ├── run_llm_judge.py       # CLI to (re-)score reviews with an LLM judge
│   ├── collect_pairs.py       # builds paired diffs from cached scores
│   ├── metric_registry.py     # 29-metric inventory + α correction
│   ├── review_metrics.py      # LLMJudge class (one method per metric family)
│   └── prompts.py             # verbatim prompts from the 4 source papers
├── data/
│   ├── reviews/               # input review text (JSONL)
│   └── results/               # cached per-review judge scores (JSON)
├── human_annotation/          # rewrite-faithfulness validation study
└── results/                   # per-metric CSV outputs of the two tests
```

See `data/README.md` for the JSON/JSONL key schema.

## What's in the data

The dataset is taken from PeerPrism and covers ICLR/NeurIPS 2021–2024.

| Item | Count | Notes |
|---|---:|---|
| Venue-years | 8 | ICLR 2021–2024 + NeurIPS 2021–2024 |
| Papers | 160 | 20 per venue-year |
| Original human reviews **R** | 674 | Pulled from OpenReview, anonymized |
| Meaning-preserving rewrites **R′** | 4,044 | Average of ~6 rewrites per original (one per rewrite model) |
| Rewrite models | 6 | Used to generate `R′`: Gemini-2.5-Flash, GPT-5, o4-mini, Claude-Haiku-4.5, DeepSeek-R1, Llama-4-Scout |
| Judge LLMs | 2 | Used to score reviews: **GPT-5-mini** (primary) and **Gemini-2.5-Flash** (replication) |

What each rewrite *preserves* (so `score(R)` and `score(R′)` are comparable):
the original review's judgments, arguments, strengths, weaknesses, and
recommendations. What it *changes*: wording, phrasing, and presentation.

Files:

- **Inputs** — review text (JSONL):
  `data/reviews/human_reviews/{venue}{year}.jsonl` (8 files, one per venue-year)
  and `data/reviews/rewritten/{venue}{year}_{provider}_{model}.jsonl`
  (48 files = 8 × 6 rewrite models).
- **Cached judge scores** — per-review scores keyed by metric (JSON):
  `data/results/human_reviews/{venue}{year}_{judge}.json` and
  `data/results/rewritten/{venue}{year}/{judge}.json` (32 files = 8 × 2 × 2).

See `data/README.md` for the JSON/JSONL key schema.

## Human annotation

The whole approach assumes the rewrites preserve the original reviews'
content. To validate this, three graduate-student annotators rated 50
(original, rewrite) pairs (each pair seen by two raters) on a 1–5
faithfulness scale and a yes/no "would the author get the same feedback?"
judgment.

`human_annotation/Guidelines.pdf` is the instruction sheet shown to
annotators. The three `annotator_*.csv` files contain their ratings.

## License

Released for research use. Review text in `data/` comes from OpenReview;
redistribution should follow OpenReview's terms.
