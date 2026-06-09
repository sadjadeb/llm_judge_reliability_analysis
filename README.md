# Judging a Review by its Cover

Code and data for paper: Judging a Review by its Cover: A Reliability Analysis of
LLM-based Peer Review Evaluation Metrics


## Research question

Do content-oriented LLM-as-judge metrics respond to **how** a review is
written, or only to **what** it evaluates?

We test this with two complementary criteria:

| Criterion | Test | Interpretation |
|---|---|---|
| **Surface sensitivity** | Wilcoxon signed-rank on paired diffs | Statistically detectable score shift under rewriting |
| **Robustness** | TOST equivalence (δ = 0.2 × SD) | Shift is practically negligible |

For each metric *m* and pair *(R, R′)*:

- Point-wise metrics: **d = S_m(R′) − S_m(R)**
- ScholarPeer: **d = score − 5.0** (5 = human baseline)

Bonferroni correction: **α_corrected = 0.05 / 29 ≈ 0.00172**.

## Metrics (32 scored → 29 analyzed)

Metrics come from four prior works. Three style/presentation metrics are
excluded a priori from the analysis.

| Source | Analyzed metrics |
|---|---|
| **ReviewEval** (9) | Literature Comparison, Methodological Scrutiny, Results Interpretation, Theoretical Contributions, Logical Gaps Identification, Overall Depth, Total Insights, Actionable Insights, Actionability Score |
| **REMOR** (7) | Criticism, Example, Importance & Relevance, Materials & Methods, Praise, Results & Discussion, Suggestion & Solution |
| **RottenReviews** (8) | Comprehensiveness, Objectivity, Fairness, Actionability, Constructiveness, Relevance Alignment, Overall Quality, Overall Score |
| **ScholarPeer** (5) | Technical Accuracy, Constructive Value, Analytical Depth, Novelty & Significance, Overall |

Excluded from analysis: Presentation & Reporting (REMOR), Usage of Technical
Terms and Clarity & Readability (RottenReviews).

## Data scope

Only two review conditions are included:

| Condition | Role |
|---|---|
| `human_reviews` | Original human reviews — baseline scores S_m(R) |
| `rewritten` | Faithful LLM rewrites — scores S_m(R′) |

- **Venues:** ICLR and NeurIPS, 2021–2024 (8 venue-years)
- **Rewrite models:** 6 generators (Gemini, GPT-5, o4-mini, Claude Haiku, DeepSeek R1, Llama Scout)
- **Judge models:** GPT-5-mini (primary), Gemini-2.5-Flash (replication)

See `data/README.md` for every JSON/JSONL key.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── pipeline/                     # all code
│   ├── review_metrics.py         # LLMJudge + ReviewEvaluationPipeline
│   ├── prompts.py                # verbatim LLM-judge prompts
│   ├── run_llm_judge.py          # score reviews (pointwise + scholarpeer)
│   ├── metric_registry.py        # 29-metric inventory, scales, exclusions
│   ├── collect_pairs.py          # build paired diffs from saved scores
│   ├── run_analysis.py           # Wilcoxon + TOST → Table 1
│   └── convert_results.py        # one-time script used to build this archive
├── data/
│   ├── reviews/                  # input review text (JSONL)
│   └── results/                  # precomputed judge scores (JSON)
├── human_annotation/             # rewrite-faithfulness annotation study
│   ├── Guidelines.pdf            # instructions shown to annotators
│   ├── annotator_1.csv           # 50 pairs
│   ├── annotator_2.csv           # 25 pairs
│   └── annotator_3.csv           # 25 pairs
└── results/                      # statistical test outputs (CSV)
```

### What each pipeline file does

| File | Purpose |
|---|---|
| `review_metrics.py` | Scores a review with all 32 metrics via `LLMJudge` |
| `prompts.py` | Verbatim prompts from ReviewEval, REMOR, RottenReviews, ScholarPeer |
| `run_llm_judge.py` | CLI to (re-)run scoring on human or rewritten reviews |
| `metric_registry.py` | Metric names, keys, scales, α correction constants |
| `collect_pairs.py` | Loads `data/results/` and computes per-metric paired diffs |
| `run_analysis.py` | Runs Wilcoxon + TOST per metric; writes Table 1 |
| `convert_results.py` | Setup utility — not needed for normal use |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...    # or GEMINI_API_KEY for the Gemini judge
```

## Scoring reviews (optional — scores are precomputed)

```bash
# Score original human reviews (baseline)
python pipeline/run_llm_judge.py pointwise --only-human --sample 0

# Score rewritten reviews (all 6 generator models per year)
python pipeline/run_llm_judge.py pointwise --only-rewritten --sample 0

# ScholarPeer comparative scoring (rewritten vs paired human)
python pipeline/run_llm_judge.py scholarpeer --full --ttype rewritten --with-paper
```

`LLMJudge` wraps any OpenAI-compatible endpoint (`base_url=` for Gemini).
Scores are written under `data/results/human_reviews/` and
`data/results/rewritten/`.

## Reproducing the surface-sensitivity & robustness tests

The 29 retained metrics are scored once and cached under `data/results/`; a
single script then computes the paired Wilcoxon and TOST tests that produce
Table 1.

### What the two tests do

For each metric *m* and original / rewrite pair *(R, R′)* we form the paired
difference

- Point-wise metrics:&nbsp;&nbsp;`d_i = S_m(R′_i) − S_m(R_i)`
- ScholarPeer:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`d_i = score_i − 5.0`&nbsp;&nbsp;(5 = human baseline)

| Test | Null hypothesis | Decision (reject H₀) |
|---|---|---|
| **Wilcoxon** signed-rank (two-sided) | median(d) = 0 | Metric is **Sensitive** to surface rewriting |
| **TOST** one-sample equivalence | \|μ\| ≥ δ_m | Metric is **Robust** under rewriting |

Both tests use the same Bonferroni-corrected threshold

```
α_corrected = 0.05 / 29 ≈ 0.00172
```

and the per-metric equivalence bound

```
δ_m = 0.2 × SD(d_i)        (Cohen's small-effect convention)
```

so the same robustness criterion applies across metrics with different scoring
ranges.

### Run the tests

```bash
# Primary judge: GPT-5-mini  (reproduces paper Table 1)
python pipeline/run_analysis.py
# → results/sensitivity_results_gpt5mini.csv

# Replication judge: Gemini-2.5-Flash
python pipeline/run_analysis.py --gemini
# → results/sensitivity_results_gemini.csv
```

What the script does in order:

1. Calls `collect_pairs.collect_observations(judge=…)` to load human + rewritten
   scores from `data/results/` and compute every paired diff.
2. Drops the 3 a-priori excluded style metrics (`EXCLUDED_KEYS` in
   `pipeline/metric_registry.py`).
3. For each of the 29 retained metrics: computes `σ`, `δ`, mean Δ, Wilcoxon `p`,
   TOST `p`, and the `sensitive` / `robust` booleans.
4. Prints a region-by-region summary to stdout and writes the CSV below.

### Output files

| File | Content |
|---|---|
| `results/results_gpt5mini.csv` | 29 rows × 11 cols — GPT-5-mini judge |
| `results/results_gemini.csv` | Same shape — Gemini-2.5-Flash judge |

Columns in both files: `metric_key, metric, source, n_obs, mean_delta, sd_delta,
delta_bound, wilcoxon_p, tost_p, sensitive, robust`.

### Reading a result row

The joint outcome of Wilcoxon + TOST partitions each metric into one of four
regions used in the paper's Figure 1:

| Wilcoxon | TOST | Region | Interpretation |
|:---:|:---:|---|---|
| ✓ | — | **Sensitive only** | Detectable shift; too large to call negligible |
| — | ✓ | **Robust only** | No detectable shift; equivalence positively established |
| ✓ | ✓ | **Sensitive ∩ Robust** | Detectable but practically negligible shift |
| — | — | **Inconclusive** | No detectable shift, but equivalence not established |



The script picks the judge by reading the matching JSON files in
`data/results/`:

- `data/results/human_reviews/{venue}{year}_{gpt5mini|gemini}.json` — baseline `S_m(R)`
- `data/results/rewritten/{venue}{year}/{gpt5mini|gemini}.json` — `S_m(R′)` and ScholarPeer

If you re-score with a different judge model, drop the new JSONs into the same
layout and re-run `run_analysis.py`.

## Human validation of rewrites

The whole analysis assumes that each rewritten review **R′** preserves the
evaluative content of the original review **R**. To check this, three
computer-science graduate students annotated a random sample of 50
original–rewrite pairs, with each pair seen by two annotators (100
annotations total). The full guidelines shown to annotators are in
[`human_annotation/Guidelines.pdf`](human_annotation/Guidelines.pdf); a
summary:

| # | Question | Scale | Meaning |
|---|---|---|---|
| **Q1** | Content Faithfulness | 1–5 Likert | How faithfully the rewrite preserves the key evaluation points |
| **Q2** | Overall Equivalence | Yes / No | Whether a paper author would receive essentially the same feedback from both reviews |

Annotators were instructed to focus on **evaluative content** (what the
reviewer thinks about the paper), not wording or style — a rewrite that uses
different words but makes the same points should be rated faithful.

### Files

```
human_annotation/
├── Guidelines.pdf            # full annotator instructions
├── annotator_1.csv           # 50 pairs   (Q1=1–5, Q2=Yes/No)
├── annotator_2.csv           # 25 pairs
└── annotator_3.csv           # 25 pairs
```

CSV columns: `pair_id, generator, venue, year, decision, q1_faithfulness,
q2_equivalence, notes, updated_at` (annotator_3 also has a leading
`annotator` column).

## License

Released for research use. Review text in `data/` comes from OpenReview;
redistribution should follow OpenReview's terms.
