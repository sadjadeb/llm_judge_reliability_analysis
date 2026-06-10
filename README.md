# Judging a Review by its Cover

Code and precomputed results for the paper
*Judging a Review by its Cover: A Reliability Analysis of LLM-based Peer Review
Evaluation Metrics*.

This README is a **how-to-use-the-code** guide. For the motivation, full
analytical framework, and discussion of findings, see the paper itself.

## What you can do with this repo

| Goal | Command |
|---|---|
| Reproduce Table 1 (GPT-5-mini) | `python pipeline/run_analysis.py` |
| Reproduce the Gemini replication | `python pipeline/run_analysis.py --gemini` |
| Re-score reviews with a different judge | `python pipeline/run_llm_judge.py ...` |
| Inspect a per-metric paired-diff CSV | open `results/results_{gpt5mini,gemini}.csv` |
| Inspect raw judge scores per review | open any `data/results/{condition}/{year}/{judge}.json` |

Everything needed for the first two rows is already in the repo — no API key
needed.

## End-to-end data flow

```
data/reviews/*.jsonl   ──┐
                         │   pipeline/run_llm_judge.py   (slow, LLM calls)
                         ▼
              data/results/{cond}/{year}/{judge}.json    (cached scores)
                         │
                         │   pipeline/collect_pairs.py → pipeline/run_analysis.py
                         ▼
              results/results_{judge}.csv                (Table 1 rows)
```

The cached scores in `data/results/` are the expensive part. If you only want
to reproduce Table 1, you skip `run_llm_judge.py` entirely and just run
`run_analysis.py`.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── pipeline/                     # all code
│   ├── review_metrics.py         # LLMJudge + ReviewEvaluationPipeline
│   ├── prompts.py                # verbatim LLM-judge prompts (4 source papers)
│   ├── run_llm_judge.py          # CLI: score reviews (pointwise + scholarpeer)
│   ├── metric_registry.py        # 29-metric inventory, scales, α correction
│   ├── collect_pairs.py          # build paired diffs from saved scores
│   ├── run_analysis.py           # Wilcoxon + TOST → Table 1
│   └── convert_results.py        # one-time helper used when building this archive
├── data/
│   ├── README.md                 # JSON/JSONL schema for every input/output key
│   ├── reviews/                  # input review text (JSONL)
│   └── results/                  # precomputed judge scores (JSON)
├── human_annotation/             # rewrite-faithfulness annotation study
│   ├── Guidelines.pdf
│   ├── annotator_1.csv
│   ├── annotator_2.csv
│   └── annotator_3.csv
└── results/                      # final statistical-test outputs (CSV)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Only needed if you want to re-score reviews (the cached scores are already here):
export OPENAI_API_KEY=...        # for the GPT-5-mini judge
export GEMINI_API_KEY=...        # for the Gemini-2.5-Flash judge
```

## Path 1: reproduce Table 1 from the cached scores

The fastest, fully offline path. Two commands, one per judge.

```bash
# GPT-5-mini judge — produces results/results_gpt5mini.csv
python pipeline/run_analysis.py

# Gemini-2.5-Flash judge — produces results/results_gemini.csv
python pipeline/run_analysis.py --gemini
```

### What "Table 1" actually contains

Table 1 of the paper is a 29-row per-metric report: each metric (drawn from
ReviewEval, REMOR, RottenReviews, ScholarPeer) gets one row with
`σ`, `δ`, mean Δ, two p-values, two booleans, and a cross-judge consistency
flag. The CSVs produced by `run_analysis.py` carry every column needed to
rebuild the table — see the column-by-column meanings in the **Output** block
below.

### The two tests, in one sentence each

For each metric *m* we form per-pair differences
`d_i = S_m(R′_i) − S_m(R_i)` (point-wise metrics) or `d_i = score_i − 5.0`
(ScholarPeer, anchored at the human baseline). Then:

| Test | Null hypothesis | What rejection means |
|---|---|---|
| **Wilcoxon signed-rank** (two-sided) | `median(d) = 0` — the metric reacts the same way to rewrites and originals | The metric is **Sensitive**: rewriting systematically shifts the score |
| **TOST** equivalence (one-sample, max of two one-sided tests) | `\|μ\| ≥ δ_m` — the true shift is at least the equivalence bound `δ_m = 0.2 × SD(d)` | The metric is **Robust**: the shift is small enough to be practically negligible (Cohen's *d* < 0.2) |

Both tests use the same Bonferroni-corrected threshold
**α = 0.05 / 29 ≈ 0.00172** (`ALPHA_CORRECTED` in `metric_registry.py`).
The two tests are independent — a metric can be neither, only one, or both
(see the four-region classification at the end of this section).

### What `run_analysis.py` does step-by-step

1. Loads cached scores from `data/results/human_reviews/` and
   `data/results/rewritten/` via `collect_pairs.collect_observations(judge=…)`.
2. Computes the paired differences described above for every (metric, paper,
   generator) triple.
3. Drops the 3 a-priori-excluded style/presentation metrics
   (`EXCLUDED_KEYS` in `pipeline/metric_registry.py`: Presentation &
   Reporting, Usage of Technical Terms, Clarity & Readability).
4. For each of the 29 retained metrics computes `σ`, `δ`, mean Δ,
   `wilcoxon_p`, `tost_p`, and sets the `sensitive` / `robust` flags by
   comparing each p-value to `ALPHA_CORRECTED`.
5. Prints a region-by-region summary to stdout and writes the CSV.

### Expected input

Both commands read **only** from `data/results/` — no review text, no API
calls. Required files:

| Path | Role | n files |
|---|---|---|
| `data/results/human_reviews/{venue}{year}_{judge}.json` | Baseline scores `S_m(R)` for the 8 venue-years | 16 (8 × 2 judges) |
| `data/results/rewritten/{venue}{year}/{judge}.json` | Rewrite scores `S_m(R′)` per generator model, plus `ScholarPeer.results[]` array | 16 (8 × 2 judges) |

Each `{judge}.json` is a dict keyed by generator-model (e.g.
`ICLR2024_openai_gpt-5`); inside each block, `per_paper[i].llm_judge`
nests the four metric families (`ReviewEval`, `REMOR`, `RottenReviews`,
`ScholarPeer`). See `data/README.md` for the full key schema.

### Expected output

```
results/
├── results_gpt5mini.csv   # produced by  python pipeline/run_analysis.py
└── results_gemini.csv     # produced by  python pipeline/run_analysis.py --gemini
```

Each CSV has 29 rows (one per retained metric) and 11 columns:

| Column | Type | Meaning |
|---|---|---|
| `metric_key` | str | Internal key, e.g. `ReviewEval__overall_depth` |
| `metric` | str | Display name from Table 1 |
| `source` | str | One of `ReviewEval`, `REMOR`, `RottenReviews`, `ScholarPeer` |
| `n_obs` | int | Number of paired observations used in the tests |
| `mean_delta` | float | Mean of `d_i` (column "Mean Δ" in Table 1) |
| `sd_delta` | float | `σ_m` = standard deviation of `d_i` |
| `delta_bound` | float | `δ_m = 0.2 × σ_m` (TOST equivalence bound) |
| `wilcoxon_p` | float | Two-sided Wilcoxon signed-rank p-value |
| `tost_p` | float | TOST p-value (max of two one-sided tests) |
| `sensitive` | bool | `wilcoxon_p < ALPHA_CORRECTED` |
| `robust` | bool | `tost_p < ALPHA_CORRECTED` |

Stdout also prints a region-by-region summary (counts in each cell of the
2×2 classification matrix and the totals).

### Reading a row — the four-region classification

The joint outcome of Wilcoxon + TOST partitions each metric into one of four
regions used in the paper's Figure 1:

| Wilcoxon | TOST | Region | What it means |
|:---:|:---:|---|---|
| ✓ | — | **Sensitive only** | Detectable shift; too large to call negligible |
| — | ✓ | **Robust only** | No detectable shift; equivalence positively established |
| ✓ | ✓ | **Sensitive ∩ Robust** | Detectable but practically negligible shift |
| — | — | **Inconclusive** | No detectable shift, but equivalence not established |

Expected reproduction counts with the bundled scores:

| Judge | Sensitive total | Robust total | Sens ∩ Rob | Inconclusive |
|---|---:|---:|---:|---:|
| GPT-5-mini | 23 | 7 | 1 | 0 |
| Gemini-2.5-Flash | 24 | 7 | 3 | 1 |

## Path 2: re-score reviews with a different judge

Use this if you want to (a) try a new judge model, (b) re-score after
modifying a prompt, or (c) bring your own review JSONLs. `run_llm_judge.py`
has two subcommands:

### `pointwise` — ReviewEval + RottenReviews + REMOR

```bash
# Smoke test (3 papers per generator model)
python pipeline/run_llm_judge.py pointwise --sample 3

# Score original human reviews (baseline S_m(R))
python pipeline/run_llm_judge.py pointwise --only-human --sample 0

# Score rewrites (S_m(R′))
python pipeline/run_llm_judge.py pointwise --only-rewritten --sample 0
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--sample N` | N papers per generator model; `0` = all |
| `--only-human` / `--only-rewritten` | Restrict to one condition |
| `--judge-model <name>` | Override default `gpt-5-mini` |
| `--judge-base-url <url>` + `--judge-api-key-env <ENV>` | Point at a non-OpenAI provider (e.g. Gemini) |
| `--model-filter <s>` | Only generator models whose key contains `<s>` |
| `--skip-existing` | Skip cells already covered by a prior full run for this judge |

Scores are written under `data/results/human_reviews/` and
`data/results/rewritten/` so a subsequent `python pipeline/run_analysis.py`
picks them up automatically.

### `scholarpeer` — comparative AI-vs-human scoring

```bash
python pipeline/run_llm_judge.py scholarpeer --full --ttype rewritten --with-paper
```

Useful flags: `--ttype`, `--year`, `--source-model`, `--n` (pilot), `--full`,
`--with-paper`, `--resume`, `--pair-workers`.

## File-by-file: what each script does

| File | What it contains | When you'd touch it |
|---|---|---|
| `review_metrics.py` | `LLMJudge` class with one method per metric family, plus the `ReviewEvaluationPipeline` orchestrator | To change how a metric is computed |
| `prompts.py` | Verbatim prompts from the 4 source papers — one constant per prompt with the source URL/file in the section header | To swap in a different prompt for a metric |
| `run_llm_judge.py` | CLI + JSONL loaders + per-condition output writers | To add a new review condition or a new judge provider |
| `metric_registry.py` | Single source of truth for the 29 retained metrics: keys, display names, sources, scale labels, exclusion set, Bonferroni α | To add or exclude a metric |
| `collect_pairs.py` | Reads `data/results/`, flattens nested score dicts, returns a tidy `metric_key, diff` DataFrame | To change how paired diffs are aggregated |
| `run_analysis.py` | Wilcoxon + TOST per metric → CSV. No LaTeX, no plots — keeps it focused. | To change the statistical tests |
| `convert_results.py` | One-off helper used when this archive was first built (legacy → current JSON layout) | Normally not needed |

### Key entry points (Python API)

```python
from review_metrics import LLMJudge, ReviewEvaluationPipeline, Paper, Review, EvaluationInput

judge = LLMJudge(model="gpt-5-mini")
pipeline = ReviewEvaluationPipeline(llm_judge=judge)

result = pipeline.evaluate_single(EvaluationInput(
    paper=Paper(title="...", full_text="..."),
    ai_reviews=[Review(text="<rewritten review>")],
    human_reviews=[Review(text="<human review 1>"), Review(text="<human review 2>")],
))
# result["llm_judge"] -> {"ReviewEval": {...}, "RottenReviews": [...], "REMOR": {...}}

# ScholarPeer is a separate call (needs paired humans):
sp = judge.scholarpeer(
    paper=Paper(title="...", full_text="..."),
    ai_review="<rewritten>",
    human_reviews=["<h1>", "<h2>"],
)
# sp -> {"technical_accuracy", "constructive_value", "analytical_depth",
#        "novelty_and_significance", "overall"}
```

## Data shapes (quick reference)

| File pattern | Contains |
|---|---|
| `data/reviews/human_reviews/{venue}{year}.jsonl` | One paper per line, `reviews` is a list of human reviews with `full_review_text` |
| `data/reviews/rewritten/{venue}{year}_{provider}_{model}.jsonl` | Same shape, plus rewrite metadata (`original_text`, `rewrite_model`, …) |
| `data/results/human_reviews/{venue}{year}_{judge}.json` | Baseline scores `S_m(R)` — one nested `llm_judge_sample` block |
| `data/results/rewritten/{venue}{year}/{judge}.json` | Rewrite scores `S_m(R′)` per generator model + a `ScholarPeer.results[]` array |

See `data/README.md` for the full key-by-key schema.

## Reference: the 29 metrics

Pulled from four prior works. Three style/presentation metrics (Presentation
& Reporting from REMOR; Usage of Technical Terms and Clarity & Readability
from RottenReviews) are excluded a priori — defined in `EXCLUDED_KEYS`.

| Source | Analyzed metrics |
|---|---|
| **ReviewEval** (9) | Literature Comparison, Methodological Scrutiny, Results Interpretation, Theoretical Contributions, Logical Gaps Identification, Overall Depth, Total Insights, Actionable Insights, Actionability Score |
| **REMOR** (7) | Criticism, Example, Importance & Relevance, Materials & Methods, Praise, Results & Discussion, Suggestion & Solution |
| **RottenReviews** (8) | Comprehensiveness, Objectivity, Fairness, Actionability, Constructiveness, Relevance Alignment, Overall Quality, Overall Score |
| **ScholarPeer** (5) | Technical Accuracy, Constructive Value, Analytical Depth, Novelty & Significance, Overall |

Venues covered: ICLR and NeurIPS, 2021–2024 (8 venue-years). Rewrite models:
6 generators (Gemini-2.5-Flash, GPT-5, o4-mini, Claude-Haiku-4.5,
DeepSeek-R1, Llama-4-Scout). Judge models: `gpt5mini` and `gemini`.

## Human validation of rewrites

The analysis assumes each rewrite **R′** preserves the evaluative content of
the original review **R**. Three computer-science graduate students annotated
50 original–rewrite pairs (two raters per pair, 100 annotations). The
[`human_annotation/Guidelines.pdf`](human_annotation/Guidelines.pdf) gives
the full instructions; questions are:

| # | Question | Scale |
|---|---|---|
| Q1 | Content Faithfulness | 1–5 Likert |
| Q2 | Overall Equivalence | Yes / No |

Annotators were told to focus on **evaluative content**, not wording or
style. CSV files (one per annotator) carry: `pair_id, generator, venue,
year, decision, q1_faithfulness, q2_equivalence, notes, updated_at`.

## License

Released for research use. Review text in `data/` comes from OpenReview;
redistribution should follow OpenReview's terms.
