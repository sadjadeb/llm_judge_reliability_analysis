# Judging a Review by its Cover

Code and precomputed results for the paper
*Judging a Review by its Cover: A Reliability Analysis of LLM-based Peer Review
Evaluation Metrics*.

This README is a **how-to-use-the-code** guide. For the motivation, full
analytical framework, and discussion of findings, see the paper itself.

## What you can do with this repo

| Goal | Command | Time |
|---|---|---|
| Reproduce Table 1 (GPT-5-mini) | `python pipeline/run_analysis.py` | ~3 s |
| Reproduce the Gemini replication | `python pipeline/run_analysis.py --gemini` | ~3 s |
| Re-score reviews with a different judge | `python pipeline/run_llm_judge.py ...` | hours, API-bound |
| Inspect a per-metric paired-diff CSV | open `results/results_{gpt5mini,gemini}.csv` | — |
| Inspect raw judge scores per review | open any `data/results/{condition}/{year}/{judge}.json` | — |

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

Each run:

1. Loads cached scores from `data/results/human_reviews/` and
   `data/results/rewritten/` via `collect_pairs.collect_observations(judge=…)`.
2. Computes paired differences (`S_m(R′) − S_m(R)` for point-wise metrics;
   `score − 5.0` for ScholarPeer's human-anchored 1–10 scale).
3. Drops the 3 a-priori-excluded style/presentation metrics (`EXCLUDED_KEYS`
   in `pipeline/metric_registry.py`).
4. For each of the 29 retained metrics runs Wilcoxon (sensitivity) and TOST
   (robustness), applies Bonferroni at `α = 0.05/29 ≈ 0.00172`, and writes
   the CSV.

Each CSV has one row per metric with 11 columns:

```
metric_key, metric, source, n_obs, mean_delta, sd_delta, delta_bound,
wilcoxon_p, tost_p, sensitive, robust
```

`sensitive` and `robust` are booleans; the four-region classification used in
the paper's Figure 1 is the joint of these two flags:

| Wilcoxon | TOST | Region |
|:---:|:---:|---|
| ✓ | — | Sensitive only |
| — | ✓ | Robust only |
| ✓ | ✓ | Sensitive ∩ Robust |
| — | — | Inconclusive |

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
