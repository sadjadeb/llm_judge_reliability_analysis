# Judging a Review by its Cover

Code and precomputed results for the paper
*Judging a Review by its Cover: A Reliability Analysis of LLM-based Peer
Review Evaluation Metrics*.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

API keys are only needed if you re-score reviews (Path 2). The bundled
scores in `data/results/` are enough to reproduce Table 1.

## Path 1 — Reproduce Table 1 (no API needed)

```bash
python pipeline/run_analysis.py           # GPT-5-mini  → results/results_gpt5mini.csv
python pipeline/run_analysis.py --gemini  # Gemini-2.5-Flash → results/results_gemini.csv
```

Each CSV has 29 rows (one per metric) and these columns:

| Column | Meaning |
|---|---|
| `metric`, `source` | Metric name and its source paper (ReviewEval / REMOR / RottenReviews / ScholarPeer) |
| `n_obs` | Number of paired observations |
| `mean_delta`, `sd_delta` | Mean and SD of the paired differences `d = S_m(R′) − S_m(R)` |
| `delta_bound` | TOST equivalence bound `δ = 0.2 × sd_delta` |
| `wilcoxon_p` | Wilcoxon signed-rank p-value (surface-sensitivity test) |
| `tost_p` | TOST p-value (robustness test) |
| `sensitive`, `robust` | Booleans: `p < 0.05/29 ≈ 0.00172` |

## Path 2 — Re-score reviews with a different judge

```bash
export OPENAI_API_KEY=...    # or GEMINI_API_KEY for Gemini

# Score human + rewritten with the point-wise families (ReviewEval/REMOR/RottenReviews)
python pipeline/run_llm_judge.py pointwise --only-human --sample 0
python pipeline/run_llm_judge.py pointwise --only-rewritten --sample 0

# Score with ScholarPeer (AI vs paired human)
python pipeline/run_llm_judge.py scholarpeer --full --ttype rewritten --with-paper
```

New scores are written into `data/results/`; afterwards rerun Path 1 to
get the updated CSVs.

Common flags: `--judge-model`, `--judge-base-url`, `--judge-api-key-env`,
`--model-filter`, `--sample`, `--skip-existing`.

## Repo layout

```
.
├── pipeline/
│   ├── run_analysis.py        # Wilcoxon + TOST → results/*.csv
│   ├── run_llm_judge.py       # CLI to (re-)score reviews
│   ├── collect_pairs.py       # builds paired diffs from data/results/
│   ├── metric_registry.py     # 29-metric inventory + Bonferroni α
│   ├── review_metrics.py      # LLMJudge class
│   └── prompts.py             # verbatim prompts from the 4 source papers
├── data/
│   ├── reviews/               # input review text (JSONL)
│   └── results/               # cached judge scores (JSON)
├── human_annotation/          # rewrite-faithfulness annotation study (CSVs + Guidelines.pdf)
└── results/                   # Wilcoxon + TOST outputs (CSV)
```

See `data/README.md` for the full JSON/JSONL key schema.

## Human annotation

`human_annotation/` contains the 3 annotator CSVs (50 + 25 + 25 pairs) and
the `Guidelines.pdf` shown to annotators for the rewrite-faithfulness
validation study described in the paper.

## License

Released for research use. Review text in `data/` comes from OpenReview;
redistribution should follow OpenReview's terms.
