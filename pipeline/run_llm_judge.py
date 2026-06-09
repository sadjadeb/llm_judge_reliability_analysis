#!/usr/bin/env python3
"""
run_llm_judge.py — LLM-as-judge runner for surface-sensitivity analysis.

Scores original human reviews and meaning-preserving rewrites with 32 metrics
from ReviewEval, REMOR, RottenReviews, and ScholarPeer (29 used in analysis).

Subcommands:
    pointwise    ReviewEval + RottenReviews + REMOR per review
    scholarpeer  ScholarPeer comparative scoring (1–10, 5 = human baseline)

Usage:
    export OPENAI_API_KEY="sk-..."

    # Point-wise (ReviewEval / RottenReviews / REMOR):
    python run_llm_judge.py pointwise --sample 3
    python run_llm_judge.py pointwise --sample 0 --model-filter google_gemini
    python run_llm_judge.py pointwise --years ICLR2022 --only-transformations \\
        --transformation-types hybrid

    # ScholarPeer (AI-vs-human comparative scoring):
    python run_llm_judge.py scholarpeer --n 3 --ttype rewritten --with-paper
    python run_llm_judge.py scholarpeer --full --ttype expanded --with-paper --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from review_metrics import (
    EvaluationInput, LLMJudge, Paper, Review, ReviewEvaluationPipeline,
)

load_dotenv()

DEFAULT_YEARS = ["ICLR2021", "ICLR2022", "ICLR2023"]
TTYPES = ["rewritten"]
REVIEW_ROOT = Path("data/reviews")
LLM_METRICS = ["llm_judge"]
JUDGE_MODEL = "gpt-5-mini"  # default; overridden by --judge-model at runtime
_ACTIVE_JUDGE_MODEL = JUDGE_MODEL  # updated in _run_pointwise()
_MODEL_FILTER: str | None = None   # updated in _run_pointwise()
MANUSCRIPT_DIR = Path("data/manuscript_markdowns")


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Shared data loaders / EvaluationInput builders
#
#  Only the review text is needed by the Table 1 metrics, so the builders
#  populate ``Review.text`` and ``Paper.title`` only.
# ════════════════════════════════════════════════════════════════════════


def load_human_papers(path: str) -> dict[str, dict]:
    """Load a human-reviews JSONL file keyed by ``forum_id``."""
    papers: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            papers[d["forum_id"]] = d
    return papers


def build_human_review(r: dict) -> Review:
    """Wrap a raw human-review record as a ``Review`` (text only)."""
    return Review(text=r.get("full_review_text") or r.get("text", ""))


def build_ai_only_review(r: dict, venue_year: str) -> Review:
    """Extract the AI review text across all venue/year field-name conventions.

    Field-name fallback chain:
        ICLR2021-              -> "review"
        ICLR2022 / NeurIPS2021 -> "main_review"
        ICLR2023+              -> "strength_and_weaknesses" + "clarity_quality..."
        NeurIPS2022-24         -> "strengths_and_weaknesses" + "limitations" + "questions"
    """
    text = (
        r.get("review")
        or r.get("main_review")
        or "\n\n".join(filter(None, [
            r.get("strength_and_weaknesses", ""),
            r.get("strengths_and_weaknesses", ""),
            r.get("limitations_and_societal_impact", ""),
            r.get("limitations", ""),
            r.get("questions", ""),
            r.get("clarity_quality_novelty_and_reproducibility", ""),
        ]))
        or ""
    )
    return Review(text=text)


def build_transformation_review(r: dict, ttype: str) -> Review | None:
    """Extract the transformed review text; returns None if no content available."""
    text = r.get("text", "").strip() or r.get("full_review_text", "").strip()
    return Review(text=text) if text else None


def build_inputs(
    human_papers: dict[str, dict],
    ai_only_records: list[dict],
    venue_year: str,
) -> list[EvaluationInput]:
    """Pair AI-Only reviews with human reviews on the same paper."""
    inputs: list[EvaluationInput] = []
    for rec in ai_only_records:
        human = human_papers.get(rec["forum_id"])
        if human is None:
            continue
        paper = Paper(title=human.get("title", ""))
        human_reviews = [build_human_review(r) for r in human.get("reviews", [])]
        ai_reviews = [build_ai_only_review(r, venue_year) for r in rec.get("reviews", [])]
        if not human_reviews or not ai_reviews:
            continue
        inputs.append(
            EvaluationInput(paper=paper, ai_reviews=ai_reviews, human_reviews=human_reviews)
        )
    return inputs


def build_transformation_inputs(
    human_papers: dict[str, dict],
    ttype_records: list[dict],
    ttype: str,
) -> list[EvaluationInput]:
    """Pair transformation (AI) reviews with human reviews on the same paper."""
    inputs: list[EvaluationInput] = []
    for rec in ttype_records:
        human = human_papers.get(rec["forum_id"])
        title = rec.get("title", "") or (human.get("title", "") if human else "")
        paper = Paper(title=title)
        human_reviews = [build_human_review(r) for r in (human or {}).get("reviews", [])]
        ai_reviews = [
            rev for r in rec.get("reviews", [])
            if (rev := build_transformation_review(r, ttype)) is not None
        ]
        if not human_reviews or not ai_reviews:
            continue
        inputs.append(
            EvaluationInput(paper=paper, ai_reviews=ai_reviews, human_reviews=human_reviews)
        )
    return inputs


# ════════════════════════════════════════════════════════════════════════
#  Point-wise mode (ReviewEval + RottenReviews + REMOR)
# ════════════════════════════════════════════════════════════════════════


def _should_skip_existing_llm(all_results: dict, model_key: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    entry = all_results.get(model_key) or {}
    sample = entry.get("llm_judge_sample")
    if not isinstance(sample, dict):
        return False
    if sample.get("judge_model") != _ACTIVE_JUDGE_MODEL:
        return False
    averaged = sample.get("averaged") or {}
    if not averaged:
        return False
    ss, tp = sample.get("sample_size"), sample.get("total_papers")
    if not isinstance(ss, int) or not isinstance(tp, int) or tp <= 0:
        return False
    if ss < tp:
        return False
    return True


def _avg_dicts(dicts: list[dict]) -> dict:
    """Average numeric values; for string values store mode + frequency distribution."""
    result: dict = {}
    all_keys = {k for d in dicts for k in d}
    for k in all_keys:
        num_vals = [d[k] for d in dicts if k in d and isinstance(d[k], (int, float))
                    and not (isinstance(d[k], float) and (math.isnan(d[k]) or math.isinf(d[k])))]
        if num_vals:
            result[k] = sum(num_vals) / len(num_vals)
            continue
        # Categorical: compute mode and frequency distribution
        str_vals = [str(d[k]).lower() for d in dicts
                    if k in d and d[k] is not None and isinstance(d[k], str)]
        if str_vals:
            from collections import Counter
            counts = Counter(str_vals)
            mode = counts.most_common(1)[0][0]
            n = len(str_vals)
            result[k] = {
                "mode": mode,
                "distribution": {v: round(c / n, 4) for v, c in counts.most_common()},
                "n": n,
            }
    return result


def _flatten_llm_result(llm_dict: dict) -> dict:
    """Flatten nested LLM judge results into a single-level dict (numeric + categorical)."""
    flat: dict = {}
    for key, val in llm_dict.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, (int, float)):
                    flat[f"{key}__{sub_key}"] = sub_val
                elif isinstance(sub_val, str):
                    flat[f"{key}__{sub_key}"] = sub_val   # preserve categorical strings
        elif isinstance(val, (int, float)):
            flat[key] = val
        elif isinstance(val, str):
            flat[key] = val                               # preserve top-level categoricals
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            merged = _avg_dicts(val)
            for sub_key, sub_val in merged.items():
                flat[f"{key}__{sub_key}"] = sub_val       # includes both numeric and categorical
    return flat


def _merge_llm_into_per_paper(model_entry: dict, llm_result: dict) -> None:
    """Attach nested ``llm_judge`` to each ``per_paper`` row (same index as ``build_inputs``)."""
    rows = model_entry.get("per_paper")
    if not isinstance(rows, list):
        return
    for idx, llm in zip(llm_result.get("evaluated_indices", []), llm_result.get("per_paper_llm", [])):
        if isinstance(idx, int) and 0 <= idx < len(rows) and isinstance(rows[idx], dict):
            rows[idx]["llm_judge"] = llm


def _llm_sample_for_json(llm_result: dict) -> dict:
    """Drop bulky nested duplicate; nested scores live under ``per_paper``."""
    return {k: v for k, v in llm_result.items() if k != "per_paper_llm"}


def run_llm_on_subset(
    inputs: list[EvaluationInput],
    pipeline: ReviewEvaluationPipeline,
    sample_n: int,
    label: str,
    paper_workers: int = 4,
) -> dict | None:
    """Evaluate LLM-judge metrics on a random subset, return aggregated results.

    ``sample_n <= 0`` means use **all** papers (original order, no subsampling).
    ``paper_workers`` controls how many papers are evaluated concurrently.
    Combined with ``max_parallel`` (LLM call concurrency within each paper),
    total concurrent API requests = paper_workers × max_parallel.

    Returns ``evaluated_indices`` (indices into ``inputs`` / ``per_paper``) and
    ``per_paper_llm`` (nested ``llm_judge`` dicts) so callers can merge into
    ``results.json`` ``per_paper`` like other metric groups.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    if not inputs:
        _log(f"  [{label}] no inputs — skipping")
        return None

    if sample_n <= 0:
        idx_list = list(range(len(inputs)))
        _log(f"  [{label}] evaluating all {len(idx_list)}/{len(inputs)} papers "
             f"(paper_workers={paper_workers})...")
    else:
        k = min(sample_n, len(inputs))
        idx_list = random.sample(range(len(inputs)), k)
        _log(f"  [{label}] evaluating {k}/{len(inputs)} papers "
             f"(paper_workers={paper_workers})...")

    results_by_idx: dict[int, tuple[dict, dict]] = {}  # idx → (flat, llm)
    completed_count = 0
    lock = threading.Lock()

    def _eval_one(ord_i: int, idx: int):
        inp = inputs[idx]
        try:
            result = pipeline.evaluate_single(inp, metrics=LLM_METRICS)
            llm  = result.get("llm_judge", {})
            flat = _flatten_llm_result(llm)
            with lock:
                nonlocal completed_count
                completed_count += 1
                results_by_idx[idx] = (flat, llm)
                _log(f"  [{label}] paper {completed_count}/{len(idx_list)} done "
                     f"({(inp.paper.title or 'untitled')[:45]!r}) — {len(flat)} metrics")
            return True
        except Exception as exc:
            _log(f"  [{label}] paper {ord_i+1} FAILED: {exc}")
            return False

    pw = max(1, min(paper_workers, len(idx_list)))
    with ThreadPoolExecutor(max_workers=pw) as ex:
        futures = {ex.submit(_eval_one, oi, idx): idx
                   for oi, idx in enumerate(idx_list)}
        for f in as_completed(futures):
            f.result()  # surface exceptions if any

    if not results_by_idx:
        return None

    # Preserve original order
    ordered_idx  = [i for i in idx_list if i in results_by_idx]
    per_paper_flat = [results_by_idx[i][0] for i in ordered_idx]
    per_paper_llm  = [results_by_idx[i][1] for i in ordered_idx]

    averaged = _avg_dicts(per_paper_flat)
    return {
        "judge_model": _ACTIVE_JUDGE_MODEL,
        "sample_size": len(per_paper_flat),
        "total_papers": len(inputs),
        "per_paper": per_paper_flat,
        "per_paper_llm": per_paper_llm,
        "evaluated_indices": ordered_idx,
        "averaged": averaged,
    }


def process_ai_only(
    pipeline,
    sample_n,
    years: list[str],
    *,
    skip_existing: bool = False,
    paper_workers: int = 4,
):
    """Run LLM judge on AI-Only reviews."""
    for year in years:
        human_file = REVIEW_ROOT / "human_reviews" / f"{year}.jsonl"
        if not human_file.exists():
            continue
        human_papers = load_human_papers(str(human_file))

        result_dir = Path(f"data/results/AI-Only/{year}")
        results_json = result_dir / "results.json"
        if not results_json.exists():
            _log(f"SKIP {year} AI-Only — no results.json")
            continue

        all_results = json.loads(results_json.read_text())
        updated = False

        for src_file in sorted(Path("data/ai_only_reviews").glob(f"{year}_*.jsonl")):
            model_key = src_file.stem
            if _MODEL_FILTER and not re.search(_MODEL_FILTER, model_key):
                continue
            if _should_skip_existing_llm(all_results, model_key, skip_existing):
                _log(f"  SKIP {model_key} (existing {_ACTIVE_JUDGE_MODEL} llm_judge_sample)")
                continue
            records = [json.loads(l) for l in open(src_file)]
            inputs = build_inputs(human_papers, records, year)

            label = f"AI-Only/{year}/{model_key}"
            llm_result = run_llm_on_subset(inputs, pipeline, sample_n, label, paper_workers=paper_workers)

            if llm_result and model_key in all_results:
                all_results[model_key].setdefault("corpus", {}).setdefault("averaged_per_paper", {})
                all_results[model_key]["corpus"]["averaged_per_paper"]["llm_judge"] = llm_result["averaged"]
                _merge_llm_into_per_paper(all_results[model_key], llm_result)
                all_results[model_key]["llm_judge_sample"] = _llm_sample_for_json(llm_result)
                updated = True
                _log(f"  [{label}] merged into results (corpus + per_paper)")

        if updated:
            with open(results_json, "w") as f:
                json.dump(all_results, f, indent=2)
            _log(f"  saved {results_json}\n")


def process_transformations(
    pipeline,
    sample_n,
    ttypes: list[str],
    years: list[str],
    *,
    skip_existing: bool = False,
    paper_workers: int = 4,
):
    """Run LLM judge on transformation reviews."""
    for ttype in ttypes:
        for year in years:
            human_file = REVIEW_ROOT / "human_reviews" / f"{year}.jsonl"
            if not human_file.exists():
                continue
            human_papers = load_human_papers(str(human_file))

            result_dir = Path(f"data/results/rewritten/{year}")
            judge_tag = "gemini" if "gemini" in _ACTIVE_JUDGE_MODEL.lower() else "gpt5mini"
            results_json = result_dir / f"{judge_tag}.json"
            all_results = json.loads(results_json.read_text()) if results_json.exists() else {}
            updated = False

            for src_file in sorted((REVIEW_ROOT / "transformations" / ttype).glob(f"{year}_*.jsonl")):
                model_key = src_file.stem
                if _MODEL_FILTER and not re.search(_MODEL_FILTER, model_key):
                    continue
                if _should_skip_existing_llm(all_results, model_key, skip_existing):
                    _log(f"  SKIP {ttype}/{year}/{model_key} (existing {_ACTIVE_JUDGE_MODEL} llm_judge_sample)")
                    continue
                records = [json.loads(l) for l in open(src_file)]
                inputs = build_transformation_inputs(human_papers, records, ttype)

                label = f"{ttype}/{year}/{model_key}"
                llm_result = run_llm_on_subset(inputs, pipeline, sample_n, label, paper_workers=paper_workers)

                if llm_result:
                    block = all_results.setdefault(model_key, {"per_paper": [{}] * len(inputs)})
                    if "per_paper" not in block or len(block["per_paper"]) < len(inputs):
                        block["per_paper"] = [{"llm_judge": {}} for _ in inputs]
                    _merge_llm_into_per_paper(block, llm_result)
                    updated = True
                    _log(f"  [{label}] merged into {results_json.name}")

            if updated:
                result_dir.mkdir(parents=True, exist_ok=True)
                results_json.write_text(json.dumps(all_results, indent=2))
                _log(f"  saved {results_json}\n")


def process_human(
    pipeline,
    sample_n,
    years: list[str],
    *,
    skip_existing: bool = False,
    paper_workers: int = 4,
    output_suffix: str = "",
):
    """Run LLM judge on human reviews (treating them as 'AI reviews' so the judge scores them).

    output_suffix: appended to the filename stem (e.g. "_gemini" -> {year}_gemini.json).
    When non-empty, the canonical {year}.json is neither read nor written.
    """
    for year in years:
        human_file = REVIEW_ROOT / "human_reviews" / f"{year}.jsonl"
        if not human_file.exists():
            continue
        human_papers = load_human_papers(str(human_file))

        inputs: list[EvaluationInput] = []
        for fid, paper_data in human_papers.items():
            h_reviews = [build_human_review(r) for r in paper_data.get("reviews", [])]
            if not h_reviews:
                continue
            paper = Paper(title=paper_data.get("title", ""))
            inputs.append(EvaluationInput(
                paper=paper,
                ai_reviews=h_reviews,
                human_reviews=h_reviews,
            ))

        label = f"human/{year}"
        result_dir = Path("data/results/human_reviews")
        suffix = output_suffix or ("_gemini" if "gemini" in _ACTIVE_JUDGE_MODEL.lower() else "_gpt5mini")
        results_json = result_dir / f"{year}{suffix}.json"
        existing = json.loads(results_json.read_text()) if results_json.exists() else {}
        human_block = existing.get("human") or {}
        if skip_existing and isinstance(human_block, dict):
            sample = human_block.get("llm_judge_sample")
            if isinstance(sample, dict) and sample.get("judge_model") == _ACTIVE_JUDGE_MODEL:
                averaged = sample.get("averaged") or {}
                ss, tp = sample.get("sample_size"), sample.get("total_papers")
                full = (
                    averaged
                    and isinstance(ss, int)
                    and isinstance(tp, int)
                    and tp > 0
                    and ss >= tp
                )
                if full:
                    _log(f"  SKIP human/{year} (existing {JUDGE_MODEL} full llm_judge_sample)")
                    continue

        llm_result = run_llm_on_subset(inputs, pipeline, sample_n, label, paper_workers=paper_workers)

        if llm_result:
            result_dir.mkdir(parents=True, exist_ok=True)
            existing["human"] = {
                "corpus": {
                    "averaged_per_paper": {
                        "llm_judge": llm_result["averaged"],
                    }
                },
                "llm_judge_sample": _llm_sample_for_json(llm_result),
            }
            with open(results_json, "w") as f:
                json.dump(existing, f, indent=2)
            _log(f"  saved {results_json}\n")


def _add_pointwise_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Papers per model (default 5). Use 0 to evaluate every matched paper.",
    )
    parser.add_argument("--only-rewritten", action="store_true",
                        help="Only evaluate rewritten reviews")
    parser.add_argument("--only-human", action="store_true", help="Only evaluate human reviews")
    parser.add_argument(
        "--skip-human",
        action="store_true",
        help="Do not run human LLM-judge pass",
    )
    parser.add_argument(
        "--skip-rewritten",
        action="store_true",
        help="Do not run rewritten LLM-judge pass",
    )
    parser.add_argument(
        "--transformation-types",
        nargs="+",
        choices=TTYPES,
        default=TTYPES,
        metavar="TYPE",
        help="Which transformation folders to use (default: all). "
        "Use 'hybrid' for human + AI-Only mixes.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=f"Skip when llm_judge_sample for {JUDGE_MODEL} already covers all papers "
        f"(sample_size >= total_papers). Partial runs (e.g. 5 of 20) are not skipped.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model-filter", type=str, default=None,
                        help="Only run on models whose key contains this substring (e.g. 'gemini')")
    parser.add_argument(
        "--years",
        nargs="+",
        default=None,
        metavar="YEAR",
        help=f"Which venue years to run (default: {DEFAULT_YEARS}). Example: --years ICLR2021",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=16,
        metavar="N",
        help="Max concurrent LLM HTTP requests per paper (default 16).",
    )
    parser.add_argument(
        "--paper-workers",
        type=int,
        default=4,
        metavar="N",
        help="Papers evaluated concurrently (default 4). "
             "Total concurrent API calls = paper-workers × max-parallel.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override the LLM judge model (default: gpt-5-mini). "
             "E.g. 'gemini-2.5-flash' for Gemini.",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="Custom OpenAI-compatible base URL for the judge. "
             "E.g. 'https://generativelanguage.googleapis.com/v1beta/openai/' for Gemini.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for the judge API key (default: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--human-output-suffix",
        type=str,
        default="",
        help="Suffix appended to the Human LLM Judge filename stem "
             "(e.g. '_gemini' -> {year}_gemini.json). "
             "When set, the canonical {year}.json is neither read nor written.",
    )


def _run_pointwise(args: argparse.Namespace) -> None:
    judge_model   = args.judge_model or JUDGE_MODEL
    global _ACTIVE_JUDGE_MODEL; _ACTIVE_JUDGE_MODEL = judge_model
    judge_api_key = os.getenv(args.judge_api_key_env)
    if not judge_api_key:
        sys.exit(f"ERROR: {args.judge_api_key_env} not set.")

    random.seed(args.seed)

    global _MODEL_FILTER
    _MODEL_FILTER = args.model_filter

    active_years = args.years if args.years else DEFAULT_YEARS
    _log(
        f"LLM Judge: model={judge_model}, sample={args.sample}/model, seed={args.seed}, "
        f"max_parallel={args.max_parallel}, paper_workers={args.paper_workers}, years={active_years}"
        + (f", base_url={args.judge_base_url}" if args.judge_base_url else "")
        + (f", filter={_MODEL_FILTER}" if _MODEL_FILTER else ""),
    )

    judge = LLMJudge(model=judge_model, temperature=1.0,
                     max_parallel=args.max_parallel,
                     base_url=args.judge_base_url,
                     api_key=judge_api_key)
    pipeline = ReviewEvaluationPipeline(llm_judge=judge)

    if args.only_human:
        _log("=== Human Reviews ===")
        process_human(pipeline, args.sample, active_years, skip_existing=args.skip_existing, paper_workers=args.paper_workers, output_suffix=args.human_output_suffix)
    elif args.only_rewritten:
        _log("=== Rewritten ===")
        process_transformations(
            pipeline, args.sample, args.transformation_types, active_years,
            skip_existing=args.skip_existing, paper_workers=args.paper_workers,
        )
    else:
        if not args.skip_human:
            _log("=== Human Reviews ===")
            process_human(pipeline, args.sample, active_years, skip_existing=args.skip_existing, paper_workers=args.paper_workers, output_suffix=args.human_output_suffix)
        if not args.skip_rewritten:
            _log("=== Rewritten ===")
            process_transformations(
                pipeline, args.sample, args.transformation_types, active_years,
                skip_existing=args.skip_existing, paper_workers=args.paper_workers,
            )

    _log("All LLM-judge evaluations complete.")


# ════════════════════════════════════════════════════════════════════════
#  ScholarPeer mode (AI review vs paired human reviews)
# ════════════════════════════════════════════════════════════════════════


def _human_file(year: str) -> Path:
    return REVIEW_ROOT / "human_reviews" / f"{year}.jsonl"


def _transformation_file(ttype: str, year: str, model_key: str) -> Path:
    return REVIEW_ROOT / "transformations" / ttype / f"{year}_{model_key}.jsonl"


def _scholarpeer_results_file(ttype: str, year: str, model_key: str, full: bool, with_paper: bool,
                              judge_model: str = "gpt-5-mini") -> str:
    model_short = model_key.replace("/", "_").replace("openrouter_", "").replace("openai_", "").replace("google_", "")
    suffix = "with_paper" if with_paper else "title_only"
    scope  = "full" if full else "pilot"
    # Include judge model name when not the default to avoid overwriting
    judge_tag = f"_judge-{judge_model.replace('/','-')}" if "gemini" in judge_model or "claude" in judge_model else ""
    base  = "data/results/scholarpeer_neurips" if "NeurIPS" in year else "data/results/scholarpeer"
    return f"{base}/{scope}_{ttype}_{year}_{model_short}_{suffix}{judge_tag}.json"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _load_manuscript(forum_id: str) -> str:
    """Load full manuscript markdown text for a paper, empty string if not found."""
    p = MANUSCRIPT_DIR / f"{forum_id}.md"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def load_ai_only_pairs(ttype_file: Path, year: str) -> list[dict]:
    """
    Load pairs for AI-Only reviews.
    A = AI-generated review  (field: 'review')
    B = one human review for same paper (paired by index mod n_human)
    all_human_texts = all human reviews (used by ScholarPeer H-Max)
    """
    human_by_forum: dict = {}
    for line in open(_human_file(year)):
        p = json.loads(line)
        human_by_forum[p["forum_id"]] = p

    pairs = []
    for line in open(ttype_file):
        rec  = json.loads(line)
        fid  = rec["forum_id"]
        human_paper       = human_by_forum.get(fid, {})
        title             = human_paper.get("title", "")
        human_reviews_raw = human_paper.get("reviews", [])
        all_human_texts   = [
            r.get("full_review_text", r.get("text", "")).strip()
            for r in human_reviews_raw
            if r.get("full_review_text", r.get("text", "")).strip()
        ]
        manuscript_text = _load_manuscript(fid)

        ai_reviews_raw = rec.get("reviews", [])
        n_human        = max(len(all_human_texts), 1)

        for si, ai_rev in enumerate(ai_reviews_raw):
            # Field name varies by venue/year (same chain as build_ai_only_review)
            ai_text = (
                ai_rev.get("review") or
                ai_rev.get("main_review") or
                "\n\n".join(filter(None, [
                    ai_rev.get("strength_and_weaknesses", ""),
                    ai_rev.get("strengths_and_weaknesses", ""),
                    ai_rev.get("limitations_and_societal_impact", ""),
                    ai_rev.get("limitations", ""),
                    ai_rev.get("questions", ""),
                    ai_rev.get("clarity_quality_novelty_and_reproducibility", ""),
                    ai_rev.get("summary_of_the_review", ""),
                ])) or ""
            ).strip()
            if not ai_text:
                continue
            # Pair with human review by index mod n_human
            human_b = all_human_texts[si % n_human] if all_human_texts else ""
            if not human_b:
                continue
            pairs.append({
                "forum_id":         fid,
                "title":            title,
                "review_id":        f"ai_only_{si}",
                "transformed_text": ai_text,     # A = AI-Only
                "original_text":    human_b,     # B = human (for ScholarPeer comparison)
                "all_human_texts":  all_human_texts,
                "manuscript_text":  manuscript_text,
            })
    return pairs


def load_all_pairs(ttype_file: Path, year: str = "ICLR2021") -> list[dict]:
    """
    Returns ALL valid (transformed, original) pairs across all papers.
    Each dict: {forum_id, title, review_id, transformed_text, original_text,
                all_human_texts, manuscript_text}
    """
    human_by_forum: dict = {}
    for line in open(_human_file(year)):
        p = json.loads(line)
        human_by_forum[p["forum_id"]] = p

    pairs = []
    for line in open(ttype_file):
        rec   = json.loads(line)
        fid   = rec["forum_id"]
        title = rec.get("title", "")

        human_paper       = human_by_forum.get(fid, {})
        human_reviews_raw = human_paper.get("reviews", [])
        all_human_texts   = [
            r.get("full_review_text", r.get("text", "")).strip()
            for r in human_reviews_raw
            if r.get("full_review_text", r.get("text", "")).strip()
        ]
        manuscript_text = _load_manuscript(fid)

        for rev in rec.get("reviews", []):
            rew_text  = rev.get("text", "").strip()
            orig_text = rev.get("original_text", "").strip()
            # Use full_review_text only if text is empty AND it differs from original
            # (if frt == original, the AI review failed — skip this pair)
            if not rew_text:
                frt = rev.get("full_review_text", "").strip()
                if frt and orig_text and frt[:80] != orig_text[:80]:
                    rew_text = frt
            if rew_text and orig_text:
                pairs.append({
                    "forum_id":        fid,
                    "title":           title,
                    "review_id":       rev.get("review_id", rev.get("id", "")),
                    "transformed_text": rew_text,
                    "original_text":   orig_text,
                    "all_human_texts": all_human_texts,
                    "manuscript_text": manuscript_text,
                })
    return pairs


def load_paired(n: int, ttype_file: Path, year: str = "ICLR2021") -> list[dict]:
    """Return first n pairs (pilot mode, one per paper)."""
    human_by_forum: dict = {}
    for line in open(_human_file(year)):
        p = json.loads(line)
        human_by_forum[p["forum_id"]] = p

    pairs = []
    seen_forums: set = set()
    for line in open(ttype_file):
        rec = json.loads(line)
        fid = rec["forum_id"]
        if fid in seen_forums:
            continue
        title = rec.get("title", "")
        human_paper     = human_by_forum.get(fid, {})
        all_human_texts = [
            r.get("full_review_text", r.get("text", "")).strip()
            for r in human_paper.get("reviews", [])
            if r.get("full_review_text", r.get("text", "")).strip()
        ]
        manuscript_text = _load_manuscript(fid)
        for rev in rec.get("reviews", []):
            rew_text  = rev.get("text", "").strip()
            orig_text = rev.get("original_text", "").strip()
            if rew_text and orig_text:
                pairs.append({"forum_id": fid, "title": title,
                               "review_id": rev.get("review_id", ""),
                               "transformed_text": rew_text,
                               "original_text":    orig_text,
                               "all_human_texts":  all_human_texts,
                               "manuscript_text":  manuscript_text})
                seen_forums.add(fid)
                break
        if len(pairs) >= n:
            break
    return pairs


def _add_scholarpeer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=3,
                        help="Number of pairs for pilot mode (one per paper)")
    parser.add_argument("--full", action="store_true",
                        help="Run all pairs across all papers/reviews")
    parser.add_argument("--ttype", default="rewritten",
                        choices=["rewritten"],
                        help="Review condition (rewritten only in this archive).")
    parser.add_argument("--year", default="ICLR2021",
                        choices=["ICLR2021", "ICLR2022", "ICLR2023", "ICLR2024",
                                 "NeurIPS2021", "NeurIPS2022", "NeurIPS2023", "NeurIPS2024"],
                        help="Venue year (default: ICLR2021)")
    parser.add_argument("--source-model", default="google_gemini-2.5-flash",
                        help="Model key whose transformation reviews to evaluate "
                             "(default: google_gemini-2.5-flash)")
    parser.add_argument("--with-paper", action="store_true",
                        help="Pass full manuscript markdown as paper_text (recommended)")
    parser.add_argument("--judge-model", default="gpt-5-mini", help="LLM judge model")
    parser.add_argument("--judge-base-url", default=None,
                        help="Custom OpenAI-compatible base URL (e.g. Gemini endpoint)")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY",
                        help="Env var name for the judge API key (default: OPENAI_API_KEY)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs already saved in the output file")
    parser.add_argument("--pair-workers", type=int, default=4,
                        help="Pairs evaluated concurrently (default 4). "
                             "Each worker runs the ScholarPeer scoring call for one pair. "
                             "Total concurrent API calls ≈ pair-workers.")


def _run_scholarpeer(args: argparse.Namespace) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    judge_api_key = os.getenv(args.judge_api_key_env)
    if not judge_api_key:
        raise SystemExit(f"ERROR: {args.judge_api_key_env} not set.")

    use_paper    = args.with_paper
    ttype        = args.ttype
    year         = args.year
    source_model = args.source_model
    ttype_file   = _transformation_file(ttype, year, source_model)

    if not ttype_file.exists():
        raise SystemExit(f"ERROR: transformation file not found: {ttype_file}")

    results_file = _scholarpeer_results_file(ttype, year, source_model, args.full, use_paper, judge_model=args.judge_model)
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

    if args.full:
        pairs      = load_all_pairs(ttype_file, year)
        mode_label = f"FULL — {ttype} vs Human"
    else:
        pairs      = load_paired(args.n, ttype_file, year)
        mode_label = f"PILOT ({args.n} pairs) — {ttype} vs Human"

    # Load existing results for resume
    done_ids: set[str] = set()
    existing_results: list[dict] = []
    if args.resume and Path(results_file).exists():
        prev = json.loads(Path(results_file).read_text())
        existing_results = prev.get("results", [])
        done_ids = {f"{r['forum_id']}_{r.get('review_id','')}" for r in existing_results}
        print(f"Resume mode: {len(done_ids)} pairs already done, skipping.")

    print(f"Mode        : {mode_label}")
    print(f"Judge model : {args.judge_model}")
    print(f"Total pairs : {len(pairs)}")
    print(f"Remaining   : {len(pairs) - len(done_ids)}")
    print("Comparison  : Transformed (A) vs Original Human (B)")
    print(f"Paper text  : {'full manuscript markdown' if use_paper else 'title only'}\n")

    judge   = LLMJudge(model=args.judge_model,
                       base_url=args.judge_base_url,
                       api_key=judge_api_key)
    results = existing_results[:]
    lock    = threading.Lock()
    completed_count = len(existing_results)

    def _save():
        out = {
            "meta": {
                "judge_model":  args.judge_model,
                "n_pairs":      len(results),
                "mode":         "full" if args.full else "pilot",
                "comparison":   "Transformed(A) vs OriginalHuman(B)",
                "paper_text":   "full_manuscript" if use_paper else "title_only",
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "note":         "ScholarPeer scores 1-10; 5=human baseline",
            },
            "results": results,
        }
        Path(results_file).write_text(json.dumps(out, indent=2))

    def _eval_pair(pair: dict) -> dict | None:
        title      = pair["title"]
        rew_text   = pair["transformed_text"]
        orig_text  = pair["original_text"]
        human_texts = pair["all_human_texts"]
        manuscript = pair.get("manuscript_text", "")
        paper_obj  = Paper(title=title,
                           full_text=manuscript if (use_paper and manuscript) else "")

        entry = {
            "forum_id":        pair["forum_id"],
            "review_id":       pair.get("review_id", ""),
            "title":           title,
            "paper_text_used": "full_manuscript" if (use_paper and manuscript) else "title_only",
            "note":            "A=Transformed(AI), B=Original Human",
        }

        # ScholarPeer
        try:
            entry["score"] = judge.scholarpeer(
                paper=paper_obj,
                ai_review=rew_text,
                human_reviews=human_texts or [orig_text],
            )
        except Exception as e:
            entry["score"] = {"error": str(e)}

        return entry

    # Filter to pending pairs
    pending = [p for p in pairs
               if f"{p['forum_id']}_{p.get('review_id','')}" not in done_ids]
    total_pending = len(pending)
    pw = max(1, min(args.pair_workers, total_pending))
    print(f"Pair workers : {pw}  (concurrent pairs per job)")
    print(f"Pending pairs: {total_pending}\n{'='*70}")

    with ThreadPoolExecutor(max_workers=pw) as ex:
        future_to_pair = {ex.submit(_eval_pair, p): p for p in pending}
        for future in as_completed(future_to_pair):
            pair  = future_to_pair[future]
            try:
                entry = future.result()
            except Exception as e:
                print(f"PAIR FAILED {pair['forum_id']}: {e}")
                continue
            with lock:
                completed_count += 1
                results.append(entry)
                overall = entry.get("score", {}).get("overall")
                print(f"  [{completed_count}/{len(existing_results)+total_pending}] "
                      f"{entry['title'][:55]}  "
                      f"overall={_fmt(overall)}")
                _save()

    print(f"\n{'='*70}")
    print(f"Saved: {results_file}")

    # ── Aggregate summary ─────────────────────────────────────────────────
    print("\n── ScholarPeer average scores (mean across pairs, scale 1-10, 5=human) ──")
    scholarpeer_dims = ["technical_accuracy", "constructive_value",
                        "analytical_depth", "novelty_and_significance",
                        "overall"]
    for dim in scholarpeer_dims:
        vals = [r["score"].get(dim) for r in results
                if isinstance(r.get("score"), dict) and r["score"].get(dim)]
        vals = [v for v in vals if v is not None]
        mean = sum(vals)/len(vals) if vals else None
        bar  = ">" if (mean and mean > 5) else ("<" if (mean and mean < 5) else "=")
        print(f"  {dim:30s}: {_fmt(mean)} {bar} 5.0 (human)")


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge runner for the four Table 1 metric families.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_point = sub.add_parser(
        "pointwise",
        help="ReviewEval + RottenReviews + REMOR on each review individually.",
    )
    _add_pointwise_args(p_point)

    p_scholar = sub.add_parser(
        "scholarpeer",
        help="ScholarPeer comparative scoring of each AI review vs paired human reviews.",
    )
    _add_scholarpeer_args(p_scholar)

    args = parser.parse_args()
    if args.mode == "pointwise":
        _run_pointwise(args)
    else:
        _run_scholarpeer(args)


if __name__ == "__main__":
    main()
