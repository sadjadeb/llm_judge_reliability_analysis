#!/usr/bin/env python3
"""Convert parent-repo results to paper archive format (ReviewEval naming, full metrics)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
PAPER1 = PARENT / "paper_human_ai_peer_review"

YEARS = [
    "ICLR2021", "ICLR2022", "ICLR2023", "ICLR2024",
    "NeurIPS2021", "NeurIPS2022", "NeurIPS2023", "NeurIPS2024",
]

DEPTH_MAP = {
    "existing_literature_comparison": "existing_literature_comparison",
    "methodological_scrutiny":        "methodological_scrutiny",
    "results_interpretation":         "results_interpretation",
    "theoretical_contributions":      "theoretical_contributions",
    "logical_gaps_identification":    "logical_gaps_identification",
    "depth_of_analysis":              "overall_depth",
}
ACTION_KEYS = ["total_insights", "actionable_insights", "actionability_score"]
REMOR_KEYS = [
    "criticism", "example", "importance_and_relevance", "materials_and_methods",
    "praise", "presentation_and_reporting", "results_and_discussion",
    "suggestion_and_solution",
]
ROTTEN_KEYS = [
    "comprehensiveness", "usage_of_technical_terms", "objectivity", "fairness",
    "actionability", "constructiveness", "relevance_alignment",
    "clarity_and_readability", "overall_quality", "overall_score_100",
]


def _nest_flat(flat: dict) -> dict:
    review_eval: dict = {}
    remor: dict = {}
    rotten: dict = {}
    for k, v in flat.items():
        if v is None:
            continue
        if k.startswith("depth_of_analysis__"):
            sub = k.split("__", 1)[1]
            review_eval["overall_depth" if sub == "depth_of_analysis" else sub] = v
        elif k.startswith("actionability__"):
            review_eval[k.split("__", 1)[1]] = v
        elif k.startswith("hprr__") and k.split("__", 1)[1] in REMOR_KEYS:
            remor[k.split("__", 1)[1]] = v
        elif k.startswith("review_inspector__"):
            sub = k.split("__", 1)[1]
            if sub in ROTTEN_KEYS:
                rotten[sub] = v
    out: dict = {}
    if review_eval:
        out["ReviewEval"] = review_eval
    if remor:
        out["REMOR"] = remor
    if rotten:
        out["RottenReviews"] = [rotten]
    return out


def _nest_nested(llm: dict) -> dict:
    out: dict = {}
    depth = llm.get("depth_of_analysis") or {}
    action = llm.get("actionability") or {}
    if depth or action:
        re = {}
        for src, dst in DEPTH_MAP.items():
            if src in depth:
                re[dst] = depth[src]
        for k in ACTION_KEYS:
            if k in action:
                re[k] = action[k]
        out["ReviewEval"] = re

    hprr = llm.get("hprr") or {}
    if hprr:
        out["REMOR"] = {k: hprr[k] for k in REMOR_KEYS if k in hprr}

    ri = llm.get("review_inspector")
    if isinstance(ri, list):
        out["RottenReviews"] = [
            {k: entry[k] for k in ROTTEN_KEYS if k in entry and entry[k] is not None}
            for entry in ri
        ]
    return out


def convert_human(judge: str) -> None:
    src_name = f"{'{year}'}.json" if judge == "gpt5mini" else "{year}_gemini.json"
    out_suffix = "gpt5mini" if judge == "gpt5mini" else "gemini"
    src_dir = PARENT / "data/results/Human LLM Judge"

    for year in YEARS:
        src = src_dir / src_name.format(year=year)
        if not src.exists():
            src = src_dir / f"{year}.json" if judge == "gpt5mini" else src_dir / f"{year}_gemini.json"
        if not src.exists():
            continue
        d = json.loads(src.read_text())
        sample = d.get("human", {}).get("llm_judge_sample", {})
        jsonl = ROOT / f"data/reviews/human_reviews/{year}.jsonl"
        forums = [json.loads(line)["forum_id"] for line in jsonl.open()] if jsonl.exists() else []
        new_pp = []
        per_paper = sample.get("per_paper", [])
        indices = sample.get("evaluated_indices", list(range(len(per_paper))))
        for idx, entry in zip(indices, per_paper):
            if "llm_judge" in entry:
                nested = _nest_nested(entry["llm_judge"])
            else:
                nested = _nest_flat(entry)
            rec: dict = {"llm_judge": nested}
            if idx < len(forums):
                rec["forum_id"] = forums[idx]
            new_pp.append(rec)
        out = {
            "human": {
                "llm_judge_sample": {
                    **{k: v for k, v in sample.items() if k != "per_paper"},
                    "per_paper": new_pp,
                }
            }
        }
        dst = ROOT / f"data/results/human_reviews/{year}_{out_suffix}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=2))
        print(f"  human {dst.name}")


def convert_rewritten(judge: str) -> None:
    src_fname = "results.json" if judge == "gpt5mini" else "results_gemini.json"
    out_fname = "gpt5mini.json" if judge == "gpt5mini" else "gemini.json"
    src_base = PARENT / "data/results/Transformations VS Human/rewritten"
    p1_base = PAPER1 / "data/results/rewritten"

    for year in YEARS:
        src = src_base / year / src_fname
        if not src.exists():
            continue
        d = json.loads(src.read_text())
        p1_path = p1_base / year / out_fname
        p1_sp = {}
        if p1_path.exists():
            p1d = json.loads(p1_path.read_text())
            p1_sp = {k: v.get("ScholarPeer") for k, v in p1d.items() if v.get("ScholarPeer")}

        out: dict = {}
        for mk, block in d.items():
            jsonl = ROOT / f"data/reviews/rewritten/{mk}.jsonl"
            forums = [json.loads(line)["forum_id"] for line in jsonl.open()] if jsonl.exists() else []
            new_pp = []
            for i, entry in enumerate(block.get("per_paper", [])):
                rec: dict = {"llm_judge": _nest_nested(entry.get("llm_judge", {}))}
                if i < len(forums):
                    rec["forum_id"] = forums[i]
                new_pp.append(rec)
            out[mk] = {"per_paper": new_pp}
            if mk in p1_sp and p1_sp[mk]:
                out[mk]["ScholarPeer"] = p1_sp[mk]

        dst = ROOT / f"data/results/rewritten/{year}/{out_fname}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=2))
        print(f"  rewritten {year}/{out_fname}")


def copy_reviews() -> None:
    for sub in ["human_reviews", "transformations/rewritten"]:
        src = PAPER1 / "data/reviews" / sub
        dst = ROOT / "data/reviews" / sub
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  reviews/{sub}")


def copy_test_results() -> None:
    src = PARENT / "results"
    dst = ROOT / "results"
    for name in [
        "tost_delta_values.csv",
        "tost_delta_values.tex",
        "tost_gemini_results.csv",
        "tost_sensitivity_results_gemini.tex",
        "tost_sensitivity_results.tex",
    ]:
        f = src / name
        if f.exists():
            shutil.copy2(f, dst / name)
            print(f"  results/{name}")


def main() -> None:
    print("Converting results...")
    convert_human("gpt5mini")
    convert_human("gemini")
    convert_rewritten("gpt5mini")
    convert_rewritten("gemini")
    print("Copying reviews...")
    copy_reviews()
    print("Copying precomputed test outputs...")
    copy_test_results()
    print("Done.")


if __name__ == "__main__":
    main()
