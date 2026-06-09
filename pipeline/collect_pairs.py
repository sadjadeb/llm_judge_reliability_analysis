"""Load human vs rewritten score pairs and compute per-metric differences."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from metric_registry import (
    MODELS,
    POINTWISE_KEYS,
    SCHOLARPEER_ANCHOR,
    SCHOLARPEER_KEYS,
    YEARS,
)

_JUDGE_MODE = "gpt5mini"


def _judge_file(year: str) -> str:
    return "gemini.json" if _JUDGE_MODE == "gemini" else "gpt5mini.json"


def _human_file(year: str) -> Path:
    suffix = "gemini" if _JUDGE_MODE == "gemini" else "gpt5mini"
    return Path(f"data/results/human_reviews/{year}_{suffix}.json")


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _flatten_llm_judge(llm: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    if not llm:
        return flat
    for group, val in llm.items():
        if isinstance(val, dict):
            for k, v in val.items():
                fv = _safe_float(v)
                if fv is not None:
                    flat[f"{group}__{k}"] = fv
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            num_keys = [k for k in val[0] if _safe_float(val[0].get(k)) is not None]
            for k in num_keys:
                vals = [_safe_float(item.get(k)) for item in val]
                vals = [v for v in vals if v is not None]
                if vals:
                    flat[f"{group}__{k}"] = sum(vals) / len(vals)
    return flat


def load_human_scores(years: list[str]) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for year in years:
        p = _human_file(year)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        sample = d.get("human", {}).get("llm_judge_sample", {})
        per_paper = sample.get("per_paper", [])
        indices = sample.get("evaluated_indices", list(range(len(per_paper))))
        for idx, entry in zip(indices, per_paper):
            if "llm_judge" in entry:
                flat = _flatten_llm_judge(entry["llm_judge"])
            else:
                flat = {k: v for k, v in entry.items() if _safe_float(v) is not None}
            out[(year, idx)] = flat
    return out


def load_rewritten_rows(years: list[str], models: list[str]) -> list[dict]:
    rows: list[dict] = []
    fname = _judge_file(years[0]) if years else "gpt5mini.json"
    for year in years:
        p = Path(f"data/results/rewritten/{year}/{fname}")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for model in models:
            mk = f"{year}_{model}"
            block = d.get(mk, {})
            for i, entry in enumerate(block.get("per_paper", [])):
                rows.append({"year": year, "model": model, "paper_idx": i, **entry})
    return rows


def load_scholarpeer_pairs(years: list[str], models: list[str]) -> list[dict]:
    rows: list[dict] = []
    fname = _judge_file(years[0]) if years else "gpt5mini.json"
    for year in years:
        p = Path(f"data/results/rewritten/{year}/{fname}")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for model in models:
            mk = f"{year}_{model}"
            sp = d.get(mk, {}).get("ScholarPeer", {})
            for r in sp.get("results", []):
                rows.append({"year": year, "model": model, **r})
    return rows


def collect_observations(
    years: list[str] | None = None,
    models: list[str] | None = None,
    judge: str = "gpt5mini",
) -> pd.DataFrame:
    global _JUDGE_MODE
    _JUDGE_MODE = judge
    years = years or YEARS
    models = models or MODELS

    human = load_human_scores(years)
    rw_rows = load_rewritten_rows(years, models)
    sp_rows = load_scholarpeer_pairs(years, models)

    records: list[dict] = []
    for row in rw_rows:
        hkey = (row["year"], row["paper_idx"])
        ai_flat = _flatten_llm_judge(row.get("llm_judge", {}))
        h_flat = human.get(hkey, {})
        for key in POINTWISE_KEYS:
            ai_v = _safe_float(ai_flat.get(key))
            h_v = _safe_float(h_flat.get(key))
            if ai_v is not None and h_v is not None:
                records.append({
                    "metric_key": key,
                    "source": key.split("__", 1)[0],
                    "diff": ai_v - h_v,
                })

    for row in sp_rows:
        score = row.get("score") or {}
        if isinstance(score, dict):
            for dim, internal in [
                ("technical_accuracy", "ScholarPeer__technical_accuracy"),
                ("constructive_value", "ScholarPeer__constructive_value"),
                ("analytical_depth", "ScholarPeer__analytical_depth"),
                ("novelty_and_significance", "ScholarPeer__novelty_and_significance"),
                ("overall", "ScholarPeer__overall"),
            ]:
                v = _safe_float(score.get(dim))
                if v is not None:
                    records.append({
                        "metric_key": internal,
                        "source": "ScholarPeer",
                        "diff": v - SCHOLARPEER_ANCHOR,
                    })

    return pd.DataFrame(records)
