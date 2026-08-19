"""Load human vs rewritten score pairs and compute per-metric differences.

Pointwise metrics (ReviewEval / REMOR / RottenReviews) are paired by
``(year, forum_id)``, reconstructed from the review JSONL file order.
ScholarPeer remains an anchor comparison (score − 5.0) and is not human-paired.
"""

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

HUMAN_JSONL = Path("data/reviews/human_reviews")
REWRITTEN_JSONL = Path("data/reviews/rewritten")


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


def _forum_ids_from_jsonl(path: Path) -> list[str]:
    """Return forum_id for each JSONL record in file order."""
    if not path.exists():
        raise SystemExit(f"ERROR: required review JSONL not found: {path}")
    forums: list[str] = []
    for i, line in enumerate(path.open()):
        rec = json.loads(line)
        forum = rec.get("forum_id")
        if not isinstance(forum, str) or not forum.strip():
            raise SystemExit(f"ERROR: missing forum_id at {path}:{i}")
        if forum.isdigit():
            raise SystemExit(
                f"ERROR: forum_id looks positional, not a paper id: "
                f"{forum!r} in {path}:{i}"
            )
        forums.append(forum)
    if len(forums) != len(set(forums)):
        dupes = [f for f in forums if forums.count(f) > 1]
        raise SystemExit(f"ERROR: duplicate forum_id in {path}: {sorted(set(dupes))}")
    return forums


def human_forum_by_index(years: list[str]) -> dict[tuple[str, int], str]:
    """Map (year, JSONL/evaluated index) → forum_id from human JSONL order."""
    mapping: dict[tuple[str, int], str] = {}
    for year in years:
        forums = _forum_ids_from_jsonl(HUMAN_JSONL / f"{year}.jsonl")
        for idx, forum in enumerate(forums):
            mapping[(year, idx)] = forum
    return mapping


def rewritten_forum_by_index(
    years: list[str], models: list[str],
) -> dict[tuple[str, str, int], str]:
    """Map (year, model, per_paper index) → forum_id from rewritten JSONL order."""
    mapping: dict[tuple[str, str, int], str] = {}
    for year in years:
        for model in models:
            path = REWRITTEN_JSONL / f"{year}_{model}.jsonl"
            if not path.exists():
                continue
            forums = _forum_ids_from_jsonl(path)
            for i, forum in enumerate(forums):
                mapping[(year, model, i)] = forum
    return mapping


def load_human_scores(years: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    """Human flattened scores keyed by (year, forum_id)."""
    idx_to_forum = human_forum_by_index(years)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for year in years:
        p = _human_file(year)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        sample = d.get("human", {}).get("llm_judge_sample", {})
        per_paper = sample.get("per_paper", [])
        indices = sample.get("evaluated_indices", list(range(len(per_paper))))
        if len(indices) != len(per_paper):
            raise SystemExit(
                f"ERROR: evaluated_indices length {len(indices)} != "
                f"per_paper length {len(per_paper)} in {p}"
            )
        for idx, entry in zip(indices, per_paper):
            if (year, idx) not in idx_to_forum:
                raise SystemExit(
                    f"ERROR: human evaluated index {idx} has no JSONL forum_id "
                    f"for {year} (JSONL shorter than scores?)"
                )
            forum = idx_to_forum[(year, idx)]
            key = (year, forum)
            if key in out:
                raise SystemExit(f"ERROR: duplicate human (year, forum_id): {key}")
            if "llm_judge" in entry:
                flat = _flatten_llm_judge(entry["llm_judge"])
            else:
                flat = {k: v for k, v in entry.items() if _safe_float(v) is not None}
            out[key] = flat
    return out


def load_rewritten_rows(years: list[str], models: list[str]) -> list[dict]:
    """Rewritten per_paper rows with forum_id from the matching JSONL line."""
    forum_map = rewritten_forum_by_index(years, models)
    rows: list[dict] = []
    for year in years:
        p = Path(f"data/results/rewritten/{year}/{_judge_file(year)}")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for model in models:
            mk = f"{year}_{model}"
            block = d.get(mk, {})
            per_paper = block.get("per_paper", [])
            jsonl_n = sum(1 for (y, m, _) in forum_map if y == year and m == model)
            if per_paper and jsonl_n == 0:
                raise SystemExit(
                    f"ERROR: rewritten scores exist for {mk} but JSONL is missing: "
                    f"{REWRITTEN_JSONL / (mk + '.jsonl')}"
                )
            if per_paper and jsonl_n != len(per_paper):
                raise SystemExit(
                    f"ERROR: {mk} per_paper length {len(per_paper)} != "
                    f"JSONL length {jsonl_n}; cannot map scores to forum_id"
                )
            seen: set[str] = set()
            for i, entry in enumerate(per_paper):
                forum = forum_map[(year, model, i)]
                if forum in seen:
                    raise SystemExit(
                        f"ERROR: duplicate rewritten forum_id {forum} in {mk}"
                    )
                seen.add(forum)
                rows.append({
                    "year": year,
                    "model": model,
                    **entry,
                    "forum_id": forum,
                })
    return rows


def load_scholarpeer_pairs(years: list[str], models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for year in years:
        p = Path(f"data/results/rewritten/{year}/{_judge_file(year)}")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for model in models:
            mk = f"{year}_{model}"
            sp = d.get(mk, {}).get("ScholarPeer", {})
            for r in sp.get("results", []):
                rows.append({"year": year, "model": model, **r})
    return rows


def _validate_forum_join(
    human: dict[tuple[str, str], dict],
    rw_rows: list[dict],
    years: list[str],
    models: list[str],
) -> None:
    """Fail loudly if the (year, forum_id) join cannot be resolved 1-to-1."""
    for (year, forum) in human:
        if not isinstance(forum, str) or forum.isdigit():
            raise SystemExit(
                f"ERROR: pairing key looks positional, not a forum_id: {(year, forum)}"
            )

    unmatched_rw: list[tuple[str, str, str]] = []
    for row in rw_rows:
        year, forum, model = row["year"], row["forum_id"], row["model"]
        if (year, forum) not in human:
            unmatched_rw.append((year, model, forum))

    human_by_year: dict[str, set[str]] = {}
    for year, forum in human:
        human_by_year.setdefault(year, set()).add(forum)

    rw_by_year_model: dict[tuple[str, str], set[str]] = {}
    for row in rw_rows:
        rw_by_year_model.setdefault((row["year"], row["model"]), set()).add(row["forum_id"])

    unmatched_human: list[tuple[str, str, str]] = []
    for year in years:
        hset = human_by_year.get(year, set())
        for model in models:
            rset = rw_by_year_model.get((year, model), set())
            if not rset:
                continue
            for forum in sorted(hset - rset):
                unmatched_human.append((year, model, forum))

    print(
        f"  Pairing validation: human papers={len(human)}, "
        f"rewritten rows={len(rw_rows)}, "
        f"unmatched rewritten={len(unmatched_rw)}, "
        f"unmatched human={len(unmatched_human)}"
    )
    if unmatched_rw:
        sample = unmatched_rw[:5]
        raise SystemExit(
            "ERROR: rewritten rows with no human (year, forum_id) match "
            f"({len(unmatched_rw)}). examples={sample}"
        )
    if unmatched_human:
        sample = unmatched_human[:5]
        raise SystemExit(
            "ERROR: human papers missing from rewritten JSONL for a model "
            f"({len(unmatched_human)}). examples={sample}"
        )


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
    _validate_forum_join(human, rw_rows, years, models)

    records: list[dict] = []
    for row in rw_rows:
        key = (row["year"], row["forum_id"])
        h_flat = human[key]  # validated 1-to-1; do not fall back to index
        ai_flat = _flatten_llm_judge(row.get("llm_judge", {}))
        for k in POINTWISE_KEYS:
            ai_v = _safe_float(ai_flat.get(k))
            h_v = _safe_float(h_flat.get(k))
            if ai_v is not None and h_v is not None:
                records.append({
                    "metric_key": k,
                    "source": k.split("__", 1)[0],
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
