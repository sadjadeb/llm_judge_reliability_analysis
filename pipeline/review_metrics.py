"""
review_metrics.py — LLM-as-judge metric computation for surface-sensitivity
analysis (29 content-oriented metrics from ReviewEval, REMOR, RottenReviews,
ScholarPeer).

Prompts are reproduced verbatim from the source papers; see ``prompts.py``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import numpy as np

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class Review:
    text: str


@dataclass
class Paper:
    title: str = ""
    abstract: str = ""
    full_text: str = ""


@dataclass
class EvaluationInput:
    paper: Paper
    ai_reviews: List[Review] = field(default_factory=list)
    human_reviews: List[Review] = field(default_factory=list)


class LLMJudge:
    """Wrapper for LLM-as-judge evaluation."""

    try:
        from .prompts import (
            PROMPT_LITERATURE as _PROMPT_LITERATURE,
            PROMPT_METHODOLOGY as _PROMPT_METHODOLOGY,
            PROMPT_RESULTS as _PROMPT_RESULTS,
            PROMPT_THEORETICAL as _PROMPT_THEORETICAL,
            PROMPT_LOGICAL_GAPS as _PROMPT_LOGICAL_GAPS,
            PROMPT_EXTRACT_CRITICISM as _PROMPT_EXTRACT_CRITICISM,
            PROMPT_EXTRACT_SUGGESTIONS as _PROMPT_EXTRACT_SUGGESTIONS,
            PROMPT_EXTRACT_METHODOLOGICAL as _PROMPT_EXTRACT_METHODOLOGICAL,
            PROMPT_CHECK_SPECIFICITY as _PROMPT_CHECK_SPECIFICITY,
            PROMPT_CHECK_FEASIBILITY as _PROMPT_CHECK_FEASIBILITY,
            PROMPT_CHECK_IMPLEMENTATION as _PROMPT_CHECK_IMPLEMENTATION,
            SCHOLARPEER_SYSTEM as _SCHOLARPEER_SYSTEM,
            SCHOLARPEER_USER as _SCHOLARPEER_USER,
            REMOR_PROMPT as _REMOR_PROMPT,
            ROTTENREVIEWS_PROMPT as _ROTTENREVIEWS_PROMPT,
        )
    except ImportError:
        from prompts import (
            PROMPT_LITERATURE as _PROMPT_LITERATURE,
            PROMPT_METHODOLOGY as _PROMPT_METHODOLOGY,
            PROMPT_RESULTS as _PROMPT_RESULTS,
            PROMPT_THEORETICAL as _PROMPT_THEORETICAL,
            PROMPT_LOGICAL_GAPS as _PROMPT_LOGICAL_GAPS,
            PROMPT_EXTRACT_CRITICISM as _PROMPT_EXTRACT_CRITICISM,
            PROMPT_EXTRACT_SUGGESTIONS as _PROMPT_EXTRACT_SUGGESTIONS,
            PROMPT_EXTRACT_METHODOLOGICAL as _PROMPT_EXTRACT_METHODOLOGICAL,
            PROMPT_CHECK_SPECIFICITY as _PROMPT_CHECK_SPECIFICITY,
            PROMPT_CHECK_FEASIBILITY as _PROMPT_CHECK_FEASIBILITY,
            PROMPT_CHECK_IMPLEMENTATION as _PROMPT_CHECK_IMPLEMENTATION,
            SCHOLARPEER_SYSTEM as _SCHOLARPEER_SYSTEM,
            SCHOLARPEER_USER as _SCHOLARPEER_USER,
            REMOR_PROMPT as _REMOR_PROMPT,
            ROTTENREVIEWS_PROMPT as _ROTTENREVIEWS_PROMPT,
        )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_parallel: int = 16,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.model = model
        self.temperature = temperature
        self.max_parallel = max(1, max_parallel)
        self._llm_slot = threading.BoundedSemaphore(self.max_parallel)

    def _call(self, system: str, user: str) -> str:
        with self._llm_slot:
            kwargs: dict = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=120,
            )
            if self.temperature != 1.0:
                kwargs["temperature"] = self.temperature
            resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _call_many(self, pairs: List[tuple[str, str]]) -> List[str]:
        if not pairs:
            return []
        if len(pairs) == 1:
            s, u = pairs[0]
            return [self._call(s, u)]
        max_w = min(len(pairs), self.max_parallel)
        with ThreadPoolExecutor(max_workers=max_w) as ex:
            return list(ex.map(lambda p: self._call(p[0], p[1]), pairs))

    @staticmethod
    def _parse_score(text: str, label: str = "Score") -> float:
        m = re.search(rf"{label}:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r"\d+(?:\.\d+)?", text)
        return float(m.group()) if m else 0.0

    def review_eval_depth(self, review_text: str) -> Dict[str, float]:
        sys_prompt = "You are a helpful AI assistant. Do the needful."
        templates = {
            "existing_literature_comparison": self._PROMPT_LITERATURE,
            "methodological_scrutiny":        self._PROMPT_METHODOLOGY,
            "results_interpretation":         self._PROMPT_RESULTS,
            "theoretical_contributions":      self._PROMPT_THEORETICAL,
            "logical_gaps_identification":    self._PROMPT_LOGICAL_GAPS,
        }
        dim_order = list(templates.keys())
        pairs = [(sys_prompt, templates[k].format(review=review_text)) for k in dim_order]
        resps = self._call_many(pairs)
        per_dim = {dim: self._parse_score(r) / 3.0 for dim, r in zip(dim_order, resps)}
        per_dim["overall_depth"] = sum(per_dim.values()) / 5.0
        return per_dim

    def review_eval_actionability(self, review_text: str) -> Dict[str, float]:
        sys = "You are a helpful AI assistant."
        extract_pairs = [
            (sys, self._PROMPT_EXTRACT_CRITICISM.format(review=review_text)),
            (sys, self._PROMPT_EXTRACT_SUGGESTIONS.format(review=review_text)),
            (sys, self._PROMPT_EXTRACT_METHODOLOGICAL.format(review=review_text)),
        ]
        criticism_resp, suggestions_resp, methodological_resp = self._call_many(extract_pairs)

        def _parse_list(text: str) -> List[str]:
            items = []
            for line in text.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    items.append(line[2:].strip())
                elif line and line[0].isdigit() and ". " in line:
                    items.append(line.split(". ", 1)[1].strip())
                elif line:
                    items.append(line)
            return [i for i in items if i]

        all_insights = (
            _parse_list(criticism_resp)
            + _parse_list(suggestions_resp)
            + _parse_list(methodological_resp)
        )
        if not all_insights:
            return {"total_insights": 0.0, "actionable_insights": 0.0, "actionability_score": 0.0}

        check_pairs: List[tuple[str, str]] = []
        for insight in all_insights:
            check_pairs.append((sys, self._PROMPT_CHECK_SPECIFICITY.format(insight=insight)))
            check_pairs.append((sys, self._PROMPT_CHECK_FEASIBILITY.format(insight=insight)))
            check_pairs.append((sys, self._PROMPT_CHECK_IMPLEMENTATION.format(insight=insight)))
        check_resps = self._call_many(check_pairs)

        actionable_count = 0
        for i in range(len(all_insights)):
            base = i * 3
            score = sum(self._parse_score(check_resps[base + k]) for k in range(3))
            if score > 1:
                actionable_count += 1

        return {
            "total_insights":      float(len(all_insights)),
            "actionable_insights": float(actionable_count),
            "actionability_score": actionable_count / len(all_insights),
        }

    def remor(self, review_text: str) -> Dict[str, float]:
        prompt = self._REMOR_PROMPT.format(review=review_text)
        resp = self._call("You are a helpful AI assistant.", prompt)
        patterns = {
            "criticism":                  r"Criticism:\s*(\d+\.?\d*)",
            "example":                    r"Example:\s*(\d+\.?\d*)",
            "importance_and_relevance":   r"Importance.*?Relevance:\s*(\d+\.?\d*)",
            "materials_and_methods":      r"Materials.*?Methods:\s*(\d+\.?\d*)",
            "praise":                     r"Praise:\s*(\d+\.?\d*)",
            "presentation_and_reporting": r"Presentation.*?Reporting:\s*(\d+\.?\d*)",
            "results_and_discussion":     r"Results.*?Discussion:\s*(\d+\.?\d*)",
            "suggestion_and_solution":    r"Suggestion.*?Solution:\s*(\d+\.?\d*)",
        }
        aspects: Dict[str, float] = {}
        for key, pat in patterns.items():
            m = re.search(pat, resp, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                aspects[key] = v if 0.0 <= v <= 1.0 else 0.0
            else:
                aspects[key] = 0.0
        return aspects

    def rotten_reviews(
        self,
        review_text: str,
        paper_title: str = "",
        paper_abstract: str = "",
    ) -> Dict[str, float]:
        import json as _json

        prompt = self._ROTTENREVIEWS_PROMPT.format(
            title=paper_title or "",
            abstract=paper_abstract or "",
            review_text=review_text,
        )
        resp = self._call("You are a helpful AI assistant.", prompt)

        tag_match = re.search(
            r"<review_assessment>\s*(\{.*?\})\s*</review_assessment>",
            resp, re.DOTALL,
        )
        block = tag_match.group(1) if tag_match else ""
        if not block:
            block_match = re.search(r"\{[\s\S]*\}", resp)
            block = block_match.group(0) if block_match else ""

        payload: Dict[str, Any] = {}
        if block:
            try:
                payload = _json.loads(block)
            except _json.JSONDecodeError:
                payload = {}

        criteria = payload.get("criteria", {}) if isinstance(payload, dict) else {}
        prompt_to_key = {
            "Comprehensiveness":        "comprehensiveness",
            "Usage of Technical Terms": "usage_of_technical_terms",
            "Objectivity":              "objectivity",
            "Fairness":                 "fairness",
            "Actionability":            "actionability",
            "Constructiveness":         "constructiveness",
            "Relevance Alignment":      "relevance_alignment",
            "Clarity and Readability":  "clarity_and_readability",
            "Overall Quality":          "overall_quality",
            "Overall Score":            "overall_score_100",
        }
        out: Dict[str, float] = {}
        for prompt_key, key in prompt_to_key.items():
            val = criteria.get(prompt_key)
            try:
                out[key] = float(val) if val is not None else None
            except (TypeError, ValueError):
                out[key] = None
        return out

    def scholarpeer(
        self,
        paper: Paper,
        ai_review: str,
        human_reviews: List[str],
        cutoff_date: str = "",
    ) -> Dict[str, float]:
        import json as _json

        hr_block = "\n\n".join(
            f"Human Review {i + 1}:\n{r}" for i, r in enumerate(human_reviews)
        )
        paper_text = paper.full_text or f"Title: {paper.title}\nAbstract: {paper.abstract}"
        system = self._SCHOLARPEER_SYSTEM.format(cutoff_date=cutoff_date or "N/A")
        user = self._SCHOLARPEER_USER.format(
            paper_text=paper_text, ai_review=ai_review, human_review=hr_block,
        )
        resp = self._call(system, user)

        json_match = re.search(r"```json\s*(.*?)```", resp, re.DOTALL)
        if json_match:
            try:
                data = _json.loads(json_match.group(1))
                dims = {
                    "technical_accuracy":       "Technical Accuracy Score",
                    "constructive_value":       "Constructive Value Score",
                    "analytical_depth":         "Analytical Depth Score",
                    "novelty_and_significance": "Novelty and Significance Assessment Score",
                    "overall":                  "Overall Score",
                }
                return {key: float(data.get(json_key, 5)) for key, json_key in dims.items()}
            except (_json.JSONDecodeError, KeyError, TypeError):
                pass

        dims_fallback = {
            "technical_accuracy":       "Technical Accuracy Score",
            "constructive_value":       "Constructive Value Score",
            "analytical_depth":         "Analytical Depth Score",
            "novelty_and_significance": "Novelty and Significance Assessment Score",
            "overall":                  "Overall Score",
        }
        scores: Dict[str, float] = {}
        for key, label in dims_fallback.items():
            m = re.search(rf'"{label}":\s*(\d+(?:\.\d+)?)', resp, re.IGNORECASE)
            scores[key] = float(m.group(1)) if m else 5.0
        return scores


METRIC_GROUPS = ("llm_judge",)


class ReviewEvaluationPipeline:
    def __init__(self, llm_judge: Optional[LLMJudge] = None):
        self.llm_judge = llm_judge

    @staticmethod
    def _avg_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
        if not dicts:
            return {}
        keys = dicts[0].keys()
        return {k: float(np.mean([d[k] for d in dicts if k in d])) for k in keys}

    def evaluate_single(
        self,
        inp: EvaluationInput,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        groups = set(metrics or METRIC_GROUPS)
        result: Dict[str, Any] = {}
        if "llm_judge" not in groups or not self.llm_judge:
            return result

        first_ai = inp.ai_reviews[0]
        n_tasks = 3 + len(inp.ai_reviews)
        pool = max(2, min(n_tasks, self.llm_judge.max_parallel))

        with ThreadPoolExecutor(max_workers=pool) as ex:
            f_depth = ex.submit(self.llm_judge.review_eval_depth, first_ai.text)
            f_action = ex.submit(self.llm_judge.review_eval_actionability, first_ai.text)
            f_remor = ex.submit(self.llm_judge.remor, first_ai.text)
            f_inspector = [
                ex.submit(
                    self.llm_judge.rotten_reviews,
                    r.text, inp.paper.title, inp.paper.abstract,
                )
                for r in inp.ai_reviews
            ]

            review_eval: Dict[str, float] = {}
            review_eval.update(f_depth.result())
            review_eval.update(f_action.result())

            llm_res: Dict[str, Any] = {
                "ReviewEval":    review_eval,
                "REMOR":         f_remor.result(),
            }
            try:
                llm_res["RottenReviews"] = [f.result() for f in f_inspector]
            except Exception as exc:
                logger.warning("RottenReviews failed: %s", exc)

        result["llm_judge"] = llm_res
        return result

    def evaluate(
        self,
        inputs: List[EvaluationInput],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        per_paper = [self.evaluate_single(inp, metrics) for inp in inputs]
        corpus: Dict[str, Any] = {}
        avg: Dict[str, Any] = {}
        for group in METRIC_GROUPS:
            group_dicts = [
                p[group] for p in per_paper
                if group in p and isinstance(p[group], dict)
                and all(isinstance(v, (int, float)) for v in p[group].values())
            ]
            if group_dicts:
                avg[group] = self._avg_dicts(group_dicts)
        corpus["averaged_per_paper"] = avg
        return {"per_paper": per_paper, "corpus": corpus}
