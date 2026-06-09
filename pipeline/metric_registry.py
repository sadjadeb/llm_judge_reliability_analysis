"""Metric inventory for surface-sensitivity analysis (29 retained of 32 scored)."""

from __future__ import annotations

# (internal_key, display_name, source)
METRICS: list[tuple[str, str, str]] = [
    # ReviewEval (9)
    ("ReviewEval__existing_literature_comparison", "Literature Comparison", "ReviewEval"),
    ("ReviewEval__methodological_scrutiny",        "Methodological Scrutiny", "ReviewEval"),
    ("ReviewEval__results_interpretation",         "Results Interpretation", "ReviewEval"),
    ("ReviewEval__theoretical_contributions",      "Theoretical Contributions", "ReviewEval"),
    ("ReviewEval__logical_gaps_identification",    "Logical Gaps Identification", "ReviewEval"),
    ("ReviewEval__overall_depth",                  "Overall Depth", "ReviewEval"),
    ("ReviewEval__total_insights",                 "Total Insights", "ReviewEval"),
    ("ReviewEval__actionable_insights",            "Actionable Insights", "ReviewEval"),
    ("ReviewEval__actionability_score",            "Actionability Score", "ReviewEval"),
    # REMOR (8)
    ("REMOR__criticism",                 "Criticism", "REMOR"),
    ("REMOR__example",                   "Example", "REMOR"),
    ("REMOR__importance_and_relevance",  "Importance & Relevance", "REMOR"),
    ("REMOR__materials_and_methods",     "Materials & Methods", "REMOR"),
    ("REMOR__praise",                    "Praise", "REMOR"),
    ("REMOR__presentation_and_reporting","Presentation & Reporting", "REMOR"),
    ("REMOR__results_and_discussion",    "Results & Discussion", "REMOR"),
    ("REMOR__suggestion_and_solution",   "Suggestion & Solution", "REMOR"),
    # RottenReviews (10)
    ("RottenReviews__comprehensiveness",        "Comprehensiveness", "RottenReviews"),
    ("RottenReviews__usage_of_technical_terms", "Usage of Technical Terms", "RottenReviews"),
    ("RottenReviews__objectivity",              "Objectivity", "RottenReviews"),
    ("RottenReviews__fairness",                 "Fairness", "RottenReviews"),
    ("RottenReviews__actionability",            "Actionability", "RottenReviews"),
    ("RottenReviews__constructiveness",         "Constructiveness", "RottenReviews"),
    ("RottenReviews__relevance_alignment",      "Relevance Alignment", "RottenReviews"),
    ("RottenReviews__clarity_and_readability",  "Clarity & Readability", "RottenReviews"),
    ("RottenReviews__overall_quality",          "Overall Quality", "RottenReviews"),
    ("RottenReviews__overall_score_100",        "Overall Score", "RottenReviews"),
    # ScholarPeer (5)
    ("ScholarPeer__technical_accuracy",        "Technical Accuracy", "ScholarPeer"),
    ("ScholarPeer__constructive_value",        "Constructive Value", "ScholarPeer"),
    ("ScholarPeer__analytical_depth",          "Analytical Depth", "ScholarPeer"),
    ("ScholarPeer__novelty_and_significance",  "Novelty & Significance", "ScholarPeer"),
    ("ScholarPeer__overall",                   "Overall", "ScholarPeer"),
]

EXCLUDED_KEYS = frozenset({
    "REMOR__presentation_and_reporting",
    "RottenReviews__usage_of_technical_terms",
    "RottenReviews__clarity_and_readability",
})

RETAINED_METRICS = [t for t in METRICS if t[0] not in EXCLUDED_KEYS]
N_METRICS = len(RETAINED_METRICS)
ALPHA = 0.05
ALPHA_CORRECTED = ALPHA / N_METRICS

METRIC_NAME: dict[str, str] = {k: n for k, n, _ in METRICS}
METRIC_SOURCE: dict[str, str] = {k: s for k, _, s in METRICS}

SCALE_LABEL: dict[str, str] = {
    "ReviewEval__existing_literature_comparison": "[0,1]",
    "ReviewEval__methodological_scrutiny":        "[0,1]",
    "ReviewEval__results_interpretation":         "[0,1]",
    "ReviewEval__theoretical_contributions":      "[0,1]",
    "ReviewEval__logical_gaps_identification":    "[0,1]",
    "ReviewEval__overall_depth":                  "[0,1]",
    "ReviewEval__total_insights":                 r"[0,$\infty$)",
    "ReviewEval__actionable_insights":            r"[0,$\infty$)",
    "ReviewEval__actionability_score":            "[0,1]",
    "REMOR__criticism":                  "[0,1]",
    "REMOR__example":                    "[0,1]",
    "REMOR__importance_and_relevance":   "[0,1]",
    "REMOR__materials_and_methods":      "[0,1]",
    "REMOR__praise":                     "[0,1]",
    "REMOR__results_and_discussion":     "[0,1]",
    "REMOR__suggestion_and_solution":    "[0,1]",
    "RottenReviews__comprehensiveness":   "[0,5]",
    "RottenReviews__objectivity":         "[0,5]",
    "RottenReviews__fairness":            "[0,5]",
    "RottenReviews__actionability":       "[0,5]",
    "RottenReviews__constructiveness":    "[0,5]",
    "RottenReviews__relevance_alignment": "[0,5]",
    "RottenReviews__overall_quality":     "[0,5]",
    "RottenReviews__overall_score_100":   "[0,100]",
    "ScholarPeer__technical_accuracy":        "[1,10]",
    "ScholarPeer__constructive_value":        "[1,10]",
    "ScholarPeer__analytical_depth":          "[1,10]",
    "ScholarPeer__novelty_and_significance":  "[1,10]",
    "ScholarPeer__overall":                   "[1,10]",
}

SOURCE_ORDER = ["ReviewEval", "REMOR", "RottenReviews", "ScholarPeer"]
SCHOLARPEER_ANCHOR = 5.0

MODELS = [
    "google_gemini-2.5-flash",
    "openai_gpt-5",
    "openai_o4-mini",
    "openrouter_anthropic_claude-haiku-4.5",
    "openrouter_deepseek_deepseek-r1",
    "openrouter_meta-llama_llama-4-scout",
]

YEARS = [
    "ICLR2021", "ICLR2022", "ICLR2023", "ICLR2024",
    "NeurIPS2021", "NeurIPS2022", "NeurIPS2023", "NeurIPS2024",
]

POINTWISE_KEYS = frozenset(k for k, _, _ in METRICS if not k.startswith("ScholarPeer__"))
SCHOLARPEER_KEYS = frozenset(k for k, _, _ in METRICS if k.startswith("ScholarPeer__"))
