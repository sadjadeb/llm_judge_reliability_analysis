#!/usr/bin/env python3
"""
run_analysis.py — Wilcoxon (surface sensitivity) + TOST (robustness) tests.

Compares original human review scores with meaning-preserving rewritten scores
for 29 content-oriented metrics (3 style metrics excluded a priori).

Usage:
    python pipeline/run_analysis.py              # GPT-5-mini judge
    python pipeline/run_analysis.py --gemini     # Gemini judge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from collect_pairs import collect_observations


def _tost_pvalue(diffs: np.ndarray, bound: float) -> float:
    """One-sample TOST p-value (max of two one-sided tests)."""
    t_lo, p_lo = stats.ttest_1samp(diffs, -bound, alternative="greater")
    t_hi, p_hi = stats.ttest_1samp(diffs, bound, alternative="less")
    return float(max(p_lo, p_hi))
from metric_registry import (
    ALPHA_CORRECTED,
    EXCLUDED_KEYS,
    METRIC_NAME,
    N_METRICS,
    RETAINED_METRICS,
)

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def per_metric_stats(obs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, paper_name, source in RETAINED_METRICS:
        diffs = obs_df.loc[obs_df["metric_key"] == key, "diff"].astype(float).to_numpy()
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size < 5:
            print(f"  WARN: {paper_name} has only {diffs.size} observations", file=sys.stderr)
            continue

        try:
            _, p_wilcoxon = stats.wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            p_wilcoxon = 1.0
        p_wilcoxon = float(p_wilcoxon)
        sensitive = bool(p_wilcoxon < ALPHA_CORRECTED)

        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
        delta = 0.2 * sd_d

        p_tost = _tost_pvalue(diffs, delta) if delta > 0 else 1.0
        robust = bool(p_tost < ALPHA_CORRECTED)

        rows.append({
            "metric_key":  key,
            "metric":      paper_name,
            "source":      source,
            "n_obs":       int(diffs.size),
            "mean_delta":  mean_d,
            "sd_delta":    sd_d,
            "delta_bound": delta,
            "wilcoxon_p":  p_wilcoxon,
            "tost_p":      p_tost,
            "sensitive":   sensitive,
            "robust":      robust,
        })
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def print_summary(df: pd.DataFrame) -> None:
    n_sens = int(df["sensitive"].sum())
    n_robust = int(df["robust"].sum())
    n_both = int((df["sensitive"] & df["robust"]).sum())
    n_sens_only = int((df["sensitive"] & ~df["robust"]).sum())
    n_robust_only = int((~df["sensitive"] & df["robust"]).sum())
    n_neither = int((~df["sensitive"] & ~df["robust"]).sum())
    n_obs = int(df["n_obs"].iloc[0]) if len(df) else 0

    print()
    print("=" * 72)
    print(f"  Metrics analyzed : {len(df)}")
    print(f"  Excluded a priori  : {len(EXCLUDED_KEYS)}")
    for k in EXCLUDED_KEYS:
        print(f"    - {METRIC_NAME[k]}")
    print(f"  alpha_corrected    : {ALPHA_CORRECTED:.7f}")
    print(f"  Observations/metric: {n_obs:,}")
    print(f"  Sensitive only     : {n_sens_only}")
    print(f"  Robust only        : {n_robust_only}")
    print(f"  Sensitive ∩ Robust: {n_both}")
    print(f"  Inconclusive       : {n_neither}")
    print(f"  Sensitive (total)  : {n_sens}")
    print(f"  Robust (total)     : {n_robust}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", action="store_true", help="Use Gemini judge results")
    args = parser.parse_args()

    judge = "gemini" if args.gemini else "gpt5mini"
    judge_label = "Gemini-2.5-Flash" if args.gemini else "GPT-5-mini"

    print(f"Surface sensitivity analysis — judge: {judge_label}")
    obs_df = collect_observations(judge=judge)
    obs_df = obs_df[~obs_df["metric_key"].isin(EXCLUDED_KEYS)]
    print(f"  Observations: {len(obs_df):,} across {obs_df['metric_key'].nunique()} metrics")

    df = per_metric_stats(obs_df)
    if len(df) != N_METRICS:
        missing = {k for k, _, _ in RETAINED_METRICS} - set(df["metric_key"])
        print(f"  ERROR: expected {N_METRICS} metrics, got {len(df)}", file=sys.stderr)
        if missing:
            print(f"  Missing: {missing}", file=sys.stderr)
        sys.exit(1)

    order = {k: i for i, (k, _, _) in enumerate(RETAINED_METRICS)}
    df = df.sort_values("metric_key", key=lambda s: s.map(order)).reset_index(drop=True)
    print_summary(df)

    suffix = "_gemini" if args.gemini else "_gpt5mini"
    save_csv(df, OUT_DIR / f"sensitivity_results{suffix}.csv")


if __name__ == "__main__":
    main()
