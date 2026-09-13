"""
metrics.py
----------
Small statistics helpers used by the benchmark and calibration flows:

- aggregate_scores(): mean/median/distribution across a list of per-response
  evaluation results, plus category breakdowns.
- spearman_correlation(): correlation between human ratings and automated
  scores for the calibration set, via scipy. Falls back to a manual
  rank-correlation implementation if scipy isn't available, so the
  calibration script still runs in a minimal environment.
"""

import statistics
from collections import defaultdict
from typing import Optional


def aggregate_scores(results: list[dict]) -> dict:
    """
    results: list of dicts as returned by evaluation_service.evaluate_response,
    each optionally tagged with a "category" key for breakdowns.
    """
    if not results:
        return {"count": 0}

    overall_scores = [r["overall_score"] for r in results]
    dim_scores = defaultdict(list)
    for r in results:
        for dim, val in r["dimensions"].items():
            dim_scores[dim].append(val["score"])

    category_scores = defaultdict(list)
    for r in results:
        cat = r.get("category", "unknown")
        category_scores[cat].append(r["overall_score"])

    return {
        "count": len(results),
        "mean_overall": round(statistics.mean(overall_scores), 1),
        "median_overall": round(statistics.median(overall_scores), 1),
        "stdev_overall": round(statistics.pstdev(overall_scores), 1) if len(overall_scores) > 1 else 0.0,
        "min_overall": round(min(overall_scores), 1),
        "max_overall": round(max(overall_scores), 1),
        "dimension_means": {
            dim: round(statistics.mean(vals), 1) for dim, vals in dim_scores.items()
        },
        "category_means": {
            cat: round(statistics.mean(vals), 1) for cat, vals in category_scores.items()
        },
        "category_counts": {cat: len(vals) for cat, vals in category_scores.items()},
    }


def spearman_correlation(x: list[float], y: list[float]) -> Optional[float]:
    """Spearman rank correlation between two equal-length lists. Returns None if undefined."""
    if len(x) != len(y) or len(x) < 2:
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        if rho != rho:  # NaN check without importing math just for this
            return None
        return round(float(rho), 3)
    except ImportError:
        return _manual_spearman(x, y)


def _manual_spearman(x: list[float], y: list[float]) -> Optional[float]:
    """Fallback rank correlation if scipy is unavailable."""
    def rank(values):
        sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(sorted_idx):
            j = i
            while j + 1 < len(sorted_idx) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(x)
    d_sq_sum = sum((a - b) ** 2 for a, b in zip(rx, ry))
    denom = n * (n ** 2 - 1)
    if denom == 0:
        return None
    rho = 1 - (6 * d_sq_sum) / denom
    return round(rho, 3)
