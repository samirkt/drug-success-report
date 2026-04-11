"""Statistical utilities for the pipeline."""

import numpy as np


def wilson_ci(k, n, alpha=0.05):
    """Wilson score confidence interval for a binomial proportion.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    if n is None or k is None:
        return (np.nan, np.nan, np.nan)
    if not np.isfinite(n) or not np.isfinite(k):
        return (np.nan, np.nan, np.nan)

    n = int(n); k = int(k)
    if n <= 0:
        return (np.nan, np.nan, np.nan)

    if abs(alpha - 0.05) < 1e-12:
        z = 1.959963984540054
    else:
        raise ValueError("For non-95% CI, either install scipy or provide a z value.")

    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi
