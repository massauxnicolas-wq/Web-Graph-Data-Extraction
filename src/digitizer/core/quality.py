from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def curve_stats(xs: np.ndarray, ys: np.ndarray) -> dict:
    """Basic descriptive stats for an extracted curve."""
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return {"count": 0, "x_range": (0.0, 0.0), "largest_gap": 0.0}
    xs_sorted = np.sort(xs)
    largest_gap = float(np.diff(xs_sorted).max()) if xs.size > 1 else 0.0
    return {
        "count": int(xs.size),
        "x_range": (float(xs_sorted[0]), float(xs_sorted[-1])),
        "largest_gap": largest_gap,
    }


def detect_outliers(xs: np.ndarray, ys: np.ndarray, window: int = 11, threshold: float = 3.5) -> np.ndarray:
    """Flag points whose Y deviates abnormally from their local neighborhood.

    Rolling-median residual + modified z-score (MAD-based). Flags isolated
    spikes; a sustained step/discontinuity that persists across roughly half
    the window pulls the rolling median with it and is not flagged - this is
    a heuristic for spotting stray points, not a discontinuity classifier.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = ys.size
    if n < 5:
        return np.zeros(n, dtype=bool)

    order = np.argsort(xs)
    ys_sorted = ys[order]

    win = min(window, n if n % 2 == 1 else n - 1)
    win = max(win, 3)
    med = median_filter(ys_sorted, size=win, mode="nearest")
    resid = ys_sorted - med

    if np.ptp(resid) < 1e-9:
        return np.zeros(n, dtype=bool)  # perfectly flat residual, nothing deviates

    # MAD alone collapses to 0 whenever outliers affect less than half the
    # points (the normal case - a lone spike among otherwise-clean residuals
    # has a median-of-residuals of 0). Fall back to a small fraction of the
    # curve's own value range so sparse spikes are still catchable.
    mad = np.median(np.abs(resid - np.median(resid)))
    scale = mad if mad > 1e-9 else max(np.ptp(ys_sorted), 1e-9) * 1e-3

    modified_z = 0.6745 * resid / scale
    flagged_sorted = np.abs(modified_z) > threshold

    flagged = np.zeros(n, dtype=bool)
    flagged[order] = flagged_sorted
    return flagged
