"""Engineering curve math: derived quantities from an extracted (xs, ys). Qt-free.

Turns "I extracted points" into "I got the answer" — tangent/secant modulus, area (toughness),
initial-slope modulus, peak/min. Series-returning functions keep the same xs so their output
exports through the same NamedSeries -> build_tables path.
"""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # renamed in numpy 2.x


def derivative(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """dy/dx at each point (= tangent modulus for a stress-strain curve)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return np.zeros_like(ys)
    return np.gradient(ys, xs)


def secant_modulus(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """y/x from the origin at each point (NaN where x == 0)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(xs != 0, ys / xs, np.nan)


def area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoidal area under the curve (= toughness / energy for stress-strain)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return 0.0
    return float(_trapz(ys, xs))


def initial_slope(xs: np.ndarray, ys: np.ndarray, frac: float = 0.1) -> float:
    """Slope of a linear fit over the first `frac` of points by X (= Young's modulus)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        raise ValueError("need >= 2 points for a slope")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    n = max(2, int(round(frac * xs.size)))
    slope, _ = np.polyfit(xs[:n], ys[:n], 1)
    return float(slope)


def curve_metrics(xs: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    """Scalar summary for a readout / export sidecar."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return {}
    i_max, i_min = int(np.argmax(ys)), int(np.argmin(ys))
    return {
        "area": area(xs, ys),
        "initial_slope": initial_slope(xs, ys) if xs.size >= 2 else float("nan"),
        "peak_x": float(xs[i_max]), "peak_y": float(ys[i_max]),
        "min_x": float(xs[i_min]), "min_y": float(ys[i_min]),
    }
