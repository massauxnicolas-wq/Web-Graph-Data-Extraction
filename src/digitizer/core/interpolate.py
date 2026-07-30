from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def fill_gaps(xs: np.ndarray, ys: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation resample ys at x_grid.
    xs must be strictly increasing.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        raise ValueError("need >= 2 points to interpolate")
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    keep = np.concatenate([[True], np.diff(xs_sorted) > 0])
    xs_sorted = xs_sorted[keep]
    ys_sorted = ys_sorted[keep]
    if xs_sorted.size < 2:
        raise ValueError("collapsed to < 2 unique x after dedup")
    interp = interp1d(xs_sorted, ys_sorted, kind='linear', bounds_error=False, fill_value=np.nan)
    grid = np.asarray(x_grid, dtype=float)
    return interp(grid)


def polynomial_best_fit(
    xs: np.ndarray, ys: np.ndarray, degree: int, num: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares polynomial regression through (xs, ys), resampled on a uniform grid."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < degree + 1:
        raise ValueError(f"need >= {degree + 1} points to fit a degree-{degree} polynomial")
    coeffs = np.polyfit(xs, ys, degree)
    x_grid = np.linspace(xs.min(), xs.max(), num or xs.size)
    return x_grid, np.polyval(coeffs, x_grid)


def polynomial_best_fit_through_points(
    xs: np.ndarray, ys: np.ndarray, degree: int,
    x0: float, y0: float, x1: float, y1: float,
    num: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Like polynomial_best_fit, but constrained to pass exactly through (x0, y0) and (x1, y1).

    Writes p(x) = line(x) + (x - x0)(x - x1) * q(x), where line() is the
    straight line through the two fixed points and q (degree - 2) is fit by
    least squares - this guarantees p(x0)==y0 and p(x1)==y1 exactly, whatever
    q ends up being. degree == 1 has no remaining freedom: p is just line().
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if x0 == x1:
        raise ValueError("the two constraint points must have different x")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    def line(x: np.ndarray) -> np.ndarray:
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    x_grid = np.linspace(min(xs.min(initial=x0), x0, x1), max(xs.max(initial=x1), x0, x1), num or max(xs.size, 2))

    if degree == 1:
        return x_grid, line(x_grid)

    m = degree - 2
    weight = (xs - x0) * (xs - x1)
    resid = ys - line(xs)
    A = np.vander(xs, m + 1, increasing=True) * weight[:, None]
    coeffs, *_ = np.linalg.lstsq(A, resid, rcond=None)

    weight_grid = (x_grid - x0) * (x_grid - x1)
    q_grid = np.vander(x_grid, m + 1, increasing=True) @ coeffs
    return x_grid, line(x_grid) + weight_grid * q_grid


def uniform_grid(xs: np.ndarray, step: float = 1.0) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    return np.arange(xs.min(), xs.max() + step / 2, step)


def fill_gaps_parametric(xs: np.ndarray, ys: np.ndarray, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Parametric (arc-length) interpolation to resample points evenly.
    This works for self-intersecting curves, vertical lines, and loops.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return xs, ys
        
    dx_pts = np.diff(xs)
    dy_pts = np.diff(ys)
    ds = np.hypot(dx_pts, dy_pts)
    
    # Remove consecutive duplicate points (where ds == 0) to avoid interp errors
    keep = np.concatenate([[True], ds > 1e-5])
    if not keep.all():
        xs = xs[keep]
        ys = ys[keep]
        dx_pts = np.diff(xs)
        dy_pts = np.diff(ys)
        ds = np.hypot(dx_pts, dy_pts)
        
    if xs.size < 2:
        return xs, ys
        
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_grid = np.arange(0, s[-1] + step / 2, step)
    
    interp_x = interp1d(s, xs, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_y = interp1d(s, ys, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    return interp_x(s_grid), interp_y(s_grid)

