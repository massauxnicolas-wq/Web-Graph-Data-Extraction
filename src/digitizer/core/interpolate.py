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

