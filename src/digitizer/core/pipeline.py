"""Curve-extraction pipeline: mask -> points -> post-processing.

Qt-free by design. This is the backend's brain — the same function the PyQt UI and the
future FastAPI layer both call, so nothing here may import PyQt or touch UI state. Recoverable
step failures (gap fill, best-fit) are collected in ``PipelineResult.warnings`` rather than
raised, so each caller can surface them its own way (statusBar / HTTP response).

Pixel-space in, pixel-space out — calibration (pixel <-> data) stays the caller's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from digitizer.core import interpolate, xstep


@dataclass
class ExtractionParams:
    """Everything the extraction pipeline needs beyond the mask and per-curve geometry.

    Replaces the 11-positional-argument ``extract_curve_requested`` Qt signal.
    """
    dx: int = 1
    reducer: str = "mean"
    upscale_factor: int = 1
    fill: bool = False
    smooth: bool = False
    smooth_window: int = 21
    poly_order: int = 3
    passes: int = 1
    bestfit: bool = False
    bestfit_degree: int = 3


@dataclass
class PipelineResult:
    xs: np.ndarray
    ys: np.ndarray
    warnings: list[str] = field(default_factory=list)


def run_pipeline(
    mask: np.ndarray,
    params: ExtractionParams,
    *,
    bbox=None,
    seed: tuple[float, float] | None = None,
    end: tuple[float, float] | None = None,
) -> PipelineResult:
    """Extract a curve from a boolean mask, then apply the post-processing chain."""
    sx, sy = seed if seed is not None else (None, None)
    ex, ey = end if end is not None else (None, None)
    xs, ys = xstep.extract_curve(
        mask, dx=params.dx, bbox=bbox, reducer=params.reducer,
        upscale_factor=params.upscale_factor,
        seed_x=sx, seed_y=sy, end_x=ex, end_y=ey,
    )
    if xs.size < 2:
        return PipelineResult(xs, ys, [])
    return apply_postprocessing(xs, ys, params, seed=seed, end=end)


def apply_postprocessing(
    xs: np.ndarray,
    ys: np.ndarray,
    params: ExtractionParams,
    *,
    seed: tuple[float, float] | None = None,
    end: tuple[float, float] | None = None,
) -> PipelineResult:
    """The smooth -> fill -> best-fit chain, split out so it is directly unit-testable
    without constructing a mask.

    ORDER IS LOAD-BEARING: marker-removal smoothing runs BEFORE gap fill, so we strip the
    diamond/star bumps off the raw points and then interpolate through the clean curve.
    Filling first would interpolate through bump-distorted points and leave a denser series
    that the fixed-size window smooths less effectively.
    """
    warnings: list[str] = []

    if params.smooth:
        xs, ys = _smooth(xs, ys, params)

    if params.fill:
        try:
            xs, ys = interpolate.fill_gaps_parametric(xs, ys, step=1.0)
        except ValueError as exc:
            warnings.append(f"Gap fill skipped: {exc}")

    if params.bestfit:
        try:
            xs, ys = _bestfit(xs, ys, params, seed, end)
        except ValueError as exc:
            warnings.append(f"Best-fit skipped: {exc}")

    return PipelineResult(xs, ys, warnings)


def _smooth(xs: np.ndarray, ys: np.ndarray, params: ExtractionParams) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay marker-bump removal. No-op if there are fewer points than the window."""
    if xs.size < params.smooth_window:
        return xs, ys
    win = params.smooth_window if params.smooth_window % 2 == 1 else params.smooth_window + 1
    win = min(win, xs.size if xs.size % 2 == 1 else xs.size - 1)
    poly = min(params.poly_order, win - 1)
    if win >= 5 and poly >= 1:
        for _ in range(params.passes):
            ys = savgol_filter(ys, window_length=win, polyorder=poly)
    return xs, ys


def _bestfit(xs, ys, params: ExtractionParams, seed, end) -> tuple[np.ndarray, np.ndarray]:
    if seed is not None and end is not None:
        return interpolate.polynomial_best_fit_through_points(
            xs, ys, params.bestfit_degree, seed[0], seed[1], end[0], end[1],
        )
    return interpolate.polynomial_best_fit(xs, ys, params.bestfit_degree)
