import numpy as np

from digitizer.core import interpolate
from digitizer.core.pipeline import (
    ExtractionParams,
    _smooth,
    apply_postprocessing,
    run_pipeline,
)


def _line_mask(h=50, w=100, slope=0.2, intercept=10):
    mask = np.zeros((h, w), dtype=bool)
    for x in range(w):
        y = int(intercept + slope * x)
        if 0 <= y < h:
            mask[y, x] = True
    return mask


def _marker_curve():
    """A flat line carrying sharp single-sample 'marker' spikes every 10 samples."""
    xs = np.arange(60, dtype=float)
    ys = np.zeros(60, dtype=float)
    ys[::10] = 15.0
    return xs, ys


def test_smooth_runs_before_fill():
    # The load-bearing regression: marker-removal smoothing must precede gap fill.
    xs, ys = _marker_curve()
    p = ExtractionParams(smooth=True, smooth_window=11, poly_order=2, fill=True)

    res = apply_postprocessing(xs.copy(), ys.copy(), p)

    # Reference A — the intended order: smooth THEN fill.
    sx, sy = _smooth(xs.copy(), ys.copy(), p)
    _, correct_y = interpolate.fill_gaps_parametric(sx, sy, step=1.0)
    # Reference B — the wrong order: fill THEN smooth.
    fx, fy = interpolate.fill_gaps_parametric(xs.copy(), ys.copy(), step=1.0)
    _, wrong_y = _smooth(fx, fy, p)

    # Implemented output matches smooth-then-fill exactly...
    assert np.allclose(res.ys, correct_y)
    # ...and is not the fill-then-smooth result (order genuinely matters).
    assert res.ys.shape != wrong_y.shape or not np.allclose(res.ys, wrong_y)

    # Behavioural justification: smoothing first attenuates the markers more, so the
    # wrong order leaves taller residual spikes.
    assert res.ys.max() < ys.max()
    assert res.ys.max() < wrong_y.max()


def test_run_pipeline_line_mask():
    mask = _line_mask(slope=0.2, intercept=10)
    res = run_pipeline(mask, ExtractionParams(dx=2, reducer="mean"))
    assert res.xs.size >= 2
    assert res.warnings == []
    # x should be non-decreasing across the traced line
    assert np.all(np.diff(res.xs) >= 0)


def test_bestfit_failure_warns_instead_of_raising():
    # Too few points for the requested degree -> polynomial_best_fit raises ValueError,
    # which the pipeline must capture as a warning rather than propagate.
    xs = np.arange(3, dtype=float)
    ys = np.array([0.0, 1.0, 0.0])
    res = apply_postprocessing(xs, ys, ExtractionParams(bestfit=True, bestfit_degree=5))
    assert res.warnings
    assert "Best-fit skipped" in res.warnings[0]
    assert res.xs.size == 3  # untouched
