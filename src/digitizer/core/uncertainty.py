"""Per-point digitization uncertainty. Qt-free.

The dominant error in hand/auto digitizing is how precisely the curve is located in pixels.
We turn a pixel error (default ±1 px) into a data-space error via the calibration: how much the
data value moves for a `pixel_error`-pixel vertical step at each point. Linear axes give a constant
dy; log axes give a dy that grows with magnitude (the 10** mapping), which is the honest behaviour.

ponytail: calibration-resolution model first. A mask-thickness refinement (fuzzier line = larger
dy) can multiply this later without changing the signature.
"""
from __future__ import annotations

import numpy as np

from digitizer.core import transform
from digitizer.core.calibration import Calibration


def point_uncertainty(
    cal: Calibration, xs_pixel: np.ndarray, ys_pixel: np.ndarray, pixel_error: float = 1.0,
) -> np.ndarray:
    """dy in data units for a `pixel_error`-pixel vertical uncertainty at each point."""
    xs_pixel = np.asarray(xs_pixel, dtype=float)
    ys_pixel = np.asarray(ys_pixel, dtype=float)
    if xs_pixel.size == 0:
        return np.empty(0)
    base = transform.pixel_to_data(np.column_stack([xs_pixel, ys_pixel]), cal)
    shifted = transform.pixel_to_data(np.column_stack([xs_pixel, ys_pixel + pixel_error]), cal)
    return np.abs(shifted[:, 1] - base[:, 1])
