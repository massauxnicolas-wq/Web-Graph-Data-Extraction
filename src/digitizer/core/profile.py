"""Extraction profiles: a reusable recipe (no image, no extracted data). Qt-free.

A profile captures everything needed to re-digitize a similar chart — calibration setup, per-curve
HSV + seed/end, and the shared extraction params — so a whole datasheet family can be batched
(consumed by the CLI). apply_profile is the headless "digitize with this recipe" primitive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from digitizer.core import masking, transform
from digitizer.core.calibration import solve_calibration
from digitizer.core.export import NamedSeries
from digitizer.core.pipeline import ExtractionParams, run_pipeline


@dataclass
class ProfileCurve:
    name: str
    hsv_center: tuple[int, int, int]
    hsv_tol: tuple[int, int, int]
    seed: tuple[float, float] | None = None
    end: tuple[float, float] | None = None


@dataclass
class Profile:
    calibration_pixel_pts: list[tuple[float, float]]
    calibration_data_pts: list[tuple[float, float]]
    x_log: bool = False
    y_log: bool = False
    params: ExtractionParams = field(default_factory=ExtractionParams)  # shared across curves
    curves: list[ProfileCurve] = field(default_factory=list)


def plot_bbox(pixel_pts, img_h: int, img_w: int) -> tuple[int, int, int, int]:
    """Pixel bbox to scan, derived from the calibration points. Shared with the GUI so batch
    and interactive extraction use the same region (X between origin/x-max, Y padded below origin)."""
    xs = [p[0] for p in pixel_pts]
    ys = [p[1] for p in pixel_pts]
    y_lo, y_hi = min(ys), max(ys)
    plot_h = max(1, y_hi - y_lo)
    below = max(8, int(0.20 * plot_h))
    above = max(4, int(0.05 * plot_h))
    y_lo = max(0, int(y_lo - above))
    y_hi = min(img_h - 1, int(y_hi + below))
    x_lo = max(0, int(min(xs)))
    x_hi = min(img_w - 1, int(max(xs)))
    return (x_lo, y_lo, x_hi, y_hi)


def apply_profile(image_rgb: np.ndarray, profile: Profile) -> list[NamedSeries]:
    """Digitize an image with a saved recipe: calibrate, mask, extract, map to data space."""
    cal = solve_calibration(
        np.array(profile.calibration_pixel_pts, dtype=float),
        np.array(profile.calibration_data_pts, dtype=float),
        x_log=profile.x_log, y_log=profile.y_log,
    )
    h, w = image_rgb.shape[:2]
    bbox = plot_bbox(profile.calibration_pixel_pts, h, w)

    out: list[NamedSeries] = []
    for pc in profile.curves:
        mask = masking.hsv_mask(image_rgb, pc.hsv_center, pc.hsv_tol)
        result = run_pipeline(mask, profile.params, bbox=bbox, seed=pc.seed, end=pc.end)
        if result.xs.size < 2:
            out.append(NamedSeries(pc.name, np.empty(0), np.empty(0)))
            continue
        data = transform.pixel_to_data(np.column_stack([result.xs, result.ys]), cal)
        out.append(NamedSeries(pc.name, data[:, 0], data[:, 1]))
    return out
