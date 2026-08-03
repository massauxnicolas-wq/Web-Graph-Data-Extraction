from __future__ import annotations

import numpy as np

from digitizer.core.calibration import Calibration, _to_calib_space


def _project(pts_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Homogeneous 3x3 projection of (N, 2) points. The sole matrix-multiply seam."""
    pts = np.asarray(pts_xy, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2))
    if pts.ndim == 1:
        pts = pts[None, :]
    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = homo @ M.T
    z = out[:, 2:]
    z[z == 0] = 1e-9
    return out[:, :2] / z


def pixel_to_data(pts_xy: np.ndarray, cal: Calibration) -> np.ndarray:
    """Pixel coords -> data coords, applying 10** on logarithmic axes."""
    uv = _project(pts_xy, cal.M)
    if uv.size == 0:
        return uv
    if cal.x_log:
        uv[:, 0] = 10.0 ** uv[:, 0]
    if cal.y_log:
        uv[:, 1] = 10.0 ** uv[:, 1]
    return uv


def data_to_pixel(pts_xy: np.ndarray, cal: Calibration) -> np.ndarray:
    """Data coords -> pixel coords (inverse of pixel_to_data), via log10 on logarithmic axes."""
    pts = np.asarray(pts_xy, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2))
    calib_pts = _to_calib_space(pts, cal.x_log, cal.y_log)
    return _project(calib_pts, np.linalg.inv(cal.M))
