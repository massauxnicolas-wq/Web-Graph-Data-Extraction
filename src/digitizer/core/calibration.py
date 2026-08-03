from __future__ import annotations

from dataclasses import dataclass

import numpy as np


import cv2


@dataclass
class Calibration:
    """A solved calibration: the 3x3 pixel->calibration-space matrix plus per-axis log flags.

    For a linear axis, calibration space *is* data space. For a log axis, calibration space is
    log10(data) — the solve and transforms handle the log/exp so callers only ever see data
    values. This is the object the project file serialises and FastAPI will return.
    """
    M: np.ndarray
    x_log: bool = False
    y_log: bool = False


def _to_calib_space(data_pts: np.ndarray, x_log: bool, y_log: bool) -> np.ndarray:
    """Map data values into calibration space (log10 on log axes)."""
    out = np.array(data_pts, dtype=float, copy=True)
    if out.ndim == 1:
        out = out[None, :]
    if x_log:
        out[:, 0] = np.log10(out[:, 0])
    if y_log:
        out[:, 1] = np.log10(out[:, 1])
    return out


def solve_calibration(
    pixel_pts: np.ndarray, data_pts: np.ndarray, x_log: bool = False, y_log: bool = False,
) -> Calibration:
    """Solve a calibration, optionally treating either axis as logarithmic.

    Log axes are calibrated in log10 space, so the same linear solver handles them. Data values
    on a log axis must be strictly positive.
    """
    data_pts = np.asarray(data_pts, dtype=float)
    if x_log and (data_pts[:, 0] <= 0).any():
        raise ValueError("logarithmic X-axis requires positive axis values")
    if y_log and (data_pts[:, 1] <= 0).any():
        raise ValueError("logarithmic Y-axis requires positive axis values")
    M = affine_from_points(pixel_pts, _to_calib_space(data_pts, x_log, y_log))
    return Calibration(M=M, x_log=x_log, y_log=y_log)


def calibration_error(cal: Calibration, pixel_pts: np.ndarray, data_pts: np.ndarray) -> float:
    """Max round-trip error in calibration space (identical to round_trip_error when linear)."""
    calib_pts = _to_calib_space(np.asarray(data_pts, dtype=float), cal.x_log, cal.y_log)
    return round_trip_error(cal.M, pixel_pts, calib_pts)


def affine_from_points(pixel_pts: np.ndarray, data_pts: np.ndarray) -> np.ndarray:
    """Solve 3x3 transform M such that [Dx, Dy, w].T = M @ [Px, Py, 1].T.

    With exactly 3 points -> exact affine solve.
    With exactly 4 points -> exact perspective transform (homography).
    With >4 points -> least-squares homography.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=float)
    data_pts = np.asarray(data_pts, dtype=float)
    if pixel_pts.shape != data_pts.shape or pixel_pts.shape[1] != 2:
        raise ValueError("pixel_pts and data_pts must both be (N, 2)")
    n = pixel_pts.shape[0]
    if n < 3:
        raise ValueError("need >= 3 calibration points")

    if n == 3:
        P = np.hstack([pixel_pts, np.ones((n, 1))])
        if abs(np.linalg.det(P)) < 1e-9:
            raise ValueError("calibration points are collinear")
        AB = np.linalg.solve(P, data_pts)
        M = np.eye(3)
        M[:2, :] = AB.T
        return M
    elif n == 4:
        M = cv2.getPerspectiveTransform(np.float32(pixel_pts), np.float32(data_pts))
        return M
    else:
        M, _ = cv2.findHomography(pixel_pts, data_pts)
        if M is None:
            raise ValueError("Failed to solve homography")
        return M


def round_trip_error(M: np.ndarray, pixel_pts: np.ndarray, data_pts: np.ndarray) -> float:
    pixel_pts = np.asarray(pixel_pts, dtype=float)
    data_pts = np.asarray(data_pts, dtype=float)
    
    homo = np.hstack([pixel_pts, np.ones((pixel_pts.shape[0], 1))])
    recovered = homo @ M.T
    z = recovered[:, 2:]
    z[z == 0] = 1e-9
    recovered_xy = recovered[:, :2] / z
    
    return float(np.linalg.norm(recovered_xy - data_pts, axis=1).max())
