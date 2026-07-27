from __future__ import annotations

import numpy as np


def pixel_to_data(pts_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
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


def data_to_pixel(pts_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    return pixel_to_data(pts_xy, np.linalg.inv(M))
