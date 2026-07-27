from __future__ import annotations

import numpy as np


import cv2


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
