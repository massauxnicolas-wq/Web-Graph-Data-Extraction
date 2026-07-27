from __future__ import annotations

import cv2
import numpy as np


def rgb_pixel_to_hsv(rgb: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    px = rgb[y:y + 1, x:x + 1]
    hsv = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def hsv_mask(
    rgb: np.ndarray,
    hsv_center: tuple[int, int, int],
    tol: tuple[int, int, int],
) -> np.ndarray:
    """Boolean mask where pixel HSV lies within `tol` of `hsv_center`.

    Hue wraps at 180 (OpenCV convention). Saturation/value clamp to [0, 255].
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_center
    dh, ds, dv = tol
    s_lo, s_hi = max(0, s - ds), min(255, s + ds)
    v_lo, v_hi = max(0, v - dv), min(255, v + dv)

    if dh >= 90:
        h_mask = np.ones(hsv.shape[:2], dtype=bool)
    else:
        h_lo = (h - dh) % 180
        h_hi = (h + dh) % 180
        if h_lo <= h_hi:
            lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
            h_mask = np.asarray(cv2.inRange(hsv, lower, upper)).astype(bool)
        else:
            l1 = np.array([0, s_lo, v_lo], dtype=np.uint8)
            u1 = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
            l2 = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            u2 = np.array([179, s_hi, v_hi], dtype=np.uint8)
            m1 = np.asarray(cv2.inRange(hsv, l1, u1)).astype(bool)
            m2 = np.asarray(cv2.inRange(hsv, l2, u2)).astype(bool)
            h_mask = m1 | m2

    sv_mask = (
        (hsv[..., 1] >= s_lo) & (hsv[..., 1] <= s_hi)
        & (hsv[..., 2] >= v_lo) & (hsv[..., 2] <= v_hi)
    )
    return h_mask & sv_mask


def mask_overlay_rgba(mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0), alpha: int = 140) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[mask, 0] = color[0]
    out[mask, 1] = color[1]
    out[mask, 2] = color[2]
    out[mask, 3] = alpha
    return out
