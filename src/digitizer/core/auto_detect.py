"""Automated plot detection using morphological analysis (OpenCV only).

Public API (consumed by main_window.py):
  PlotBox                       — dataclass
  detect_plot_box(img)          — returns PlotBox
  detect_curve_colors(img, box) — returns list[(R,G,B)]

Tesseract OCR axis-label reading lives on the `feature/ocr` branch, not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlotBox:
    """Pixel-coordinate bounding box of the plot data region."""
    x: int
    y: int
    w: int
    h: int
    origin_x: int   # Pixel X of the axis intersection (bottom-left of data)
    origin_y: int   # Pixel Y of the axis intersection (bottom-left of data)


# ---------------------------------------------------------------------------
# Internal: ChartRegionSegmenter
# ---------------------------------------------------------------------------

class _Seg:
    """Internal segmenter operating on a BGR numpy array."""

    def __init__(self, img: np.ndarray):
        self.img = img
        self.h_img, self.w_img = img.shape[:2]
        self.img_area = self.h_img * self.w_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        self.plot_box = None
        self.legend_box = None
        self.has_grid = False
        self.has_legend = False

    def extract_structure(self):
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (self.w_img // 30, 1))
        self.horizontal = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel_h)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.h_img // 30))
        self.vertical = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel_v)
        self.axes_img = cv2.add(self.horizontal, self.vertical)
        self.text_mask = cv2.subtract(self.thresh, self.axes_img)

    @staticmethod
    def _get_cluster_centers(indices):
        if len(indices) == 0: return []
        clusters = np.split(indices, np.where(np.diff(indices) > 5)[0] + 1)
        return [int(np.mean(c)) for c in clusters]

    def locate_plot_box_and_grid(self):
        kernel_close = np.ones((5, 5), np.uint8)
        axes_dilated = cv2.dilate(self.axes_img, kernel_close, iterations=2)
        contours, _ = cv2.findContours(axes_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_boxes = [cv2.boundingRect(c) for c in contours
                       if cv2.boundingRect(c)[2] > self.w_img * 0.05
                       and cv2.boundingRect(c)[3] > self.h_img * 0.05]

        if not valid_boxes:
            self.plot_box = (0, 0, self.w_img, self.h_img)
            self.has_grid = False
            return

        valid_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        rough_box = valid_boxes[0]

        gx, gy, gw, gh = rough_box
        box_h = self.horizontal[gy:gy+gh, gx:gx+gw]
        box_v = self.vertical[gy:gy+gh, gx:gx+gw]
        h_proj = np.sum(box_h, axis=1)
        v_proj = np.sum(box_v, axis=0)
        h_peaks = np.where(h_proj > (gw * 0.3 * 255))[0]
        v_peaks = np.where(v_proj > (gh * 0.3 * 255))[0]

        h_centers = self._get_cluster_centers(h_peaks)
        v_centers = self._get_cluster_centers(v_peaks)

        exact_gy, exact_gh = gy, gh
        if h_centers:
            exact_origin_y = gy + h_centers[-1]
            exact_top_y = gy + h_centers[0] if len(h_centers) > 1 else gy
            exact_gy = exact_top_y
            exact_gh = exact_origin_y - exact_top_y

        exact_gx, exact_gw = gx, gw
        if v_centers:
            exact_origin_x = gx + v_centers[0]
            exact_right_x = gx + v_centers[-1] if len(v_centers) > 1 else gx + gw
            exact_gx = exact_origin_x
            exact_gw = exact_right_x - exact_origin_x

        self.plot_box = (exact_gx, exact_gy, exact_gw, exact_gh)
        gx, gy, gw, gh = self.plot_box

        plot_h = self.horizontal[gy:gy+gh, gx:gx+gw]
        plot_v = self.vertical[gy:gy+gh, gx:gx+gw]
        hgp, _ = find_peaks(np.sum(plot_h, axis=1) / 255, height=gw * 0.5, distance=10)
        vgp, _ = find_peaks(np.sum(plot_v, axis=0) / 255, height=gh * 0.5, distance=10)
        self.has_grid = len(hgp) > 2 and len(vgp) > 2

        for box in valid_boxes[1:]:
            bx, by, bw, bh = box
            box_area = bw * bh
            if box_area > self.img_area * 0.01 and box_area < self.img_area * 0.25:
                text_roi = self.text_mask[by:by+bh, bx:bx+bw]
                if (np.count_nonzero(text_roi) / box_area) > 0.02:
                    self.legend_box = box
                    self.has_legend = True
                    break


# ---------------------------------------------------------------------------
# 1. detect_plot_box
# ---------------------------------------------------------------------------

def detect_plot_box(img: np.ndarray) -> PlotBox:
    """Detect the plot data region using morphological analysis + spine correction."""
    seg = _Seg(img)
    seg.extract_structure()
    seg.locate_plot_box_and_grid()
    gx, gy, gw, gh = seg.plot_box
    return PlotBox(x=gx, y=gy, w=gw, h=gh, origin_x=gx, origin_y=gy + gh)

# Backward-compat alias
detect_plot_box_robust = detect_plot_box


# ---------------------------------------------------------------------------
# 2. detect_curve_colors (Frequency-Based HSV)
# ---------------------------------------------------------------------------

def detect_curve_colors(img: np.ndarray, box: PlotBox, max_colors: int = 6) -> list[tuple[int, int, int]]:
    """Detect dominant curve colors inside the plot region. Returns (R,G,B) tuples."""
    data_img = img[box.y:box.y + box.h, box.x:box.x + box.w]
    if data_img.size == 0: return []

    hsv = cv2.cvtColor(data_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    valid_pixels = data_img[mask > 0]
    if len(valid_pixels) < 100: return []

    pixels32 = np.float32(valid_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels32, max_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    unique, counts = np.unique(labels, return_counts=True)
    colors: list[tuple[int, int, int]] = []
    for c, count in zip(centers, counts):
        if count / len(valid_pixels) > 0.05:
            colors.append((int(c[2]), int(c[1]), int(c[0])))

    return colors
