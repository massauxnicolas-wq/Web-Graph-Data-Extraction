"""Automated plot detection using morphological analysis and Tesseract OCR.

Public API (consumed by main_window.py):
  PlotBox, AxisInfo          — dataclasses
  detect_plot_box(img)       — returns PlotBox
  detect_axis_labels(img, box) — returns AxisInfo
  detect_curve_colors(img, box) — returns list[(R,G,B)]
  get_ocr_reader()           — no-op (Tesseract needs no persistent reader)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import re
import pytesseract
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Public dataclasses (unchanged)
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


@dataclass
class AxisInfo:
    """Detected axis values and labels."""
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    x_label: str = ""
    y_label: str = ""
    title: str = ""
    legend: str = ""
    x_ticks_text: str = ""
    y_ticks_text: str = ""
    # Pixel positions of the boundary tick marks (used for calibration mapping)
    x_min_pixel: float | None = None
    x_max_pixel: float | None = None
    y_min_pixel: float | None = None   # Bottom Y-tick (largest pixel Y)
    y_max_pixel: float | None = None   # Top Y-tick (smallest pixel Y)


def get_ocr_reader():
    """No-op for API compatibility. Tesseract requires no persistent reader."""
    return None


# ---------------------------------------------------------------------------
# Internal: ChartRegionSegmenter (from new_axis_pipeline.py — verbatim)
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
        self.title_box = None
        self.has_grid = False
        self.has_legend = False
        self.has_title = False
        self.regions = {}

    def extract_structure(self):
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (self.w_img // 30, 1))
        self.horizontal = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel_h)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.h_img // 30))
        self.vertical = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel_v)
        self.axes_img = cv2.add(self.horizontal, self.vertical)
        self.text_mask = cv2.subtract(self.thresh, self.axes_img)
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.text_mask_dilated = cv2.dilate(self.text_mask, kernel_text, iterations=1)

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

    def detect_title(self):
        gx, gy, gw, gh = self.plot_box
        top_region = self.text_mask_dilated[0:max(0, gy - 5), 0:self.w_img]
        if top_region.size > 0:
            row_proj = np.sum(top_region, axis=1)
            nz = np.where(row_proj > 0)[0]
            if len(nz) > 0:
                clusters = np.split(nz, np.where(np.diff(nz) > 5)[0] + 1)
                for cluster in reversed(clusters):
                    t_s, t_e = cluster[0], cluster[-1]
                    if t_e - t_s < 10: continue
                    band = top_region[t_s:t_e, :]
                    cols = np.where(np.sum(band, axis=0) > 0)[0]
                    if len(cols) > 0:
                        c_s, c_e = cols[0], cols[-1]
                        if c_e - c_s < 30: continue
                        area = (c_e - c_s) * (t_e - t_s)
                        roi = self.text_mask[t_s:t_e, c_s:c_e]
                        if (np.count_nonzero(roi) / area) > 0.05:
                            self.title_box = (c_s, t_s, c_e - c_s, t_e - t_s)
                            self.has_title = True
                            return

    def isolate_ocr_regions(self):
        gx, gy, gw, gh = self.plot_box
        safe_top_y = (self.title_box[1] + self.title_box[3] + 5) if self.has_title else 0

        # Measure tick bridge lengths
        sb = gy + gh
        x_band = self.thresh[sb:min(self.h_img, sb + 40), gx:gx+gw]
        x_bridge = 0
        if x_band.size > 0:
            for i, val in enumerate(np.sum(x_band, axis=1)):
                if val > 0: x_bridge = i + 1
                elif x_bridge > 0: break

        sl = gx
        y_band = self.thresh[gy:gy+gh, max(0, sl - 40):sl]
        y_bridge = 0
        if y_band.size > 0:
            cp = np.sum(y_band, axis=0)
            bw = len(cp)
            for i in range(bw - 1, -1, -1):
                if cp[i] > 0: y_bridge = bw - i
                elif y_bridge > 0: break

        safe_x = min(30, x_bridge + 2)
        safe_y = min(30, y_bridge + 2)

        # X-Axis
        x_rs = gy + gh + safe_x
        x_region = self.text_mask_dilated[x_rs:self.h_img, gx:gx+gw]
        x_ticks_box, x_label_box = None, None
        if x_region.size > 0:
            rp = np.sum(x_region, axis=1)
            nz = np.where(rp > 0)[0]
            if len(nz) > 0:
                cl = np.split(nz, np.where(np.diff(nz) > 5)[0] + 1)
                if len(cl) >= 1:
                    ts, te = cl[0][0], cl[0][-1]
                    x_ticks_box = (0, x_rs + ts - 3, self.w_img, te - ts + 6)
                if len(cl) >= 2:
                    ls, le = cl[-1][0], cl[-1][-1]
                    x_label_box = (0, x_rs + ls, self.w_img, le - ls)

        # Y-Axis
        y_re = gx - safe_y
        y_region = self.text_mask_dilated[safe_top_y:self.h_img, 0:max(0, y_re)]
        y_ticks_box, y_label_box = None, None
        if y_region.size > 0 and y_re > 0:
            cp = np.sum(y_region, axis=0)
            nz = np.where(cp > 0)[0]
            if len(nz) > 0:
                cl = np.split(nz, np.where(np.diff(nz) > 5)[0] + 1)
                if len(cl) >= 1:
                    ts, te = cl[-1][0], cl[-1][-1]
                    y_ticks_box = (ts, safe_top_y, te - ts, x_rs - safe_top_y)
                if len(cl) >= 2:
                    ls, le = cl[0][0], cl[0][-1]
                    y_label_box = (ls, safe_top_y, le - ls, x_rs - safe_top_y)

        self.regions = {
            "x_ticks": x_ticks_box, "x_label": x_label_box,
            "y_ticks": y_ticks_box, "y_label": y_label_box,
        }


# ---------------------------------------------------------------------------
# Internal: Tesseract OCR helpers
# ---------------------------------------------------------------------------

def _ocr_text(img, box, psm=7, rotate=None):
    """Crop, optionally rotate, return Tesseract text."""
    if box is None: return ""
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    crop = img[y:y+h, x:x+w]
    if crop.size == 0: return ""
    if rotate is not None:
        crop = cv2.rotate(crop, rotate)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config=f"--psm {psm} -l eng").strip().replace('\n', ' ')


def _ocr_ticks(img, box, psm=6):
    """Crop, run Tesseract image_to_data, return list of (pixel_center, value)."""
    if box is None: return []
    bx, by, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    crop = img[by:by+bh, bx:bx+bw]
    if crop.size == 0: return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, config=f"--psm {psm} -l eng",
                                     output_type=pytesseract.Output.DICT)
    results = []
    for i, text in enumerate(data['text']):
        text = text.strip().replace(',', '.').replace('\u2212', '-')
        if not text: continue
        if re.match(r'^-?\d+(\.\d+)?$', text):
            try:
                val = float(text)
                cx = data['left'][i] + data['width'][i] / 2.0
                cy = data['top'][i] + data['height'][i] / 2.0
                results.append((cx, cy, val))
            except ValueError:
                continue
    return results


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
# 2. detect_axis_labels
# ---------------------------------------------------------------------------

def detect_axis_labels(img: np.ndarray, box: PlotBox, reader=None) -> AxisInfo:
    """Detect axis tick values and labels using Tesseract OCR."""
    seg = _Seg(img)
    seg.extract_structure()
    seg.plot_box = (box.x, box.y, box.w, box.h)
    seg.detect_title()
    seg.isolate_ocr_regions()

    info = AxisInfo()

    # X-ticks
    xtb = seg.regions.get("x_ticks")
    if xtb:
        ticks = _ocr_ticks(img, xtb, psm=6)
        if ticks:
            # cx is in crop coords; offset by crop X origin
            mapped = [(cx + xtb[0], val) for cx, cy, val in ticks]
            mapped.sort(key=lambda t: t[0])
            info.x_min, info.x_min_pixel = mapped[0][1], mapped[0][0]
            info.x_max, info.x_max_pixel = mapped[-1][1], mapped[-1][0]

    # Y-ticks (not rotated — numbers are horizontal)
    ytb = seg.regions.get("y_ticks")
    if ytb:
        ticks = _ocr_ticks(img, ytb, psm=6)
        if ticks:
            # cy is in crop coords; offset by crop Y origin
            mapped = [(cy + ytb[1], val) for cx, cy, val in ticks]
            mapped.sort(key=lambda t: t[0])  # ascending pixel Y (top first)
            info.y_max, info.y_max_pixel = mapped[0][1], mapped[0][0]
            info.y_min, info.y_min_pixel = mapped[-1][1], mapped[-1][0]

    # Labels and text regions
    info.x_label = _ocr_text(img, seg.regions.get("x_label"), psm=7)
    info.y_label = _ocr_text(img, seg.regions.get("y_label"), psm=7,
                             rotate=cv2.ROTATE_90_CLOCKWISE)
    info.title = _ocr_text(img, seg.title_box, psm=7) if seg.has_title else ""
    info.legend = _ocr_text(img, seg.legend_box, psm=6) if seg.has_legend else ""
    info.x_ticks_text = _ocr_text(img, seg.regions.get("x_ticks"), psm=6)
    info.y_ticks_text = _ocr_text(img, seg.regions.get("y_ticks"), psm=6)

    return info


# ---------------------------------------------------------------------------
# 3. detect_curve_colors (Frequency-Based HSV — from new_axis_pipeline.py)
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


# ---------------------------------------------------------------------------
# 4. Debug overlay (visual diagnostic)
# ---------------------------------------------------------------------------

def generate_debug_overlay(img: np.ndarray, box: PlotBox) -> np.ndarray:
    """Draw detected regions on the image + OCR summary panel below. Returns RGB."""
    seg = _Seg(img)
    seg.extract_structure()
    seg.locate_plot_box_and_grid()          # finds legend_box too
    seg.plot_box = (box.x, box.y, box.w, box.h)  # use caller's refined box
    seg.detect_title()
    seg.isolate_ocr_regions()

    out = img.copy()
    h_img, w_img = out.shape[:2]

    # --- Color map for region boxes ---
    CMAP = {
        "plot_area": ((0, 255, 0), 3),
        "title":     ((255, 255, 0), 2),
        "legend":    ((0, 165, 255), 2),
        "x_ticks":   ((255, 0, 255), 2),
        "x_label":   ((0, 255, 255), 2),
        "y_ticks":   ((255, 0, 255), 2),
        "y_label":   ((0, 255, 255), 2),
    }

    # Collect all regions
    all_r = dict(seg.regions)
    all_r["plot_area"] = seg.plot_box
    if seg.title_box:  all_r["title"] = seg.title_box
    if seg.legend_box: all_r["legend"] = seg.legend_box

    # Draw boxes + labels on the image
    for name, region in all_r.items():
        if region is None: continue
        rx, ry, rw, rh = int(region[0]), int(region[1]), int(region[2]), int(region[3])
        color, thick = CMAP.get(name, ((255, 255, 255), 1))
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color, thick)
        cv2.putText(out, name, (rx + 4, max(14, ry - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # --- Run OCR on all regions ---
    ocr = {}
    ocr["title"]   = _ocr_text(img, all_r.get("title"), psm=7)
    ocr["x_ticks"] = _ocr_text(img, seg.regions.get("x_ticks"), psm=6)
    ocr["x_label"] = _ocr_text(img, seg.regions.get("x_label"), psm=7)
    ocr["y_ticks"] = _ocr_text(img, seg.regions.get("y_ticks"), psm=6)
    ocr["y_label"] = _ocr_text(img, seg.regions.get("y_label"), psm=7,
                                rotate=cv2.ROTATE_90_CLOCKWISE)
    ocr["legend"]  = _ocr_text(img, all_r.get("legend"), psm=6)

    # Grid / legend flags
    ocr["has_grid"]   = "Yes" if seg.has_grid else "No"
    ocr["has_legend"] = "Yes" if seg.has_legend else "No"

    # --- Build summary panel below the image ---
    lines = [
        f"TITLE:    {ocr['title'] or '(not detected)'}",
        f"X-LABEL:  {ocr['x_label'] or '(not detected)'}",
        f"Y-LABEL:  {ocr['y_label'] or '(not detected)'}",
        f"X-TICKS:  {ocr['x_ticks'] or '(not detected)'}",
        f"Y-TICKS:  {ocr['y_ticks'] or '(not detected)'}",
        f"LEGEND:   {ocr['legend'] or '(not detected)'}",
        f"GRID:     {ocr['has_grid']}   |   LEGEND BOX: {ocr['has_legend']}",
    ]

    line_h = 22
    pad = 12
    panel_h = pad * 2 + line_h * len(lines)
    panel = np.zeros((panel_h, w_img, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # dark background

    for i, line in enumerate(lines):
        y = pad + (i + 1) * line_h
        # White outline for readability
        cv2.putText(panel, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Stack image + panel vertically
    combined = np.vstack([out, panel])
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
