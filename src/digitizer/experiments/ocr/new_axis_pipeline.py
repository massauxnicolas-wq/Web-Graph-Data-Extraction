#!/usr/bin/env python3
import cv2
import numpy as np
import os
import argparse
import sys
from scipy.signal import find_peaks
import pytesseract

class ChartRegionSegmenter:
    def __init__(self, image_path: str, debug_dir: str = "debug_output"):
        self.image_path = image_path
        self.debug_dir = debug_dir
        self.img = cv2.imread(image_path)
        
        if self.img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        self.h_img, self.w_img = self.img.shape[:2]
        self.img_area = self.h_img * self.w_img
        
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        
        self.plot_box = None
        self.legend_box = None
        self.title_box = None
        self.has_grid = False
        self.has_legend = False
        self.has_title = False
        self.curve_colors = []
        self.regions = {}

    def extract_structure(self):
        """Isolates lines, creates structural mask, and a text-only mask."""
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (self.w_img // 30, 1))
        self.horizontal = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel_h)
        
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.h_img // 30))
        self.vertical = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel_v)

        self.axes_img = cv2.add(self.horizontal, self.vertical)
        
        self.text_mask = cv2.subtract(self.thresh, self.axes_img)
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.text_mask_dilated = cv2.dilate(self.text_mask, kernel_text, iterations=1)

    def _get_cluster_centers(self, indices):
        if len(indices) == 0: return []
        clusters = np.split(indices, np.where(np.diff(indices) > 5)[0] + 1)
        return [int(np.mean(c)) for c in clusters]

    def locate_plot_box_and_grid(self):
        """Finds plot area, corrects spines, detects grids, and isolates the legend."""
        kernel_close = np.ones((5, 5), np.uint8)
        axes_dilated = cv2.dilate(self.axes_img, kernel_close, iterations=2)
        contours, _ = cv2.findContours(axes_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_boxes = [cv2.boundingRect(c) for c in contours 
                       if cv2.boundingRect(c)[2] > self.w_img * 0.05 and cv2.boundingRect(c)[3] > self.h_img * 0.05]
        
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

        plot_h_lines = self.horizontal[gy:gy+gh, gx:gx+gw]
        plot_v_lines = self.vertical[gy:gy+gh, gx:gx+gw]
        
        h_grid_peaks, _ = find_peaks(np.sum(plot_h_lines, axis=1) / 255, height=gw * 0.5, distance=10)
        v_grid_peaks, _ = find_peaks(np.sum(plot_v_lines, axis=0) / 255, height=gh * 0.5, distance=10)
        
        self.has_grid = len(h_grid_peaks) > 2 and len(v_grid_peaks) > 2

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
        """Finds the title using strict heuristics to avoid false positives."""
        gx, gy, gw, gh = self.plot_box
        
        top_region = self.text_mask_dilated[0:max(0, gy - 5), 0:self.w_img]
        
        if top_region.size > 0:
            row_proj = np.sum(top_region, axis=1)
            non_zero_rows = np.where(row_proj > 0)[0]
            
            if len(non_zero_rows) > 0:
                clusters = np.split(non_zero_rows, np.where(np.diff(non_zero_rows) > 5)[0] + 1)
                
                for cluster in reversed(clusters):
                    t_start, t_end = cluster[0], cluster[-1]
                    c_height = t_end - t_start
                    
                    if c_height < 10:
                        continue
                        
                    title_band = top_region[t_start:t_end, :]
                    col_proj = np.sum(title_band, axis=0)
                    cols = np.where(col_proj > 0)[0]
                    
                    if len(cols) > 0:
                        c_start, c_end = cols[0], cols[-1]
                        c_width = c_end - c_start
                        
                        if c_width < 30:
                            continue
                            
                        box_area = c_width * c_height
                        text_roi = self.text_mask[t_start:t_end, c_start:c_end]
                        if (np.count_nonzero(text_roi) / box_area) > 0.05:
                            self.title_box = (c_start, t_start, c_width, c_height)
                            self.has_title = True
                            return

    def extract_curve_colors(self):
        """Uses Frequency-Based HSV Tolerance to dynamically find the number of curves."""
        gx, gy, gw, gh = self.plot_box
        plot_bgr = self.img[gy:gy+gh, gx:gx+gw]
        
        grid_mask = self.axes_img[gy:gy+gh, gx:gx+gw]
        text_mask = self.text_mask_dilated[gy:gy+gh, gx:gx+gw]
        structure_mask = cv2.bitwise_or(grid_mask, text_mask)
        
        all_pixels = plot_bgr.reshape(-1, 3)
        unique_colors, counts = np.unique(all_pixels, axis=0, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]
        
        axis_pixels = plot_bgr[grid_mask > 0]
        axis_color = bg_color 
        if len(axis_pixels) > 0:
            u_ax_colors, ax_counts = np.unique(axis_pixels, axis=0, return_counts=True)
            axis_color = u_ax_colors[np.argmax(ax_counts)]
            
        valid_pixels = plot_bgr[structure_mask == 0]
        if len(valid_pixels) == 0: return
        
        dist_to_bg = np.linalg.norm(valid_pixels - bg_color, axis=1)
        dist_to_axis = np.linalg.norm(valid_pixels - axis_color, axis=1)
        curve_candidates_bgr = valid_pixels[(dist_to_bg > 30) & (dist_to_axis > 30)]
        
        if len(curve_candidates_bgr) == 0: return
        
        candidates_reshaped = curve_candidates_bgr.reshape(-1, 1, 3).astype(np.uint8)
        curve_candidates_hsv = cv2.cvtColor(candidates_reshaped, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        
        u_hsv, hsv_counts = np.unique(curve_candidates_hsv, axis=0, return_counts=True)
        
        sorted_indices = np.argsort(hsv_counts)[::-1]
        u_hsv_sorted = u_hsv[sorted_indices]
        counts_sorted = hsv_counts[sorted_indices]
        
        total_pixels = len(curve_candidates_hsv)
        found_hsv_centers = []
        self.curve_colors = []
        
        for i, hsv_color in enumerate(u_hsv_sorted):
            if counts_sorted[i] < total_pixels * 0.005:
                break
                
            H, S, V = hsv_color
            is_new_color = True
            
            for center_hsv in found_hsv_centers:
                cH, cS, cV = center_hsv
                dH = min(abs(int(H) - int(cH)), 180 - abs(int(H) - int(cH)))
                dS = abs(int(S) - int(cS))
                dV = abs(int(V) - int(cV))
                
                if dH < 15 and dS < 50 and dV < 50:
                    is_new_color = False
                    break
                    
                if S < 40 and cS < 40 and dV < 40:
                    is_new_color = False
                    break
            
            if is_new_color:
                found_hsv_centers.append(hsv_color)
                bgr_pixel = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
                b, g, r = int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])
                self.curve_colors.append(f"#{r:02x}{g:02x}{b:02x}")
                
            if len(self.curve_colors) >= 10:
                break

    def isolate_ocr_regions(self):
        """Measures physical tick lengths, hard-crops them out, bounds using title limit."""
        gx, gy, gw, gh = self.plot_box
        
        # Safe top Y defaults to 0. Only restricted if title is detected.
        safe_top_y = (self.title_box[1] + self.title_box[3] + 5) if self.has_title else 0
        
        sb = gy + gh
        x_band = self.thresh[sb:min(self.h_img, sb + 40), gx:gx+gw]
        x_bridge_len = 0
        if x_band.size > 0:
            for i, val in enumerate(np.sum(x_band, axis=1)):
                if val > 0: x_bridge_len = i + 1
                elif x_bridge_len > 0: break

        sl = gx
        y_band = self.thresh[gy:gy+gh, max(0, sl - 40):sl]
        y_bridge_len = 0
        if y_band.size > 0:
            col_proj = np.sum(y_band, axis=0)
            bw = len(col_proj)
            for i in range(bw - 1, -1, -1):
                if col_proj[i] > 0: y_bridge_len = bw - i
                elif y_bridge_len > 0: break

        safe_x_offset = min(30, x_bridge_len + 2) 
        safe_y_offset = min(30, y_bridge_len + 2)

        # 1. X-Axis Processing
        x_region_start = gy + gh + safe_x_offset
        x_region = self.text_mask_dilated[x_region_start:self.h_img, gx:gx+gw]
        
        x_ticks_box, x_label_box = None, None
        
        if x_region.size > 0:
            row_proj = np.sum(x_region, axis=1)
            non_zero_rows = np.where(row_proj > 0)[0]
            if len(non_zero_rows) > 0:
                clusters = np.split(non_zero_rows, np.where(np.diff(non_zero_rows) > 5)[0] + 1)
                if len(clusters) >= 1:
                    t_start, t_end = clusters[0][0], clusters[0][-1]
                    x_ticks_box = (0, x_region_start + t_start - 3, self.w_img, t_end - t_start + 6)
                if len(clusters) >= 2:
                    l_start, l_end = clusters[-1][0], clusters[-1][-1]
                    x_label_box = (0, x_region_start + l_start, self.w_img, l_end - l_start)

        # 2. Y-Axis Processing
        y_region_end = gx - safe_y_offset
        # BUGFIX: Scan all the way down to self.h_img to prevent truncating long/rotated labels
        y_region = self.text_mask_dilated[safe_top_y:self.h_img, 0:max(0, y_region_end)]
        
        y_ticks_box, y_label_box = None, None
        
        if y_region.size > 0 and y_region_end > 0:
            col_proj = np.sum(y_region, axis=0)
            non_zero_cols = np.where(col_proj > 0)[0]
            if len(non_zero_cols) > 0:
                clusters = np.split(non_zero_cols, np.where(np.diff(non_zero_cols) > 5)[0] + 1)
                if len(clusters) >= 1:
                    t_start, t_end = clusters[-1][0], clusters[-1][-1]
                    y_ticks_box = (t_start, safe_top_y, t_end - t_start, x_region_start - safe_top_y)
                if len(clusters) >= 2:
                    l_start, l_end = clusters[0][0], clusters[0][-1]
                    y_label_box = (l_start, safe_top_y, l_end - l_start, x_region_start - safe_top_y)

        self.regions = {
            "plot_area": self.plot_box,
            "title": self.title_box,
            "legend": self.legend_box,
            "x_ticks": x_ticks_box,
            "x_label": x_label_box,
            "y_ticks": y_ticks_box,
            "y_label": y_label_box
        }

    def _ocr_crop(self, box, psm=6, lang="eng", rotate=None):
        """Crop region, optionally rotate, run Tesseract, return stripped text."""
        if box is None: return ""
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = self.img[y:y+h, x:x+w]
        if crop.size == 0: return ""
        if rotate is not None:
            crop = cv2.rotate(crop, rotate)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        cfg = f"--psm {psm} -l {lang}"
        return pytesseract.image_to_string(gray, config=cfg).strip().replace('\n', ' ')

    def run_ocr(self):
        """Read all detected regions with Tesseract."""
        self.ocr = {}
        self.ocr["title"]   = self._ocr_crop(self.regions["title"],   psm=7)
        self.ocr["x_ticks"] = self._ocr_crop(self.regions["x_ticks"], psm=6)
        self.ocr["x_label"] = self._ocr_crop(self.regions["x_label"], psm=7)
        self.ocr["y_ticks"] = self._ocr_crop(self.regions["y_ticks"], psm=6)
        self.ocr["y_label"] = self._ocr_crop(self.regions["y_label"], psm=7,
                                              rotate=cv2.ROTATE_90_CLOCKWISE)

    def debug_export(self):
        """Draws bounding boxes for visual verification and saves masks."""
        debug_img = self.img.copy()
        os.makedirs(self.debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        colors = {
            "plot_area": (255, 0, 0),     "title": (0, 0, 0),
            "legend": (255, 165, 0),      "x_ticks": (0, 255, 0),
            "x_label": (0, 0, 255),       "y_ticks": (0, 255, 255),
            "y_label": (255, 0, 255)
        }
        
        for name, box in self.regions.items():
            if box:
                bx, by, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(debug_img, (bx, by), (bx+bw, by+bh), colors.get(name, (255,255,255)), 2)
                cv2.putText(debug_img, name, (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(name, (255,255,255)), 1)

        cv2.imwrite(os.path.join(self.debug_dir, f"{base_name}_01_regions.png"), debug_img)
        cv2.imwrite(os.path.join(self.debug_dir, f"{base_name}_02_text_mask.png"), self.text_mask)
        cv2.imwrite(os.path.join(self.debug_dir, f"{base_name}_03_axes_mask.png"), self.axes_img)


def main():
    parser = argparse.ArgumentParser(description="Extract chart regions and metadata for OCR.")
    parser.add_argument("image_path", type=str, help="Path to input chart.")
    parser.add_argument("--debug-dir", type=str, default="debug_output", help="Debug output dir.")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: File not found -> '{args.image_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Processing: {args.image_path}")
    
    try:
        segmenter = ChartRegionSegmenter(args.image_path, args.debug_dir)
        segmenter.extract_structure()
        segmenter.locate_plot_box_and_grid()
        segmenter.detect_title()
        segmenter.extract_curve_colors()
        segmenter.isolate_ocr_regions()
        segmenter.run_ocr()
        segmenter.debug_export()
        
        print("\n--- Detection Results ---")
        print(f"{'Has Title':<15}: {segmenter.has_title}")
        print(f"{'Has Grid':<15}: {segmenter.has_grid}")
        print(f"{'Has Legend':<15}: {segmenter.has_legend}")
        print(f"{'Curve Colors':<15}: {', '.join(segmenter.curve_colors) if segmenter.curve_colors else 'None detected'}")
        print("\n--- Regions ---")
        for region, box in segmenter.regions.items():
            print(f"{region.capitalize():<15}: {str(box) if box else 'Not Found'}")
        print("\n--- OCR Results ---")
        for key, text in segmenter.ocr.items():
            print(f"{key:<15}: {repr(text)}")
        print(f"\n[*] Debug images exported to '{args.debug_dir}/'")

    except Exception as e:
        print(f"\n[!] Fatal Error during processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()