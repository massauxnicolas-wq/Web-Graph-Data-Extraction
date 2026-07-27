import sys
import os
import cv2
import numpy as np
import easyocr
import re

DEBUG_DIR = "debug_crops"


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0 — Grid & Legend Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_axes(image_path: str):
    print(f"Loading {image_path} for axis detection...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 40, 200]))

    filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered[gray_mask == 0] = 255
    filtered_inv = cv2.bitwise_not(filtered)
    _, thresh = cv2.threshold(filtered_inv, 30, 255, cv2.THRESH_BINARY)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 20, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 20))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

    axes_img = cv2.add(horizontal, vertical)
    kernel_close = np.ones((5, 5), np.uint8)
    axes_img_dilated = cv2.dilate(axes_img, kernel_close, iterations=1)

    contours, _ = cv2.findContours(axes_img_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > w_img * 0.05 and h > h_img * 0.05 and w < w_img * 0.99 and h < h_img * 0.99:
            valid_boxes.append((x, y, w, h))

    if not valid_boxes:
        return (0, 0, w_img, h_img), None, 0, 0

    unique_boxes = []
    for box in valid_boxes:
        bx, by, bw, bh = box
        is_dup = any(abs(bx-ux)<20 and abs(by-uy)<20 and abs(bw-uw)<20 for ux, uy, uw, uh in unique_boxes)
        if not is_dup:
            unique_boxes.append(box)

    unique_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    plot_box = unique_boxes[0]
    legend_box = None

    if len(unique_boxes) > 1:
        cand = unique_boxes[1]
        area = cand[2] * cand[3]
        if area > img_area * 0.01:
            legend_box = cand
            print(f"Legend detected: X={cand[0]}, Y={cand[1]}, W={cand[2]}, H={cand[3]}")

    # --- Spine Correction ---
    gx, gy, gw, gh = plot_box
    box_h = horizontal[gy:gy+gh, gx:gx+gw]
    box_v = vertical[gy:gy+gh, gx:gx+gw]
    h_proj = np.sum(box_h, axis=1)
    v_proj = np.sum(box_v, axis=0)
    h_peaks = np.where(h_proj > (gw * 0.3 * 255))[0]
    v_peaks = np.where(v_proj > (gh * 0.3 * 255))[0]

    def get_cluster_centers(indices):
        if len(indices) == 0: return []
        clusters = np.split(indices, np.where(np.diff(indices) > 5)[0] + 1)
        return [int(np.mean(c)) for c in clusters]

    h_centers = get_cluster_centers(h_peaks)
    v_centers = get_cluster_centers(v_peaks)

    if h_centers:
        exact_origin_y = gy + h_centers[-1]
        exact_top_y = gy + h_centers[0] if len(h_centers) > 1 else gy
        plot_box = (gx, exact_top_y, gw, exact_origin_y - exact_top_y)
    if v_centers:
        exact_origin_x = gx + v_centers[0]
        exact_right_x = gx + v_centers[-1] if len(v_centers) > 1 else gx + gw
        plot_box = (exact_origin_x, plot_box[1], exact_right_x - exact_origin_x, plot_box[3])

    # --- Measure tick mark protrusion from spines ---
    gx, gy, gw, gh = plot_box

    # X-axis: scan downward from bottom spine for vertical tick marks
    sb = gy + gh
    x_band = thresh[sb:min(h_img, sb + 40), gx:gx+gw]
    x_tick_len = 0
    if x_band.size > 0:
        row_proj = np.sum(x_band, axis=1)
        for i, val in enumerate(row_proj):
            if val > 0:
                x_tick_len = i + 1
            elif x_tick_len > 0:
                break

    # Y-axis: scan leftward from left spine for horizontal tick marks
    sl = gx
    y_band = thresh[gy:gy+gh, max(0, sl - 40):sl]
    y_tick_len = 0
    if y_band.size > 0:
        col_proj = np.sum(y_band, axis=0)
        bw = len(col_proj)
        for i in range(bw - 1, -1, -1):
            if col_proj[i] > 0:
                y_tick_len = bw - i
            elif y_tick_len > 0:
                break

    os.makedirs(DEBUG_DIR, exist_ok=True)
    if x_band.size > 0: cv2.imwrite(f"{DEBUG_DIR}/tick_band_x.png", x_band)
    if y_band.size > 0: cv2.imwrite(f"{DEBUG_DIR}/tick_band_y.png", y_band)
    print(f"Tick mark length: X={x_tick_len}px below spine, Y={y_tick_len}px left of spine")

    return plot_box, legend_box, x_tick_len, y_tick_len


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 & 2 — OCR: whole-strip tick + label detection
# ═══════════════════════════════════════════════════════════════════════════

def _parse_ocr(results, min_prob=0.1):
    parsed = []
    for res in results:
        if len(res) == 3:
            bbox, text, prob = res
            if prob < min_prob: continue
        else:
            bbox, text = res
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        tx, ty = int(min(xs)), int(min(ys))
        tw, th = int(max(xs) - min(xs)), int(max(ys) - min(ys))
        cx, cy = tx + tw / 2.0, ty + th / 2.0
        parsed.append((cx, cy, text.strip(), (tx, ty, tw, th)))
    return parsed


def extract_metadata(image_path: str, plot_box: tuple, legend_box,
                     x_tick_len: int = 0, y_tick_len: int = 0):
    print("\nInitializing EasyOCR...")
    reader = easyocr.Reader(['en'])
    img = cv2.imread(image_path)
    if img is None: return
    h_img, w_img = img.shape[:2]
    px, py, pw, ph = plot_box
    os.makedirs(DEBUG_DIR, exist_ok=True)

    debug_img = img.copy()
    x_ticks, y_ticks = [], []
    x_title, y_title = "Unknown", "Unknown"
    legend_entries = []

    cv2.rectangle(debug_img, (px, py), (px + pw, py + ph), (0, 255, 0), 3)
    if legend_box:
        lx, ly, lw, lh = legend_box
        cv2.rectangle(debug_img, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 3)

    # ── X-Axis: strip below plot, split tick row / label row via projection ──
    x_crop_top = py + ph + max(x_tick_len, 8) + 2
    x_crop_left = max(0, px - 50)
    x_strip = img[x_crop_top:h_img, x_crop_left:w_img]
    if x_strip.size > 0:
        cv2.imwrite(f"{DEBUG_DIR}/x_strip_full.png", x_strip)

        # Row projection to find gap between tick numbers and axis label
        gray_x = cv2.cvtColor(x_strip, cv2.COLOR_BGR2GRAY)
        _, bin_x = cv2.threshold(gray_x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_proj = np.sum(bin_x, axis=1) / 255
        content_rows = row_proj > bin_x.shape[1] * 0.005
        x_tick_end = len(content_rows)
        gap = 0
        for i, has in enumerate(content_rows):
            if not has:
                gap += 1
                if gap >= 3:
                    x_tick_end = i - gap + 1
                    break
            else:
                gap = 0

        x_tick_crop = x_strip[0:x_tick_end, :]
        x_label_crop = x_strip[x_tick_end:, :]
        if x_tick_crop.size == 0:
            x_tick_crop = x_strip
            x_tick_end = x_strip.shape[0]
        if x_tick_crop.size > 0:
            cv2.imwrite(f"{DEBUG_DIR}/x_tick_crop.png", x_tick_crop)
        if x_label_crop.size > 0:
            cv2.imwrite(f"{DEBUG_DIR}/x_label_crop.png", x_label_crop)
        print(f"  X-axis split: ticks=0:{x_tick_end}px, label={x_tick_end}:end")

        # Tick OCR on tick-only crop (no label text to confuse)
        print("Scanning X-axis ticks...")
        res_n = reader.readtext(x_tick_crop, decoder='beamsearch', beamWidth=10,
                                mag_ratio=1.5, text_threshold=0.5, low_text=0.3)
        for cx, cy, text, rect in _parse_ocr(res_n, min_prob=0.05):
            val_text = text.replace(',', '.').replace('O', '0').replace('o', '0')
            if re.match(r'^-?\d+(\.\d+)?$', val_text):
                try:
                    val = float(val_text)
                    x_ticks.append(val)
                    ox, oy = rect[0] + x_crop_left, rect[1] + x_crop_top
                    cv2.rectangle(debug_img, (ox, oy),
                                  (ox + rect[2], oy + rect[3]), (255, 0, 255), 2)
                except:
                    continue
            print(f"  X OCR: '{text}' @ ({rect[0]},{rect[1]}) size={rect[2]}x{rect[3]}")

        # Label OCR on label-only crop
        if x_label_crop.size > 100:
            res_p = reader.readtext(x_label_crop, paragraph=True)
            parsed = _parse_ocr(res_p)
            if parsed:
                x_title = " ".join([p[2] for p in parsed])
                rect = parsed[0][3]
                ox = rect[0] + x_crop_left
                oy = rect[1] + x_crop_top + x_tick_end
                cv2.rectangle(debug_img, (ox, oy),
                              (ox + rect[2], oy + rect[3]), (0, 255, 255), 2)
            print(f"  X-label: '{x_title}'")

    # ── Y-Axis: tight horizontal cut (tick_len), tall vertical (top of image → top of X tick labels) ──
    y_crop_right = px - max(y_tick_len, 4) - 2
    y_crop_bottom = x_crop_top + 5
    y_strip = img[0:y_crop_bottom, 0:max(1, y_crop_right)]
    if y_strip.size > 0:
        cv2.imwrite(f"{DEBUG_DIR}/y_strip_full.png", y_strip)

        # Column projection to find gap between label (left) and tick numbers (right)
        gray_y = cv2.cvtColor(y_strip, cv2.COLOR_BGR2GRAY)
        _, bin_y = cv2.threshold(gray_y, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        col_proj = np.sum(bin_y, axis=0) / 255
        content_cols = col_proj > bin_y.shape[0] * 0.005
        y_tick_start = 0
        gap = 0
        for i in range(len(content_cols) - 1, -1, -1):
            if not content_cols[i]:
                gap += 1
                if gap >= 3:
                    y_tick_start = i + gap
                    break
            else:
                gap = 0

        y_tick_crop = y_strip[:, y_tick_start:]
        y_label_crop = y_strip[:, 0:max(1, y_tick_start)]
        if y_tick_crop.size == 0:
            y_tick_crop = y_strip
            y_tick_start = 0
        if y_tick_crop.size > 0:
            cv2.imwrite(f"{DEBUG_DIR}/y_tick_crop.png", y_tick_crop)
        if y_label_crop.size > 0:
            cv2.imwrite(f"{DEBUG_DIR}/y_label_crop.png", y_label_crop)
        print(f"  Y-axis split: label=0:{y_tick_start}px, ticks={y_tick_start}:end")

        # Tick OCR on tick-only crop
        print("Scanning Y-axis ticks...")
        res_n = reader.readtext(y_tick_crop, decoder='beamsearch', beamWidth=10,
                                mag_ratio=1.5, text_threshold=0.5, low_text=0.3)
        for cx, cy, text, rect in _parse_ocr(res_n, min_prob=0.05):
            val_text = text.replace(',', '.').replace('O', '0').replace('o', '0')
            if re.match(r'^-?\d+(\.\d+)?$', val_text):
                try:
                    val = float(val_text)
                    y_ticks.append(val)
                    ox, oy = rect[0] + y_tick_start, rect[1]
                    cv2.rectangle(debug_img, (ox, oy),
                                  (ox + rect[2], oy + rect[3]), (255, 0, 255), 2)
                except:
                    continue
            print(f"  Y OCR: '{text}' @ ({rect[0]},{rect[1]}) size={rect[2]}x{rect[3]}")

        if y_label_crop.size > 100:
            lh, lw = y_label_crop.shape[:2]
            best_text, best_score = "Unknown", -1.0
            best_bboxes, best_rot = [], None
            for rot_code in (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE):
                rot_name = "CCW" if rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE else "CW"
                rot = cv2.rotate(y_label_crop, rot_code)
                cv2.imwrite(f"{DEBUG_DIR}/y_label_rot_{rot_name}.png", rot)
                res_rot = reader.readtext(rot, paragraph=False)
                parts = []
                for r in res_rot:
                    txt = r[1].strip()
                    conf = r[2] if len(r) == 3 else 0.5
                    print(f"    {rot_name} fragment: '{txt}' conf={conf:.2f}")
                    if conf >= 0.3 and txt:
                        parts.append(r)
                if not parts:
                    continue
                parts.sort(key=lambda r: min(pt[0] for pt in r[0]))
                score = sum((r[2] if len(r) == 3 else 0.5) for r in parts) / len(parts)
                text = " ".join(r[1].strip() for r in parts).strip()
                print(f"  Y-label {rot_name}: '{text}' conf={score:.2f}")
                if text and score > best_score:
                    best_text, best_score = text, score
                    best_bboxes = [r[0] for r in parts]
                    best_rot = rot_code

            if best_score > 0:
                y_title = best_text
                # Map bboxes from rotated label crop back to original image
                all_pts = [pt for bbox in best_bboxes for pt in bbox]
                rx1 = min(p[0] for p in all_pts)
                ry1 = min(p[1] for p in all_pts)
                rx2 = max(p[0] for p in all_pts)
                ry2 = max(p[1] for p in all_pts)
                if best_rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    bx1, by1 = int(lw - 1 - ry2), int(rx1)
                    bx2, by2 = int(lw - 1 - ry1), int(rx2)
                else:
                    bx1, by1 = int(ry1), int(lh - 1 - rx2)
                    bx2, by2 = int(ry2), int(lh - 1 - rx1)
                cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                print(f"  Detected Y-Axis Title: '{y_title}' (conf={best_score:.2f})")

    # ── Legend ──
    if legend_box:
        lx, ly, lw, lh = legend_box
        l_crop = img[ly:ly+lh, lx:lx+lw]
        if l_crop.size > 0:
            cv2.imwrite(f"{DEBUG_DIR}/legend_crop.png", l_crop)
            print("Mapping Legend Content...")
            l_res = reader.readtext(l_crop, paragraph=False)
            hsv_l = cv2.cvtColor(l_crop, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_l, np.array([0, 30, 50]), np.array([180, 255, 255]))
            cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            markers = []
            for c in cnts:
                mx, my, mw, mh = cv2.boundingRect(c)
                if mw * mh > 10:
                    markers.append((mx + mw / 2, my + mh / 2, (mx, my, mw, mh)))

            for cx, cy, text, rect in _parse_ocr(l_res):
                cv2.rectangle(debug_img, (rect[0]+lx, rect[1]+ly),
                              (rect[0]+rect[2]+lx, rect[1]+rect[3]+ly), (255, 255, 0), 2)
                if markers:
                    closest = min(markers, key=lambda m: abs(m[1] - cy))
                    mx, my, mw, mh = closest[2]
                    cv2.circle(debug_img, (int(mx+lx+mw/2), int(my+ly+mh/2)), 5, (0, 255, 255), -1)
                    legend_entries.append(f"'{text}' (Marker at X={mx+lx:.0f}, Y={my+ly:.0f})")

    # ── Summary ──
    print("\n" + "=" * 45)
    print("        AUTO-DETECTION SUMMARY")
    print("=" * 45)
    if x_ticks:
        print(f"X-Bounds: {min(x_ticks)} to {max(x_ticks)}")
    print(f"X-Axis Label: {x_title}")
    if y_ticks:
        print(f"Y-Bounds: {min(y_ticks)} to {max(y_ticks)}")
    print(f"Y-Axis Label: {y_title}")
    if legend_entries:
        print("\nLegend Properties:")
        for entry in legend_entries:
            print(f"  - {entry}")
    else:
        print("\nNo structured legend detected.")
    print("=" * 45)

    cv2.imwrite("ocr_debug.png", debug_img)
    print("\nSaved visual dashboard to ocr_debug.png")


def detect_curve_colors(image_path: str, plot_box: tuple):
    img = cv2.imread(image_path)
    if img is None: return []
    px, py, pw, ph = plot_box
    data_img = img[py:py + ph, px:px + pw]
    if data_img.size == 0: return []

    hsv = cv2.cvtColor(data_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    valid_pixels = data_img[mask > 0]
    if len(valid_pixels) < 100: return []

    pixels32 = np.float32(valid_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels32, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    unique, counts = np.unique(labels, return_counts=True)
    colors = []
    for c, count in zip(centers, counts):
        if count / len(valid_pixels) > 0.05:
            colors.append((int(c[2]), int(c[1]), int(c[0])))

    print("\n--- Detected Curve Colors (RGB) ---")
    for r, g, b in colors: print(f"  RGB({r}, {g}, {b})")
    return colors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_detect.py <path_to_image>")
        sys.exit(1)

    img_p = sys.argv[1]
    p_box, l_box, xtl, ytl = detect_axes(img_p)
    extract_metadata(img_p, p_box, l_box, xtl, ytl)
    detect_curve_colors(img_p, p_box)
