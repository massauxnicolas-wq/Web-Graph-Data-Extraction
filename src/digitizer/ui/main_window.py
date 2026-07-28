from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal, QObject
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QStatusBar,
    QToolBar,
    QWidget,
)

from digitizer.core import calibration as calib_mod
from digitizer.core import image_io, interpolate, masking, transform, xstep
from digitizer.core.auto_detect import (
    PlotBox, AxisInfo,
    detect_plot_box, detect_axis_labels, detect_curve_colors, get_ocr_reader,
    generate_debug_overlay,
)
from digitizer.io import clipboard as clip_mod
from digitizer.io import csv_export, json_export
from digitizer.ui.calibration_panel import CalibrationPanel
from digitizer.ui.curve_panel import Curve, CurvePanel
from digitizer.ui.export_dialog import ExportPanel
from digitizer.ui.image_view import ImageView


# ---------------------------------------------------------------------------
# Background worker for OCR (runs EasyOCR off the UI thread)
# ---------------------------------------------------------------------------

class _OcrWorker(QObject):
    """Runs detect_axis_labels in a background thread."""
    finished = Signal(object, object)  # (AxisInfo | None, error_msg | None)

    def __init__(self, bgr_img: np.ndarray, box: PlotBox) -> None:
        super().__init__()
        self._bgr = bgr_img
        self._box = box

    def run(self) -> None:
        try:
            info = detect_axis_labels(self._bgr, self._box)
            self.finished.emit(info, None)
        except Exception as exc:
            self.finished.emit(None, str(exc))


class Mode(Enum):
    IDLE = auto()
    CALIBRATING = auto()
    PICKING_COLOR = auto()
    SETTING_SEED = auto()
    SETTING_END = auto()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Plot Digitizer")
        self.resize(1400, 900)

        # State
        self._image_rgb: np.ndarray | None = None
        self._image_path: Path | None = None
        self._calibration_M: np.ndarray | None = None
        self._calibration_pixel_pts: list[tuple[float, float]] = []
        self._calibration_data_pts: list[tuple[float, float]] = []
        self._curves_dict: dict[int, Curve] = {}
        self._curve_id_counter = 0
        self._selected_curve_id: int | None = None
        self._mode: Mode = Mode.IDLE

        # Widgets
        self.image_view = ImageView()
        self.image_view.image_clicked.connect(self._on_image_click)

        self.calib_panel = CalibrationPanel()
        self.calib_panel.start_capture.connect(self._enter_calibration_mode)
        self.calib_panel.cancel_capture.connect(self._exit_to_idle)
        self.calib_panel.solve_requested.connect(self._solve_calibration)
        self.calib_panel.reset_requested.connect(self._reset_calibration)
        self.calib_panel.auto_calibrate_requested.connect(self._auto_calibrate)
        self.calib_panel.debug_overlay_requested.connect(self._show_debug_overlay)
        self.calib_panel.points_changed.connect(self._on_calib_points_changed)
        self.calib_panel.grid_visibility_toggled.connect(self.image_view.set_grid_visible)
        self.image_view.calib_marker_moved.connect(self._on_calib_marker_moved)

        self.curve_panel = CurvePanel()
        self.curve_panel.sample_curves_requested.connect(self._toggle_sample_curves_mode)
        self.curve_panel.hsv_changed.connect(self._on_hsv_changed)
        self.curve_panel.overlay_toggled.connect(self._on_overlay_toggled)
        self.curve_panel.extract_curve_requested.connect(self._on_extract_single)
        self.curve_panel.extract_all_requested.connect(self._on_extract_all)
        self.curve_panel.delete_curve_requested.connect(self._delete_curve)
        self.curve_panel.select_curve_changed.connect(self._on_curve_selected)
        self.curve_panel.curve_visibility_changed.connect(self._on_curve_visibility_changed)
        self.curve_panel.curve_name_changed.connect(self._on_curve_name_changed)
        self.curve_panel.exclusion_toggled.connect(self.image_view.set_exclusion_roi_visible)
        self.image_view.exclusion_thumbnail_changed.connect(self.curve_panel.set_exclusion_thumbnail)
        self.curve_panel.auto_detect_curves_requested.connect(self._auto_detect_curves)
        self.curve_panel.curve_hsv_changed.connect(self._on_curve_hsv_changed)
        self.curve_panel.curve_display_color_changed.connect(self._on_curve_display_color_changed)
        self.curve_panel.set_seed_requested.connect(self._on_set_seed_requested)
        self.curve_panel.set_end_requested.connect(self._on_set_end_requested)

        self.export_panel = ExportPanel()
        self.export_panel.export_csv_active.connect(self._export_csv_active)
        self.export_panel.export_csv_wide.connect(self._export_csv_wide)
        self.export_panel.export_csv_checked.connect(self._export_csv_checked)
        self.export_panel.export_json.connect(self._export_json)
        self.export_panel.copy_clipboard.connect(self._copy_clipboard)
        self.export_panel.refresh_requested.connect(self._refresh_export_panel)
        self.export_panel.expert_debug_requested.connect(self._show_expert_debug_plot)
        self.export_panel.points_edited.connect(self._on_curve_points_edited)

        self._tabs = QTabWidget()
        self._tabs.addTab(self.calib_panel, "1. Calibrate")
        self._tabs.addTab(self.curve_panel, "2. Curves")
        self._tabs.addTab(self.export_panel, "3. Export")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.image_view)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Toolbar
        tb = QToolBar("Main")
        self.addToolBar(tb)
        open_act = QAction("Open Image", self)
        open_act.triggered.connect(self._open_image)
        tb.addAction(open_act)

        self.setStatusBar(QStatusBar())
        self._set_mode(Mode.IDLE)

    # --- Mode management ----------------------------------------------------
    def _set_mode(self, mode: Mode) -> None:
        self._mode = mode
        self.statusBar().showMessage(f"Mode: {mode.name}")
        active = mode in (Mode.CALIBRATING, Mode.PICKING_COLOR, Mode.SETTING_SEED, Mode.SETTING_END)
        self.image_view.setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor)

    def _on_calib_points_changed(self, pts: list[tuple[float, float]]) -> None:
        xs, ys = zip(*pts) if pts else ([], [])
        self.image_view.set_calibration_markers(list(xs), list(ys), list(CalibrationPanel.LABELS))

    def _on_calib_marker_moved(self, index: int, x: float, y: float) -> None:
        pts = self.calib_panel.pixel_points()
        if index >= len(pts):
            return
        pts[index] = (x, y)
        self.calib_panel.set_pixel_points(pts)
        if self._calibration_M is not None:
            self._solve_calibration(silent=True)

    def _enter_calibration_mode(self) -> None:
        if self._image_rgb is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            self.calib_panel._capture_btn.setChecked(False)
            return
        self._set_mode(Mode.CALIBRATING)
        self.statusBar().showMessage("Calibration capture started. Click the required points on the image.")

    def _toggle_sample_curves_mode(self, on: bool) -> None:
        if on:
            if self._image_rgb is None:
                QMessageBox.warning(self, "No image", "Open an image first.")
                self.curve_panel.uncheck_sample_button()
                return
            self._set_mode(Mode.PICKING_COLOR)
            self.statusBar().showMessage("Multi-Sample Mode: Click curves to create profiles.")
        else:
            self._set_mode(Mode.IDLE)

    def _exit_to_idle(self) -> None:
        self._set_mode(Mode.IDLE)

    # --- File ---------------------------------------------------------------
    def _open_image(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open graph image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if not path_str:
            return
        try:
            rgb = image_io.load_image(path_str)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self._image_rgb = rgb
        self._image_path = Path(path_str)
        self.image_view.set_image(rgb)
        self.calib_panel.reset()
        self._reset_calibration()
        self._curves_dict.clear()
        self.image_view.clear_all_curves()
        self._selected_curve_id = None
        self._auto_box = None
        self._preview_plot_box()
        self.statusBar().showMessage(f"Loaded {self._image_path.name} ({rgb.shape[1]}x{rgb.shape[0]})")

    def _preview_plot_box(self) -> None:
        """Show a quick green outline of the detected plot box (no OCR needed)."""
        if self._image_rgb is None:
            return
        try:
            bgr = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)
            box = detect_plot_box(bgr)
            self._auto_box = box
            rect = np.array([
                [box.x, box.y], [box.x + box.w, box.y],
                [box.x + box.w, box.y + box.h], [box.x, box.y + box.h], [box.x, box.y],
            ], dtype=float)
            self.image_view.set_plotbox_preview(rect)
        except Exception:
            self.image_view.set_plotbox_preview(None)

    # --- Click router -------------------------------------------------------
    def _on_image_click(self, x: float, y: float) -> None:
        if self._image_rgb is None:
            return
        if self._mode == Mode.CALIBRATING:
            done = self.calib_panel.add_pixel_point(x, y)
            xs, ys = zip(*self.calib_panel.pixel_points()) if self.calib_panel.pixel_points() else ([], [])
            self.image_view.set_calibration_markers(list(xs), list(ys), list(CalibrationPanel.LABELS))
            if done:
                self.statusBar().showMessage("3 calibration points captured. Enter data values, then 'Solve calibration'.")
                self._set_mode(Mode.IDLE)
        elif self._mode == Mode.PICKING_COLOR:
            ix, iy = int(round(x)), int(round(y))
            ix = max(0, min(self._image_rgb.shape[1] - 1, ix))
            iy = max(0, min(self._image_rgb.shape[0] - 1, iy))
            hsv = masking.rgb_pixel_to_hsv(self._image_rgb, ix, iy)
            
            self._curve_id_counter += 1
            cid = self._curve_id_counter
            curve = Curve(
                id=cid,
                name=f"Curve {cid}",
                hsv_center=tuple(int(v) for v in hsv),
                hsv_tol=self.curve_panel.hsv_tol()
            )
            self._curves_dict[cid] = curve
            self.curve_panel.add_curve_card(curve)
            self.image_view.set_curve_points(cid, np.empty(0), np.empty(0), curve.hsv_center, curve.visible, curve.display_color)
            
            self._selected_curve_id = cid
            self.curve_panel.set_card_selected(cid)
            
            self.statusBar().showMessage(f"Created '{curve.name}' with HSV={curve.hsv_center}")
            self._refresh_mask_overlay()
        elif self._mode == Mode.SETTING_SEED:
            cid = getattr(self, "_seed_target_curve_id", None)
            if cid in self._curves_dict:
                self._curves_dict[cid].seed_point = (x, y)
                self.image_view.set_seed_marker(cid, x, y)
                self.statusBar().showMessage(f"Start point set for '{self._curves_dict[cid].name}'.")
            self._set_mode(Mode.IDLE)
        elif self._mode == Mode.SETTING_END:
            cid = getattr(self, "_seed_target_curve_id", None)
            if cid in self._curves_dict:
                self._curves_dict[cid].end_point = (x, y)
                self.image_view.set_end_marker(cid, x, y)
                self.statusBar().showMessage(f"End point set for '{self._curves_dict[cid].name}'.")
            self._set_mode(Mode.IDLE)

    # --- Auto-Calibrate (OCR) ------------------------------------------------
    def _auto_calibrate(self) -> None:
        """Run plot-box detection (fast, on main thread) then OCR (threaded)."""
        if self._image_rgb is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return

        self.calib_panel.set_auto_enabled(False)
        self.calib_panel.set_auto_status("Detecting plot grid...")
        self.statusBar().showMessage("Auto-calibrate: detecting plot grid...")

        bgr = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)

        try:
            self._auto_box = detect_plot_box(bgr)
        except Exception as exc:
            self.calib_panel.set_auto_enabled(True)
            self.calib_panel.set_auto_status(f"Grid detection failed: {exc}")
            return

        self.calib_panel.set_auto_status(
            f"Grid found ({self._auto_box.w}×{self._auto_box.h}). Running OCR..."
        )
        self.statusBar().showMessage("Auto-calibrate: running OCR (this may take a few seconds)...")

        # Launch OCR in a background thread
        self._ocr_thread = QThread()
        self._ocr_worker = _OcrWorker(bgr, self._auto_box)
        self._ocr_worker.moveToThread(self._ocr_thread)
        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.finished.connect(self._ocr_thread.quit)
        self._ocr_thread.start()

    def _on_ocr_finished(self, info: AxisInfo | None, error: str | None) -> None:
        """Callback when the OCR background thread completes."""
        self.calib_panel.set_auto_enabled(True)
        self._last_ocr_info = info

        if error or info is None:
            self.calib_panel.set_auto_status(f"OCR failed: {error}")
            self.statusBar().showMessage("Auto-calibrate failed.")
            return

        # Check we have enough info to calibrate
        if info.x_min is None or info.x_max is None or info.y_min is None or info.y_max is None:
            missing = []
            if info.x_min is None: missing.append("X-min")
            if info.x_max is None: missing.append("X-max")
            if info.y_min is None: missing.append("Y-min")
            if info.y_max is None: missing.append("Y-max")
            self.calib_panel.set_auto_status(
                f"OCR could not detect: {', '.join(missing)}. Try manual calibration."
            )
            return

        if info.x_min_pixel is None or info.x_max_pixel is None \
           or info.y_min_pixel is None or info.y_max_pixel is None:
            self.calib_panel.set_auto_status("Could not map tick positions. Try manual.")
            return

        # Build 3 calibration points from detected tick pixel positions
        # Origin pixel = (leftmost X-tick pixel X, bottommost Y-tick pixel Y)
        # X-max pixel  = (rightmost X-tick pixel X, bottommost Y-tick pixel Y)
        # Y-max pixel  = (leftmost X-tick pixel X, topmost Y-tick pixel Y)
        origin_px = (info.x_min_pixel, info.y_min_pixel)
        xmax_px = (info.x_max_pixel, info.y_min_pixel)
        ymax_px = (info.x_min_pixel, info.y_max_pixel)

        # Set pixel points on the panel
        self.calib_panel.set_pixel_points([origin_px, xmax_px, ymax_px])
        xs_markers = [origin_px[0], xmax_px[0], ymax_px[0]]
        ys_markers = [origin_px[1], xmax_px[1], ymax_px[1]]
        self.image_view.set_calibration_markers(xs_markers, ys_markers, list(CalibrationPanel.LABELS))

        # Set data values
        self.calib_panel.set_axis_values(
            x_origin=info.x_min, y_origin=info.y_min,
            x_max=info.x_max, y_max=info.y_max,
        )

        # Auto-solve
        self._solve_calibration()

        # Report success
        status_parts = [f"X: ({info.x_min}, {info.x_max})", f"Y: ({info.y_min}, {info.y_max})"]
        if info.x_label:
            status_parts.append(f"X-label: '{info.x_label}'")
        if info.y_label:
            status_parts.append(f"Y-label: '{info.y_label}'")
        self.calib_panel.set_auto_status("✅ " + " | ".join(status_parts))

        # Also auto-detect curve colors in one shot
        self._auto_detect_curves()

    # --- Debug Overlay --------------------------------------------------------
    def _show_debug_overlay(self) -> None:
        """Toggle: show detection debug overlay / restore original image."""
        if self._image_rgb is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return

        # If already showing debug, restore original
        if getattr(self, '_debug_showing', False):
            self.image_view.set_image(self._image_rgb)
            self._debug_showing = False
            self.statusBar().showMessage("Restored original image.")
            return

        bgr = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)
        try:
            box = getattr(self, '_auto_box', None)
            if box is None:
                box = detect_plot_box(bgr)
            self.statusBar().showMessage("Generating debug overlay (running OCR)...")
            QApplication.processEvents()
            overlay_rgb = generate_debug_overlay(bgr, box)
            self.image_view.set_image(overlay_rgb)
            self._debug_showing = True
            self.statusBar().showMessage("Debug overlay active. Click \U0001f50d again to restore.")
        except Exception as exc:
            QMessageBox.warning(self, "Debug failed", str(exc))

    # --- Auto-Detect Curves --------------------------------------------------
    def _auto_detect_curves(self) -> None:
        """Detect dominant curve colors via K-Means and create curve profiles."""
        if self._image_rgb is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return

        # Need a plot box — use cached from auto-calibrate or detect now
        bgr = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)
        box = getattr(self, '_auto_box', None)
        if box is None:
            try:
                box = detect_plot_box(bgr)
                self._auto_box = box
            except Exception as exc:
                QMessageBox.warning(self, "Detection failed", str(exc))
                return

        colors = detect_curve_colors(bgr, box)
        if not colors:
            QMessageBox.information(self, "No curves", "Could not detect any colored curves.")
            return

        count = 0
        for r, g, b in colors:
            # Convert RGB to HSV for the curve profile
            pixel = np.uint8([[[r, g, b]]])
            hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

            self._curve_id_counter += 1
            cid = self._curve_id_counter
            curve = Curve(
                id=cid,
                name=f"Curve {cid}",
                hsv_center=(h, s, v),
                hsv_tol=self.curve_panel.hsv_tol(),
            )
            self._curves_dict[cid] = curve
            self.curve_panel.add_curve_card(curve)
            self.image_view.set_curve_points(cid, np.empty(0), np.empty(0), (h, s, v), curve.visible, curve.display_color)
            count += 1

        self.statusBar().showMessage(f"Auto-detected {count} curve color(s).")

    # --- Calibration --------------------------------------------------------
    def _solve_calibration(self, silent: bool = False) -> None:
        if not self.calib_panel.has_enough_pixel_points():
            if not silent:
                QMessageBox.warning(self, "Need 3 points", "Click required calibration points first.")
            return
        pixel_pts = np.array(self.calib_panel.pixel_points(), dtype=float)
        data_pts = np.array(self.calib_panel.data_points(), dtype=float)
        if np.ptp(data_pts[:, 0]) == 0 or np.ptp(data_pts[:, 1]) == 0:
            self.calib_panel.set_solved_status(
                False,
                "X-axis or Y-axis range is zero — set non-zero max values before solving.",
            )
            return
        try:
            M = calib_mod.affine_from_points(pixel_pts, data_pts)
        except ValueError as exc:
            self.calib_panel.set_solved_status(False, str(exc))
            return
        self._calibration_M = M
        self._calibration_pixel_pts = [tuple(p) for p in pixel_pts.tolist()]
        self._calibration_data_pts = [tuple(p) for p in data_pts.tolist()]
        err = calib_mod.round_trip_error(M, pixel_pts, data_pts)
        self._recompute_curve_data()
        self._draw_calibration_grid()
        self.image_view.set_plotbox_preview(None)
        self.calib_panel.set_solved_status(True, f"max round-trip error = {err:.4g}")
        self.statusBar().showMessage(f"Calibration solved (err = {err:.4g}). Switch to Curves panel.")
        if not silent:
            QMessageBox.information(self, "Success", "Calibration solved successfully!\nA grid preview has been overlaid on your image.")

    def _draw_calibration_grid(self) -> None:
        if self._calibration_M is None:
            self.image_view.set_grid_lines([])
            return
        
        data_pts = np.array(self.calib_panel.data_points())
        x_min, x_max = np.min(data_pts[:, 0]), np.max(data_pts[:, 0])
        y_min, y_max = np.min(data_pts[:, 1]), np.max(data_pts[:, 1])
        
        if x_min == x_max or y_min == y_max:
            return
            
        xs = np.linspace(x_min, x_max, 10)
        ys = np.linspace(y_min, y_max, 10)
        
        lines = []
        for x in xs:
            pts = np.array([[x, y_min], [x, y_max]])
            pix = transform.data_to_pixel(pts, self._calibration_M)
            lines.append(pix)
            
        for y in ys:
            pts = np.array([[x_min, y], [x_max, y]])
            pix = transform.data_to_pixel(pts, self._calibration_M)
            lines.append(pix)
            
        self.image_view.set_grid_lines(lines)

    def _recompute_curve_data(self) -> None:
        if self._calibration_M is None:
            return
        for c in self._curves_dict.values():
            if c.pixel_xs.size == 0:
                continue
            pts = np.column_stack([c.pixel_xs, c.pixel_ys])
            data = transform.pixel_to_data(pts, self._calibration_M)
            c.data_xs = data[:, 0]
            c.data_ys = data[:, 1]

    def _reset_calibration(self) -> None:
        self._calibration_M = None
        self._calibration_pixel_pts.clear()
        self._calibration_data_pts.clear()
        self.image_view.set_calibration_markers([], [])
        self.image_view.set_grid_lines([])

    # --- Curve picking & masking -------------------------------------------
    def _on_hsv_changed(self, _center, _tol) -> None:
        self._refresh_mask_overlay()

    def _on_overlay_toggled(self, on: bool) -> None:
        if on:
            self._refresh_mask_overlay()
        else:
            self.image_view.clear_mask()

    def _refresh_mask_overlay(self) -> None:
        if self._image_rgb is None:
            return
            
        cid = getattr(self, "_selected_curve_id", None)
        if cid is None or cid not in self._curves_dict:
            return
            
        center = self._curves_dict[cid].hsv_center
        mask = masking.hsv_mask(self._image_rgb, center, self.curve_panel.hsv_tol())
        self.image_view.set_mask_overlay(masking.mask_overlay_rgba(mask))

    # --- X-Step extraction --------------------------------------------------
    def _calibration_bbox(self) -> tuple[int, int, int, int] | None:
        """Pixel bbox to scan. X is constrained to between origin and X-axis-max
        pixel columns. Y is the calibrated range expanded asymmetrically: more
        room below the origin pixel-row (where slightly-negative values live)
        and a small buffer above the Y-axis-max row."""
        if not self._calibration_pixel_pts or self._image_rgb is None:
            return None
        xs = [p[0] for p in self._calibration_pixel_pts]
        ys = [p[1] for p in self._calibration_pixel_pts]
        h, w = self._image_rgb.shape[:2]
        y_lo, y_hi = min(ys), max(ys)  # y_lo = top of plot, y_hi = origin row
        plot_h = max(1, y_hi - y_lo)
        below_margin = max(8, int(0.20 * plot_h))  # generous below origin
        above_margin = max(4, int(0.05 * plot_h))  # tighter above y-max
        y_lo = max(0, int(y_lo - above_margin))
        y_hi = min(h - 1, int(y_hi + below_margin))
        x_lo = max(0, int(min(xs)))
        x_hi = min(w - 1, int(max(xs)))
        return (x_lo, y_lo, x_hi, y_hi)

    def _on_extract_single(self, curve_id: int, dx: int, fill: bool, reducer: str, smooth: bool, smooth_window: int, poly_order: int, passes: int, upscale_factor: int, bestfit: bool, bestfit_degree: int) -> None:
        if curve_id not in self._curves_dict:
            return
        if self._calibration_M is None:
            QMessageBox.warning(self, "Not calibrated", "Solve calibration first before extracting.")
            return
        self._run_xstep_for_curve(self._curves_dict[curve_id], dx, fill, reducer, smooth, smooth_window, poly_order, passes, upscale_factor, bestfit, bestfit_degree)

    def _on_extract_all(self, dx: int, fill: bool, reducer: str, smooth: bool, smooth_window: int, poly_order: int, passes: int, upscale_factor: int, bestfit: bool, bestfit_degree: int) -> None:
        if self._calibration_M is None:
            QMessageBox.warning(self, "Not calibrated", "Solve calibration first before extracting.")
            return
        for c in self._curves_dict.values():
            if c.visible:
                self._run_xstep_for_curve(c, dx, fill, reducer, smooth, smooth_window, poly_order, passes, upscale_factor, bestfit, bestfit_degree)

    def _run_xstep_for_curve(self, curve: Curve, dx: int, fill: bool, reducer: str,
                             smooth: bool = False, smooth_window: int = 21,
                             poly_order: int = 3, passes: int = 1, upscale_factor: int = 1,
                             bestfit: bool = False, bestfit_degree: int = 3) -> None:
        if self._image_rgb is None:
            return
        if curve.manually_edited:
            reply = QMessageBox.question(
                self, "Discard manual edits?",
                f"'{curve.name}' has manually edited points. Re-extracting will discard them. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            curve.manually_edited = False
            
        tol = self.curve_panel.hsv_tol()
        mask = masking.hsv_mask(self._image_rgb, curve.hsv_center, tol)
        
        exc_rect = self.image_view.get_exclusion_roi_rect()
        if exc_rect is not None:
            ex_x0, ex_y0, ex_x1, ex_y1 = exc_rect
            h, w = mask.shape
            ex_x0, ex_x1 = max(0, min(w, ex_x0)), max(0, min(w, ex_x1))
            ex_y0, ex_y1 = max(0, min(h, ex_y0)), max(0, min(h, ex_y1))
            if ex_x1 > ex_x0 and ex_y1 > ex_y0:
                mask = mask.copy()
                mask[ex_y0:ex_y1, ex_x0:ex_x1] = 0

        bbox = self._calibration_bbox()
        seed_x, seed_y = curve.seed_point if curve.seed_point is not None else (None, None)
        end_x = curve.end_point[0] if curve.end_point is not None else None
        xs, ys = xstep.extract_curve(
            mask, dx=dx, bbox=bbox, reducer=reducer, upscale_factor=upscale_factor,
            seed_x=seed_x, seed_y=seed_y, end_x=end_x,
        )
        if xs.size < 2:
            self.statusBar().showMessage(f"X-Step found < 2 points for '{curve.name}'. Widen HSV tolerance.")
            return
        if fill:
            try:
                xs, ys = interpolate.fill_gaps_parametric(xs, ys, step=1.0)
            except ValueError as exc:
                self.statusBar().showMessage(f"Gap fill skipped: {exc}")
        
        if smooth and xs.size >= smooth_window:
            from scipy.signal import savgol_filter
            win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            win = min(win, xs.size if xs.size % 2 == 1 else xs.size - 1)
            poly = min(poly_order, win - 1)
            if win >= 5 and poly >= 1:
                for _ in range(passes):
                    ys = savgol_filter(ys, window_length=win, polyorder=poly)

        if bestfit:
            try:
                xs, ys = interpolate.polynomial_best_fit(xs, ys, bestfit_degree)
            except ValueError as exc:
                self.statusBar().showMessage(f"Best-fit skipped: {exc}")

        curve.pixel_xs = xs
        curve.pixel_ys = ys
        
        if self._calibration_M is not None:
            pts_pixel = np.column_stack([xs, ys])
            pts_data = transform.pixel_to_data(pts_pixel, self._calibration_M)
            curve.data_xs = pts_data[:, 0]
            curve.data_ys = pts_data[:, 1]
            
        self.image_view.set_curve_points(curve.id, xs, ys, curve.hsv_center, curve.visible, curve.display_color)
        self.statusBar().showMessage(f"Extracted {xs.size} points for '{curve.name}'.")

    def _delete_curve(self, curve_id: int) -> None:
        if curve_id in self._curves_dict:
            del self._curves_dict[curve_id]
            self.curve_panel.remove_curve_card(curve_id)
            self.image_view.remove_curve(curve_id)
            if self._selected_curve_id == curve_id:
                self._selected_curve_id = None
                self.image_view.clear_mask()

    def _on_curve_selected(self, curve_id: int) -> None:
        if curve_id in self._curves_dict:
            self._selected_curve_id = curve_id
            self.curve_panel.set_card_selected(curve_id)
            self._refresh_mask_overlay()

    def _on_curve_visibility_changed(self, curve_id: int, visible: bool) -> None:
        if curve_id in self._curves_dict:
            self._curves_dict[curve_id].visible = visible
            c = self._curves_dict[curve_id]
            self.image_view.set_curve_points(curve_id, c.pixel_xs, c.pixel_ys, c.hsv_center, visible, c.display_color)

    def _on_curve_name_changed(self, curve_id: int, name: str) -> None:
        if curve_id in self._curves_dict:
            self._curves_dict[curve_id].name = name

    def _on_curve_hsv_changed(self, curve_id: int, hsv: tuple) -> None:
        if curve_id not in self._curves_dict:
            return
        c = self._curves_dict[curve_id]
        c.hsv_center = hsv
        self.image_view.set_curve_points(curve_id, c.pixel_xs, c.pixel_ys, c.hsv_center, c.visible, c.display_color)
        if self._selected_curve_id == curve_id:
            self._refresh_mask_overlay()

    def _on_curve_display_color_changed(self, curve_id: int, rgb: tuple) -> None:
        if curve_id not in self._curves_dict:
            return
        c = self._curves_dict[curve_id]
        c.display_color = rgb
        self.image_view.set_curve_points(curve_id, c.pixel_xs, c.pixel_ys, c.hsv_center, c.visible, c.display_color)

    def _on_curve_points_edited(self, curve_id: int, xs, ys) -> None:
        if curve_id not in self._curves_dict:
            return
        c = self._curves_dict[curve_id]
        c.data_xs = np.asarray(xs, dtype=float)
        c.data_ys = np.asarray(ys, dtype=float)
        c.manually_edited = True
        if self._calibration_M is not None and c.data_xs.size:
            pts_data = np.column_stack([c.data_xs, c.data_ys])
            pts_pixel = transform.data_to_pixel(pts_data, self._calibration_M)
            c.pixel_xs = pts_pixel[:, 0]
            c.pixel_ys = pts_pixel[:, 1]
            self.image_view.set_curve_points(curve_id, c.pixel_xs, c.pixel_ys, c.hsv_center, c.visible, c.display_color)
        self.statusBar().showMessage(f"'{c.name}' edited manually ({c.data_xs.size} points).")

    def _on_set_seed_requested(self, curve_id: int) -> None:
        if curve_id not in self._curves_dict:
            return
        if self._image_rgb is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        self._seed_target_curve_id = curve_id
        self._set_mode(Mode.SETTING_SEED)
        self.statusBar().showMessage("Click the graph to set this curve's start point.")

    def _on_set_end_requested(self, curve_id: int) -> None:
        if curve_id not in self._curves_dict:
            return
        if self._image_rgb is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        self._seed_target_curve_id = curve_id
        self._set_mode(Mode.SETTING_END)
        self.statusBar().showMessage("Click the graph to set this curve's end point.")

    # --- Export -------------------------------------------------------------
    def _selected_curve(self) -> Curve | None:
        cid = getattr(self, "_selected_curve_id", None)
        if cid is not None:
            return self._curves_dict.get(cid)
        return None

    def _export_csv_active(self) -> None:
        c = self._selected_curve()
        if c is None:
            QMessageBox.warning(self, "No curve", "Select a curve first.")
            return
        path_str, _ = QFileDialog.getSaveFileName(self, "Save curve CSV", f"{c.name}.csv", "CSV (*.csv)")
        if not path_str:
            return
        csv_export.write_curve_csv(path_str, c.data_xs, c.data_ys)
        self.statusBar().showMessage(f"Wrote {path_str}")

    def _export_csv_wide(self) -> None:
        if not self._curves_dict:
            QMessageBox.warning(self, "No curves", "Add at least one curve first.")
            return
        path_str, _ = QFileDialog.getSaveFileName(self, "Save wide CSV", "curves.csv", "CSV (*.csv)")
        if not path_str:
            return
        csv_export.write_curves_wide(
            path_str,
            [(c.name, c.data_xs, c.data_ys) for c in self._curves_dict.values()],
        )
        self.statusBar().showMessage(f"Wrote {path_str}")

    def _export_json(self) -> None:
        if self._image_rgb is None:
            return
        path_str, _ = QFileDialog.getSaveFileName(self, "Save session JSON", "session.json", "JSON (*.json)")
        if not path_str:
            return
        calib_pts_payload = [
            {"pixel": list(p), "data": list(d)}
            for p, d in zip(self._calibration_pixel_pts, self._calibration_data_pts)
        ]
        curves_payload = [
            json_export.serialize_curve(
                c.name, c.hsv_center, c.hsv_tol,
                c.pixel_xs, c.pixel_ys, c.data_xs, c.data_ys,
            )
            for c in self._curves_dict.values()
        ]
        payload = json_export.build_payload(
            str(self._image_path) if self._image_path else None,
            self._calibration_M,
            calib_pts_payload,
            curves_payload,
        )
        json_export.write_payload(path_str, payload)
        self.statusBar().showMessage(f"Wrote {path_str}")

    def _copy_clipboard(self) -> None:
        c = self._selected_curve()
        if c is None:
            QMessageBox.warning(self, "No curve", "Select a curve first.")
            return
        clip_mod.copy_curve_tsv(c.data_xs, c.data_ys)
        self.statusBar().showMessage(f"Copied '{c.name}' to clipboard.")

    def _on_tab_changed(self, index: int) -> None:
        if index == 2:  # Export tab
            self._refresh_export_panel()

    def _refresh_export_panel(self) -> None:
        curves = list(self._curves_dict.values())
        self.export_panel.populate_curves(curves)

    def _export_csv_checked(self, curve_ids: list[int]) -> None:
        curves = [self._curves_dict[cid] for cid in curve_ids if cid in self._curves_dict]
        if not curves:
            QMessageBox.warning(self, "No curves", "No curves selected for export.")
            return
        
        if len(curves) == 1:
            c = curves[0]
            path_str, _ = QFileDialog.getSaveFileName(self, "Save curve CSV", f"{c.name}.csv", "CSV (*.csv)")
            if not path_str:
                return
            csv_export.write_curve_csv(path_str, c.data_xs, c.data_ys)
            self.statusBar().showMessage(f"Wrote {path_str}")
        else:
            folder = QFileDialog.getExistingDirectory(self, "Choose folder for CSVs")
            if not folder:
                return
            from pathlib import Path
            count = 0
            for c in curves:
                if c.data_xs.size == 0:
                    continue
                fpath = Path(folder) / f"{c.name}.csv"
                csv_export.write_curve_csv(str(fpath), c.data_xs, c.data_ys)
                count += 1
            self.statusBar().showMessage(f"Exported {count} curves to {folder}")

    def _show_expert_debug_plot(self) -> None:
        """Plot the original image with extracted graphs and OCR data overlay for debugging."""
        if self._image_rgb is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency", "Matplotlib is not installed. Please install it to use this feature.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self._image_rgb)

        # Plot each visible curve in pixel coordinates
        for c in self._curves_dict.values():
            if c.visible and c.pixel_xs is not None and c.pixel_xs.size > 0:
                h, s, v = c.hsv_center
                pixel = np.uint8([[[h, s, v]]])
                rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0]
                color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
                ax.plot(c.pixel_xs, c.pixel_ys, label=c.name, color=color, linewidth=2)

        info = getattr(self, '_last_ocr_info', None)
        if info is not None:
            title = info.title if info.title else "(No title detected)"
            ax.set_title(f"OCR Title: {title}", fontsize=14, color='darkblue', fontweight='bold')

            x_label = info.x_label if info.x_label else "(No X-label detected)"
            ax.set_xlabel(f"OCR X-Label: {x_label}", fontsize=12, color='darkred', fontweight='bold')

            y_label = info.y_label if info.y_label else "(No Y-label detected)"
            ax.set_ylabel(f"OCR Y-Label: {y_label}", fontsize=12, color='darkred', fontweight='bold')

            text_str = (
                f"Legend: {info.legend}\n"
                f"X-Ticks: {info.x_ticks_text}\n"
                f"Y-Ticks: {info.y_ticks_text}"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        else:
            ax.set_title("No OCR data available (Run Auto-Calibrate first)", fontsize=14, color='red')

        if any(c.visible and c.pixel_xs is not None and c.pixel_xs.size > 0 for c in self._curves_dict.values()):
            ax.legend(loc='upper right')
            
        plt.tight_layout()
        plt.show(block=False)

