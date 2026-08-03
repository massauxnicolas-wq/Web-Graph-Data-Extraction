from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
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
from digitizer.core import image_io, interpolate, masking, quality, transform, xstep
from digitizer.core.auto_detect import detect_plot_box, detect_curve_colors
from digitizer.io import clipboard as clip_mod
from digitizer.io import csv_export, json_export
from digitizer.ui.calibration_panel import CalibrationPanel
from digitizer.ui.curve_panel import Curve, CurvePanel
from digitizer.ui.edit_panel import EditPanel
from digitizer.ui.export_dialog import ExportPanel
from digitizer.ui.image_view import ImageView


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
        self._edit_curve_id: int | None = None
        self._mode: Mode = Mode.IDLE

        # Widgets
        self.image_view = ImageView()
        self.image_view.image_clicked.connect(self._on_image_click)

        self.calib_panel = CalibrationPanel()
        self.calib_panel.start_capture.connect(self._enter_calibration_mode)
        self.calib_panel.cancel_capture.connect(self._exit_to_idle)
        self.calib_panel.solve_requested.connect(self._solve_calibration)
        self.calib_panel.reset_requested.connect(self._reset_calibration)
        self.calib_panel.points_changed.connect(self._on_calib_points_changed)
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

        self.edit_panel = EditPanel()
        self.edit_panel.edit_mode_toggled.connect(self._on_edit_mode_toggled)
        self.edit_panel.curve_selected.connect(self._on_edit_curve_selected)
        self.edit_panel.image_opacity_changed.connect(lambda v: self.image_view.set_image_opacity(v / 100))
        self.edit_panel.delete_point_requested.connect(self._on_delete_point)
        self.edit_panel.delete_outliers_requested.connect(self._on_delete_outliers)
        self.edit_panel.refresh_requested.connect(self._refresh_edit_panel)
        self.image_view.curve_points_edited.connect(self._on_curve_canvas_edited)
        self.image_view.edit_selection_changed.connect(self._on_edit_selection_changed)

        self.export_panel = ExportPanel()
        self.export_panel.export_csv_active.connect(self._export_csv_active)
        self.export_panel.export_csv_wide.connect(self._export_csv_wide)
        self.export_panel.export_csv_checked.connect(self._export_csv_checked)
        self.export_panel.export_json.connect(self._export_json)
        self.export_panel.copy_clipboard.connect(self._copy_clipboard)
        self.export_panel.refresh_requested.connect(self._refresh_export_panel)

        self._tabs = QTabWidget()
        self._tabs.addTab(self.calib_panel, "1. Calibrate")
        self._tabs.addTab(self.curve_panel, "2. Curves")
        self._tabs.addTab(self.edit_panel, "3. Editing")
        self._tabs.addTab(self.export_panel, "4. Export")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self.image_view)
        self._splitter.addWidget(self._tabs)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)
        self.setCentralWidget(self._splitter)

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
        # Clicking places points from scratch, so drop the auto-placed guess.
        self.calib_panel.set_pixel_points([])
        self.image_view.set_calibration_markers([], [])
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
        self._edit_curve_id = None
        self._auto_box = None
        self._preview_plot_box()
        self._place_default_calibration_points()
        self.statusBar().showMessage(f"Loaded {self._image_path.name} ({rgb.shape[1]}x{rgb.shape[0]})")

    def _place_default_calibration_points(self) -> None:
        """Pre-place Origin / X-max / Y-max on the detected plot box corners.

        A starting guess the user can drag into place, rather than an empty
        canvas. Starting a manual capture clears them (see
        _enter_calibration_mode) so clicking always starts from scratch.
        """
        box = getattr(self, "_auto_box", None)
        if box is None:
            return
        origin = (float(box.x), float(box.y + box.h))
        x_max = (float(box.x + box.w), float(box.y + box.h))
        y_max = (float(box.x), float(box.y))
        pts = [origin, x_max, y_max]
        self.calib_panel.set_pixel_points(pts)
        self.image_view.set_calibration_markers(
            [p[0] for p in pts], [p[1] for p in pts], list(CalibrationPanel.LABELS)
        )

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
            QMessageBox.information(self, "Success", "Calibration solved successfully!")

    def _draw_calibration_grid(self) -> None:
        """Grid in calibrated data steps, plus the X and Y axis lines.

        Drawn bold and blue on the Calibrate tab, faint black elsewhere -
        see ImageView.set_grid_mode.
        """
        if self._calibration_M is None:
            self.image_view.set_grid_lines([])
            return

        data_pts = np.array(self.calib_panel.data_points())
        x_min, x_max = np.min(data_pts[:, 0]), np.max(data_pts[:, 0])
        y_min, y_max = np.min(data_pts[:, 1]), np.max(data_pts[:, 1])
        if x_min == x_max or y_min == y_max:
            return

        def to_px(pts):
            return transform.data_to_pixel(np.array(pts, dtype=float), self._calibration_M)

        lines = []
        for x in np.linspace(x_min, x_max, 10):
            lines.append(to_px([[x, y_min], [x, y_max]]))
        for y in np.linspace(y_min, y_max, 10):
            lines.append(to_px([[x_min, y], [x_max, y]]))

        axis_lines = [
            to_px([[x_min, y_min], [x_max, y_min]]),  # X axis
            to_px([[x_min, y_min], [x_min, y_max]]),  # Y axis
        ]
        self.image_view.set_grid_lines(lines, axis_lines)

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
        end_x, end_y = curve.end_point if curve.end_point is not None else (None, None)
        xs, ys = xstep.extract_curve(
            mask, dx=dx, bbox=bbox, reducer=reducer, upscale_factor=upscale_factor,
            seed_x=seed_x, seed_y=seed_y, end_x=end_x, end_y=end_y,
        )
        if xs.size < 2:
            self.statusBar().showMessage(f"X-Step found < 2 points for '{curve.name}'. Widen HSV tolerance.")
            return
        # Marker removal (smooth) runs BEFORE gap fill: strip the diamond/star bumps
        # off the raw extracted points first, then interpolate through the clean curve.
        # Filling first would interpolate through bump-distorted points and leave a
        # denser series that the fixed window smooths less effectively.
        if smooth and xs.size >= smooth_window:
            from scipy.signal import savgol_filter
            win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            win = min(win, xs.size if xs.size % 2 == 1 else xs.size - 1)
            poly = min(poly_order, win - 1)
            if win >= 5 and poly >= 1:
                for _ in range(passes):
                    ys = savgol_filter(ys, window_length=win, polyorder=poly)

        if fill:
            try:
                xs, ys = interpolate.fill_gaps_parametric(xs, ys, step=1.0)
            except ValueError as exc:
                self.statusBar().showMessage(f"Gap fill skipped: {exc}")

        if bestfit:
            try:
                if curve.seed_point is not None and curve.end_point is not None:
                    xs, ys = interpolate.polynomial_best_fit_through_points(
                        xs, ys, bestfit_degree,
                        curve.seed_point[0], curve.seed_point[1],
                        curve.end_point[0], curve.end_point[1],
                    )
                else:
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
            self.image_view.remove_curve(curve_id)  # also stops editing if it was the edited one
            if self._edit_curve_id == curve_id:
                self._edit_curve_id = None
                self._refresh_edit_panel()
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

    # --- canvas point editing ------------------------------------------------
    def _on_edit_mode_toggled(self, on: bool) -> None:
        self._apply_canvas_editing(on)

    def _on_edit_curve_selected(self, curve_id: int) -> None:
        self._edit_curve_id = curve_id
        c = self._curves_dict.get(curve_id)
        self.edit_panel.update_quality(
            c.pixel_xs if c is not None else None,
            c.pixel_ys if c is not None else None,
        )
        self._apply_canvas_editing(self.edit_panel.is_edit_mode())

    def _apply_canvas_editing(self, on: bool) -> None:
        """Start/stop canvas editing, then always redraw every curve.

        Redrawing unconditionally is what keeps the canvas honest: whichever
        curve is editable gets routed to the editable item by
        ImageView.set_curve_points, and everything else back to its scatter.
        """
        self.image_view.stop_curve_editing()
        curve = self._curves_dict.get(self._edit_curve_id) if on else None
        if curve is not None and curve.pixel_xs.size:
            self.image_view.start_curve_editing(curve.id, curve.pixel_xs, curve.pixel_ys)
        for c in self._curves_dict.values():
            self.image_view.set_curve_points(c.id, c.pixel_xs, c.pixel_ys, c.hsv_center, c.visible, c.display_color)
        self.edit_panel.set_selected_point(-1, None, None)

    def _on_edit_selection_changed(self, index: int) -> None:
        xs, ys = self.image_view.edited_points()
        if 0 <= index < len(xs):
            self.edit_panel.set_selected_point(index, float(xs[index]), float(ys[index]))
        else:
            self.edit_panel.set_selected_point(-1, None, None)

    def _on_delete_point(self) -> None:
        self.image_view.delete_selected_point()

    def _on_delete_outliers(self) -> None:
        xs, ys = self.image_view.edited_points()
        if len(xs) < 5:
            return
        removed = self.image_view.delete_points_mask(quality.detect_outliers(xs, ys))
        self.statusBar().showMessage(f"Removed {removed} outlier(s)." if removed else "No outliers to remove.")

    def _on_curve_canvas_edited(self, curve_id: int, xs, ys) -> None:
        """Points edited on the canvas (pixel space)."""
        if curve_id not in self._curves_dict:
            return
        c = self._curves_dict[curve_id]
        c.pixel_xs = np.asarray(xs, dtype=float)
        c.pixel_ys = np.asarray(ys, dtype=float)
        c.manually_edited = True
        if self._calibration_M is not None and c.pixel_xs.size:
            pts_pixel = np.column_stack([c.pixel_xs, c.pixel_ys])
            pts_data = transform.pixel_to_data(pts_pixel, self._calibration_M)
            c.data_xs = pts_data[:, 0]
            c.data_ys = pts_data[:, 1]
        self.edit_panel.update_quality(c.pixel_xs, c.pixel_ys)
        self.statusBar().showMessage(f"'{c.name}' edited manually ({c.pixel_xs.size} points).")

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
        if index == 2:  # Editing
            self._refresh_edit_panel()
        elif index == 3:  # Export
            self._refresh_export_panel()
        # Bold blue grid while calibrating; a faint black reference grid with
        # visible axes everywhere else, so the plot area stays readable even
        # when the source image is at full opacity.
        self.image_view.set_grid_mode("calibration" if index == 0 else "reference")

    def _refresh_edit_panel(self) -> None:
        self.edit_panel.populate_curves(list(self._curves_dict.values()), self._edit_curve_id)

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

