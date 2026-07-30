from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from digitizer.ui.editable_curve_item import EditableCurveItem


class ImageView(pg.GraphicsLayoutWidget):
    """pyqtgraph view that emits pixel-coordinate clicks on the underlying image."""

    image_clicked = pyqtSignal(float, float)
    calib_marker_moved = pyqtSignal(int, float, float)  # index, x, y
    exclusion_thumbnail_changed = pyqtSignal(object)  # QPixmap | None
    curve_points_edited = pyqtSignal(int, object, object)  # curve_id, pixel xs, pixel ys
    edit_selection_changed = pyqtSignal(int)  # index of selected point, -1 for none

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setBackground("w")
        self.view: pg.ViewBox = self.addViewBox()  # pyright: ignore[reportAttributeAccessIssue]
        self.view.setAspectLocked(True)
        self.view.invertY(True)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image_item)

        self.mask_item = pg.ImageItem(axisOrder="row-major")
        self.mask_item.setZValue(5)
        self.view.addItem(self.mask_item)

        self.curve_scatters: dict[int, pg.ScatterPlotItem] = {}
        self._editable_item: EditableCurveItem | None = None
        self._editable_curve_id: int | None = None

        self.calib_targets: list[pg.TargetItem] = []
        self.seed_markers: dict[int, pg.TargetItem] = {}
        self.end_markers: dict[int, pg.TargetItem] = {}
        self.plotbox_preview: pg.PlotCurveItem | None = None

        self.grid_curves: list[pg.PlotCurveItem] = []
        self.axis_curves: list[pg.PlotCurveItem] = []
        self._grid_mode = "calibration"

        self.exclusion_roi = pg.RectROI([50, 50], [100, 100], pen=pg.mkPen("r", width=2))
        self.exclusion_roi.setZValue(20)
        self.exclusion_roi.setVisible(False)
        self.exclusion_roi.sigRegionChanged.connect(self._update_exclusion_thumbnail)
        self.view.addItem(self.exclusion_roi)

        scene = self.scene()
        scene.sigMouseClicked.connect(self._on_scene_click)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

    def set_image(self, rgb: np.ndarray) -> None:
        self.image_item.setImage(rgb, autoLevels=False)
        h, w = rgb.shape[:2]
        self.view.setRange(xRange=(0, w), yRange=(0, h), padding=0)
        self.clear_mask()
        self.clear_all_curves()
        self.set_calibration_markers([], [])
        self.set_grid_lines([])
        self.set_plotbox_preview(None)
        for cid in list(self.seed_markers):
            self.set_seed_marker(cid, None, None)
        for cid in list(self.end_markers):
            self.set_end_marker(cid, None, None)

    def set_mask_overlay(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self.clear_mask()
            return
        self.mask_item.setImage(rgba, autoLevels=False)
        self.mask_item.setVisible(True)

    def clear_mask(self) -> None:
        self.mask_item.clear()
        self.mask_item.setVisible(False)

    def set_curve_points(
        self, curve_id: int, xs: np.ndarray, ys: np.ndarray, hsv: tuple[int, int, int],
        visible: bool, display_color: tuple[int, int, int] | None = None,
    ) -> None:
        """The one way points reach the canvas.

        Routes to the editable item when this curve is the one being edited,
        otherwise to its plain scatter. It must always draw something -- an
        early return here silently swallows extraction results and made the
        canvas look dead in a previous revision.
        """
        if curve_id == self._editable_curve_id and self._editable_item is not None:
            self._editable_item.set_points(xs, ys, keep_selection=True)
            self._editable_item.setVisible(visible)
            return

        if display_color is not None:
            cr, cg, cb = display_color
        else:
            import cv2
            pixel = np.uint8([[[hsv[0], hsv[1], hsv[2]]]])
            rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0]
            # Use complementary color for contrast against the curve
            cr, cg, cb = 255 - int(rgb[0]), 255 - int(rgb[1]), 255 - int(rgb[2])
            # If the complement is too similar (greyish), force to bright cyan or magenta
            if abs(cr - int(rgb[0])) < 60 and abs(cg - int(rgb[1])) < 60 and abs(cb - int(rgb[2])) < 60:
                cr, cg, cb = 0, 255, 255

        if curve_id not in self.curve_scatters:
            scatter = pg.ScatterPlotItem(size=4, brush=pg.mkBrush(cr, cg, cb, 220), pen=None)
            scatter.setZValue(10)
            self.view.addItem(scatter)
            self.curve_scatters[curve_id] = scatter

        scatter = self.curve_scatters[curve_id]
        # While another curve is being edited everything else stays hidden.
        scatter.setVisible(visible and self._editable_curve_id is None)
        scatter.setBrush(pg.mkBrush(cr, cg, cb, 220))

        if len(xs) == 0:
            scatter.clear()
            return

        scatter.setData(x=np.asarray(xs), y=np.asarray(ys))

    def remove_curve(self, curve_id: int) -> None:
        if curve_id == self._editable_curve_id:
            self.stop_curve_editing()
        if curve_id in self.curve_scatters:
            scatter = self.curve_scatters.pop(curve_id)
            self.view.removeItem(scatter)
        self.set_seed_marker(curve_id, None, None)
        self.set_end_marker(curve_id, None, None)

    def clear_all_curves(self) -> None:
        self.stop_curve_editing()
        for scatter in self.curve_scatters.values():
            self.view.removeItem(scatter)
        self.curve_scatters.clear()

    # --- point editing on the canvas ---------------------------------------
    def start_curve_editing(self, curve_id: int, xs: np.ndarray, ys: np.ndarray) -> None:
        """Make one curve's points draggable/deletable, hiding the others."""
        self.stop_curve_editing()

        for cid, scatter in self.curve_scatters.items():
            scatter.setVisible(False)

        item = EditableCurveItem()
        item.set_points(xs, ys)
        item.setZValue(10)
        item.sigPointsEdited.connect(self._emit_edited)
        item.sigPointSelected.connect(self.edit_selection_changed)
        self.view.addItem(item)
        self._editable_item = item
        self._editable_curve_id = curve_id

    def stop_curve_editing(self) -> None:
        """Drop the editable item. Callers redraw curves via set_curve_points()."""
        if self._editable_item is not None:
            self.view.removeItem(self._editable_item)
        self._editable_item = None
        self._editable_curve_id = None

    def editing_curve_id(self) -> int | None:
        return self._editable_curve_id

    def edited_points(self) -> tuple[np.ndarray, np.ndarray]:
        if self._editable_item is None:
            return np.empty(0), np.empty(0)
        return self._editable_item.points()

    def delete_selected_point(self) -> bool:
        return self._editable_item.delete_selected() if self._editable_item is not None else False

    def delete_points_mask(self, mask: np.ndarray) -> int:
        return self._editable_item.delete_mask(mask) if self._editable_item is not None else 0

    def _emit_edited(self) -> None:
        if self._editable_item is None or self._editable_curve_id is None:
            return
        xs, ys = self._editable_item.points()
        self.curve_points_edited.emit(self._editable_curve_id, xs, ys)

    def set_image_opacity(self, value: float) -> None:
        self.image_item.setOpacity(value)

    def set_calibration_markers(self, xs: list[float], ys: list[float], labels: list[str] | None = None) -> None:
        """Rebuild the draggable calibration point markers.

        Each marker is an individually-draggable pg.TargetItem; dragging one
        emits calib_marker_moved(index, x, y) on release (not during the drag).
        """
        for t in self.calib_targets:
            self.view.removeItem(t)
        self.calib_targets = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            label = labels[i] if labels and i < len(labels) else str(i + 1)
            t = pg.TargetItem(
                pos=(x, y), size=14, symbol="crosshair",
                pen=pg.mkPen("r", width=2), movable=True, label=label,
            )
            t.setZValue(11)
            t.sigPositionChangeFinished.connect(
                lambda item, idx=i: self.calib_marker_moved.emit(idx, item.pos().x(), item.pos().y())
            )
            self.view.addItem(t)
            self.calib_targets.append(t)

    def _set_point_marker(
        self, markers: dict[int, pg.TargetItem], curve_id: int,
        x: float | None, y: float | None, symbol: str, color,
    ) -> None:
        if curve_id in markers:
            self.view.removeItem(markers.pop(curve_id))
        if x is None or y is None:
            return
        t = pg.TargetItem(pos=(x, y), size=14, symbol=symbol, pen=pg.mkPen(color, width=2), movable=False)
        t.setZValue(11)
        self.view.addItem(t)
        markers[curve_id] = t

    def set_seed_marker(self, curve_id: int, x: float | None, y: float | None) -> None:
        """Show (or clear, if x/y is None) a curve's forced start-point marker."""
        self._set_point_marker(self.seed_markers, curve_id, x, y, "star", "m")

    def set_end_marker(self, curve_id: int, x: float | None, y: float | None) -> None:
        """Show (or clear, if x/y is None) a curve's forced end-point marker."""
        self._set_point_marker(self.end_markers, curve_id, x, y, "s", (255, 140, 0))

    def set_plotbox_preview(self, rect_pixels: np.ndarray | None) -> None:
        """Show (or clear, if None) a green outline of the auto-detected plot box."""
        if self.plotbox_preview is not None:
            self.view.removeItem(self.plotbox_preview)
            self.plotbox_preview = None
        if rect_pixels is None:
            return
        pen = pg.mkPen(color=(0, 200, 0, 255), width=2)
        self.plotbox_preview = pg.PlotCurveItem(x=rect_pixels[:, 0], y=rect_pixels[:, 1], pen=pen)
        self.plotbox_preview.setZValue(9)
        self.view.addItem(self.plotbox_preview)

    def set_grid_lines(self, lines: list[np.ndarray], axis_lines: list[np.ndarray] | None = None) -> None:
        """Set the calibrated grid, plus the two axis lines drawn on top of it."""
        for c in self.grid_curves + self.axis_curves:
            self.view.removeItem(c)
        self.grid_curves.clear()
        self.axis_curves.clear()

        for pix in lines:
            c = pg.PlotCurveItem(x=pix[:, 0], y=pix[:, 1])
            c.setZValue(3)
            self.view.addItem(c)
            self.grid_curves.append(c)

        for pix in axis_lines or []:
            c = pg.PlotCurveItem(x=pix[:, 0], y=pix[:, 1])
            c.setZValue(4)
            self.view.addItem(c)
            self.axis_curves.append(c)

        self._apply_grid_style()

    def set_grid_mode(self, mode: str) -> None:
        """'calibration' = bold blue grid, 'reference' = faint black grid + axes."""
        self._grid_mode = mode
        self._apply_grid_style()

    def _apply_grid_style(self) -> None:
        if self._grid_mode == "calibration":
            grid_pen = pg.mkPen(color=(0, 0, 255, 255), width=2, style=Qt.PenStyle.DashLine)
            axis_pen = pg.mkPen(color=(0, 0, 255, 255), width=3)
        else:
            # Faint black so it reads over a fully opaque chart without fighting it.
            grid_pen = pg.mkPen(color=(0, 0, 0, 77), width=1)
            axis_pen = pg.mkPen(color=(0, 0, 0, 180), width=2)
        for c in self.grid_curves:
            c.setPen(grid_pen)
        for c in self.axis_curves:
            c.setPen(axis_pen)

    def set_exclusion_roi_visible(self, visible: bool) -> None:
        self.exclusion_roi.setVisible(visible)
        self._update_exclusion_thumbnail()

    def _update_exclusion_thumbnail(self, *_args) -> None:
        rect = self.get_exclusion_roi_rect()
        img = self.image_item.image
        if rect is None or img is None:
            self.exclusion_thumbnail_changed.emit(None)
            return
        h, w = img.shape[:2]
        x0, y0, x1, y1 = rect
        x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
        y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            self.exclusion_thumbnail_changed.emit(None)
            return
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        from PyQt6.QtGui import QImage, QPixmap
        qimg = QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, QImage.Format.Format_RGB888)
        self.exclusion_thumbnail_changed.emit(QPixmap.fromImage(qimg.copy()))


    def get_exclusion_roi_rect(self) -> tuple[int, int, int, int] | None:
        if not self.exclusion_roi.isVisible():
            return None
        pos = self.exclusion_roi.pos()
        size = self.exclusion_roi.size()
        x0, y0 = int(round(pos.x())), int(round(pos.y()))
        w, h = int(round(size.x())), int(round(size.y()))
        return (x0, y0, x0 + w, y0 + h)

    def _on_scene_click(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if ev.isAccepted():
            return  # an item (e.g. a point being selected) already handled this
        scene_pos = ev.scenePos()
        if not self.view.sceneBoundingRect().contains(scene_pos):
            return
        view_pos = self.view.mapSceneToView(scene_pos)
        x, y = float(view_pos.x()), float(view_pos.y())
        img = self.image_item.image
        if img is None:
            return
        h, w = img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        # While editing, a click on empty canvas adds a point to that curve.
        if self._editable_item is not None:
            self._editable_item.insert_point(x, y)
            return

        self.image_clicked.emit(x, y)
