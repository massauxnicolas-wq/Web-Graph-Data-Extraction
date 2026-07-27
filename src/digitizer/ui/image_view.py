from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget


class ImageView(pg.GraphicsLayoutWidget):
    """pyqtgraph view that emits pixel-coordinate clicks on the underlying image."""

    image_clicked = pyqtSignal(float, float)

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

        self.calib_scatter = pg.ScatterPlotItem(
            size=14, symbol="+", pen=pg.mkPen("r", width=2), brush=pg.mkBrush(0, 0, 0, 0)
        )
        self.calib_scatter.setZValue(11)
        self.view.addItem(self.calib_scatter)

        self.grid_curves: list[pg.PlotCurveItem] = []

        self.exclusion_roi = pg.RectROI([50, 50], [100, 100], pen=pg.mkPen("r", width=2))
        self.exclusion_roi.setZValue(20)
        self.exclusion_roi.setVisible(False)
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

    def set_mask_overlay(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self.clear_mask()
            return
        self.mask_item.setImage(rgba, autoLevels=False)
        self.mask_item.setVisible(True)

    def clear_mask(self) -> None:
        self.mask_item.clear()
        self.mask_item.setVisible(False)

    def set_curve_points(self, curve_id: int, xs: np.ndarray, ys: np.ndarray, hsv: tuple[int, int, int], visible: bool) -> None:
        if curve_id not in self.curve_scatters:
            import cv2
            pixel = np.uint8([[[hsv[0], hsv[1], hsv[2]]]])
            rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0]
            # Use complementary color for contrast against the curve
            cr, cg, cb = 255 - int(rgb[0]), 255 - int(rgb[1]), 255 - int(rgb[2])
            # If the complement is too similar (greyish), force to bright cyan or magenta
            if abs(cr - int(rgb[0])) < 60 and abs(cg - int(rgb[1])) < 60 and abs(cb - int(rgb[2])) < 60:
                cr, cg, cb = 0, 255, 255
            scatter = pg.ScatterPlotItem(size=4, brush=pg.mkBrush(cr, cg, cb, 220), pen=None)
            scatter.setZValue(10)
            self.view.addItem(scatter)
            self.curve_scatters[curve_id] = scatter
            
        scatter = self.curve_scatters[curve_id]
        scatter.setVisible(visible)
        
        if len(xs) == 0:
            scatter.clear()
            return
            
        scatter.setData(x=np.asarray(xs), y=np.asarray(ys))

    def remove_curve(self, curve_id: int) -> None:
        if curve_id in self.curve_scatters:
            scatter = self.curve_scatters.pop(curve_id)
            self.view.removeItem(scatter)

    def clear_all_curves(self) -> None:
        for scatter in self.curve_scatters.values():
            self.view.removeItem(scatter)
        self.curve_scatters.clear()

    def set_calibration_markers(self, xs: list[float], ys: list[float]) -> None:
        if not xs:
            self.calib_scatter.clear()
            return
        self.calib_scatter.setData(x=xs, y=ys)

    def set_grid_lines(self, lines: list[np.ndarray]) -> None:
        for c in self.grid_curves:
            self.view.removeItem(c)
        self.grid_curves.clear()
        
        pen = pg.mkPen(color=(0, 0, 255, 255), width=2, style=Qt.PenStyle.DashLine)
        for pix in lines:
            c = pg.PlotCurveItem(x=pix[:, 0], y=pix[:, 1], pen=pen)
            c.setZValue(12)
            self.view.addItem(c)
            self.grid_curves.append(c)

    def set_exclusion_roi_visible(self, visible: bool) -> None:
        self.exclusion_roi.setVisible(visible)
        
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
        self.image_clicked.emit(x, y)
