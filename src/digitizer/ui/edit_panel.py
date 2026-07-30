from __future__ import annotations

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from digitizer.core import quality
from digitizer.ui.curve_panel import Curve
from digitizer.ui.editable_curve_item import EditableCurveItem


class EditPanel(QWidget):
    """Curve point editing + quality analysis, in calibrated data space.

    Editing lives here rather than on the main image canvas so that the
    canvas keeps its plain pan/zoom behaviour untouched. MainWindow widens
    the side panel while this tab is active so there is room to work.
    """

    points_edited = pyqtSignal(int, object, object)  # curve_id, xs, ys
    refresh_requested = pyqtSignal()
    expert_debug_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curves: list[Curve] = []
        self._curve_id: int | None = None
        self._item: EditableCurveItem | None = None

        layout = QVBoxLayout(self)

        # --- Which curve ---
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Curve:"))
        self._curve_combo = QComboBox()
        self._curve_combo.currentIndexChanged.connect(self._on_curve_chosen)
        pick_row.addWidget(self._curve_combo, 1)
        self._refresh_btn = QPushButton("🔄")
        self._refresh_btn.setFixedWidth(34)
        self._refresh_btn.setToolTip("Reload curves from the extraction step")
        self._refresh_btn.clicked.connect(self.refresh_requested)
        pick_row.addWidget(self._refresh_btn)
        layout.addLayout(pick_row)

        # --- The editing plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "X")
        self.plot_widget.setLabel("left", "Y")
        self.plot_widget.setMinimumHeight(320)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)
        layout.addWidget(self.plot_widget, 1)

        hint = QLabel(
            "Drag a point to move it  ·  Click a point to select it  ·  "
            "Click empty space to add a point"
        )
        hint.setStyleSheet("QLabel { font-size: 11px; color: #666; }")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # --- Point actions ---
        act_box = QGroupBox("Points")
        act_layout = QVBoxLayout(act_box)

        self._sel_lbl = QLabel("No point selected.")
        self._sel_lbl.setWordWrap(True)
        act_layout.addWidget(self._sel_lbl)

        self._del_btn = QPushButton("🗑 Delete point")
        self._del_btn.setToolTip("Delete the selected point. Shortcut: Delete key.")
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._delete_selected)
        act_layout.addWidget(self._del_btn)

        self._del_outliers_btn = QPushButton("🧹 Delete outliers")
        self._del_outliers_btn.setToolTip("Remove every point currently flagged as an outlier (red ×).")
        self._del_outliers_btn.clicked.connect(self._delete_outliers)
        act_layout.addWidget(self._del_outliers_btn)
        layout.addWidget(act_box)

        # Delete/Backspace also removes the selected point.
        for seq in (QKeySequence.StandardKey.Delete, QKeySequence("Backspace")):
            QShortcut(seq, self, activated=self._delete_selected)

        # --- Quality ---
        quality_box = QGroupBox("Curve Quality")
        ql = QVBoxLayout(quality_box)
        self._quality_lbl = QLabel("Pick a curve to see quality stats.")
        self._quality_lbl.setWordWrap(True)
        ql.addWidget(self._quality_lbl)
        layout.addWidget(quality_box)

        self._debug_btn = QPushButton("🔍 Show Expert Debug Plot")
        self._debug_btn.setToolTip("Matplotlib window with the original image, extracted curves and OCR overlay.")
        self._debug_btn.clicked.connect(self.expert_debug_requested)
        layout.addWidget(self._debug_btn)

    # --- population ---------------------------------------------------------
    def populate_curves(self, curves: list[Curve]) -> None:
        """Refresh the curve dropdown, keeping the current selection if possible."""
        self._curves = [c for c in curves if c.data_xs.size > 0]
        previous = self._curve_id

        self._curve_combo.blockSignals(True)
        self._curve_combo.clear()
        for c in self._curves:
            self._curve_combo.addItem(f"{c.name}  ({c.data_xs.size} pts)", userData=c.id)
        self._curve_combo.blockSignals(False)

        if not self._curves:
            self._curve_id = None
            self._item = None
            self.plot_widget.clear()
            self._quality_lbl.setText("No extracted curves yet - run an extraction first.")
            self._set_selected_index(-1)
            return

        ids = [c.id for c in self._curves]
        target = previous if previous in ids else ids[0]
        self._curve_combo.setCurrentIndex(ids.index(target))
        self._curve_id = target
        self._rebuild_plot()

    def _on_curve_chosen(self, index: int) -> None:
        if index < 0:
            return
        self._curve_id = self._curve_combo.itemData(index)
        self._rebuild_plot()

    def _current_curve(self) -> Curve | None:
        return next((c for c in self._curves if c.id == self._curve_id), None)

    # --- plot ---------------------------------------------------------------
    def _rebuild_plot(self) -> None:
        self.plot_widget.clear()
        self._item = None
        self._set_selected_index(-1)

        curve = self._current_curve()
        if curve is None or curve.data_xs.size == 0:
            self._update_quality(None, None)
            return

        # Other curves render as faint context so you can see relative shape.
        for other in self._curves:
            if other.id == curve.id or other.data_xs.size == 0:
                continue
            pixel = np.uint8([[[other.hsv_center[0], other.hsv_center[1], other.hsv_center[2]]]])
            rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0]
            self.plot_widget.plot(
                other.data_xs, other.data_ys,
                pen=pg.mkPen(color=(int(rgb[0]), int(rgb[1]), int(rgb[2]), 70), width=1),
            )

        item = EditableCurveItem()
        item.set_points(curve.data_xs, curve.data_ys)
        item.sigPointsEdited.connect(self._on_points_edited)
        item.sigPointSelected.connect(self._set_selected_index)
        self.plot_widget.addItem(item)
        self._item = item

        self._draw_outlier_markers(curve.data_xs, curve.data_ys)
        self._update_quality(curve.data_xs, curve.data_ys)

    def _draw_outlier_markers(self, xs: np.ndarray, ys: np.ndarray) -> None:
        if xs is None or len(xs) < 5:
            return
        flags = quality.detect_outliers(xs, ys)
        if not flags.any():
            return
        marker = pg.ScatterPlotItem(
            x=np.asarray(xs)[flags], y=np.asarray(ys)[flags],
            size=18, symbol="x", pen=pg.mkPen("r", width=2), brush=None,
        )
        marker.setZValue(20)
        self.plot_widget.addItem(marker)

    # --- editing ------------------------------------------------------------
    def _on_scene_clicked(self, ev) -> None:
        if self._item is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        if ev.isAccepted():
            return  # a point handled this click (select / shift-delete)
        vb = self.plot_widget.getPlotItem().vb
        view_pos = vb.mapSceneToView(ev.scenePos())
        self._item.insert_point(float(view_pos.x()), float(view_pos.y()))

    def _on_points_edited(self) -> None:
        if self._item is None or self._curve_id is None:
            return
        xs, ys = self._item.points()
        curve = self._current_curve()
        if curve is not None:
            curve.data_xs, curve.data_ys = xs, ys
        self.points_edited.emit(self._curve_id, xs, ys)
        self._update_quality(xs, ys)
        self._redraw_outliers_only(xs, ys)

    def _redraw_outliers_only(self, xs: np.ndarray, ys: np.ndarray) -> None:
        for item in list(self.plot_widget.getPlotItem().items):
            if isinstance(item, pg.ScatterPlotItem) and item.zValue() == 20:
                self.plot_widget.removeItem(item)
        self._draw_outlier_markers(xs, ys)

    def _set_selected_index(self, index: int) -> None:
        self._del_btn.setEnabled(index >= 0)
        if index < 0 or self._item is None:
            self._sel_lbl.setText("No point selected.")
            return
        xs, ys = self._item.points()
        if index >= len(xs):
            self._sel_lbl.setText("No point selected.")
            return
        self._sel_lbl.setText(f"Selected point #{index + 1}:  x = {xs[index]:.4g}   y = {ys[index]:.4g}")

    def _delete_selected(self) -> None:
        if self._item is not None:
            self._item.delete_selected()

    def _delete_outliers(self) -> None:
        if self._item is None:
            return
        xs, ys = self._item.points()
        if len(xs) < 5:
            return
        removed = self._item.delete_mask(quality.detect_outliers(xs, ys))
        if removed:
            self._quality_lbl.setText(f"Removed {removed} outlier(s). " + self._quality_lbl.text())

    # --- quality ------------------------------------------------------------
    def _update_quality(self, xs: np.ndarray | None, ys: np.ndarray | None) -> None:
        if xs is None or len(xs) == 0:
            self._quality_lbl.setText("Pick a curve to see quality stats.")
            return
        stats = quality.curve_stats(xs, ys)
        outliers = quality.detect_outliers(xs, ys)
        x0, x1 = stats["x_range"]
        self._quality_lbl.setText(
            f"{stats['count']} points | X: {x0:.3g} to {x1:.3g} | "
            f"largest gap: {stats['largest_gap']:.3g} | {int(outliers.sum())} outlier(s) flagged"
        )
