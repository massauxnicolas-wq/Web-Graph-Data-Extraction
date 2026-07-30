from __future__ import annotations

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from digitizer.core import quality
from digitizer.ui.curve_panel import Curve


class EditableCurveItem(pg.GraphItem):
    """Draggable + shift-click-deletable curve points.

    One GraphItem (wrapping a single internal ScatterPlotItem) per curve,
    not one item per point - scales fine to hundreds of points. Pattern
    adapted from pyqtgraph's own examples/CustomGraphItem.py.
    """

    sigPointsEdited = pyqtSignal()

    def __init__(self) -> None:
        self.dragPoint = None
        self.dragOffset = None
        super().__init__()
        self.scatter.sigClicked.connect(self._on_scatter_clicked)

    def set_points(self, xs: np.ndarray, ys: np.ndarray) -> None:
        pos = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        self._push(pos)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.get("pos") if self.data else None
        if pos is None or len(pos) == 0:
            return np.empty(0), np.empty(0)
        return pos[:, 0].copy(), pos[:, 1].copy()

    def _push(self, pos: np.ndarray) -> None:
        n = len(pos)
        adj = np.column_stack([np.arange(n - 1), np.arange(1, n)]) if n > 1 else np.empty((0, 2), dtype=int)
        self.setData(
            pos=pos, adj=adj, size=8, symbol="o", pxMode=True,
            brush=pg.mkBrush(0, 120, 255, 200), pen=pg.mkPen(0, 120, 255, width=2),
        )

    def setData(self, **kwds) -> None:
        self.data = kwds
        self.updateGraph()

    def updateGraph(self) -> None:
        pg.GraphItem.setData(self, **self.data)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            pts = self.scatter.pointsAt(ev.buttonDownPos())
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            self.dragOffset = self.data["pos"][pts[0].index()] - ev.buttonDownPos()
        elif ev.isFinish():
            self.dragPoint = None
            self.sigPointsEdited.emit()
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        ind = self.dragPoint.index()
        self.data["pos"][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()

    def _on_scatter_clicked(self, _scatter, points, ev) -> None:
        if not (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier) or len(points) == 0:
            return
        idx = points[0].index()
        self._push(np.delete(self.data["pos"], idx, axis=0))
        self.sigPointsEdited.emit()


class ExportCurveRow(QFrame):
    """Single row in the export curve list with checkbox and info."""

    row_selected = pyqtSignal(int)

    def __init__(self, curve: Curve, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.curve_id = curve.id
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)

        # Color block
        self.color_lbl = QLabel()
        self.color_lbl.setFixedSize(16, 16)
        pixel = np.uint8([[[curve.hsv_center[0], curve.hsv_center[1], curve.hsv_center[2]]]])
        rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0]
        self.color_lbl.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #888;"
        )
        layout.addWidget(self.color_lbl)

        self.name_lbl = QLabel(curve.name)
        layout.addWidget(self.name_lbl, 1)

        pts = curve.data_xs.size
        self.pts_lbl = QLabel(f"{pts} pts" if pts > 0 else "Not extracted")
        layout.addWidget(self.pts_lbl)

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet("QFrame { background-color: #d0e8ff; border: 1px solid #0066cc; }")
        else:
            self.setStyleSheet("")

    def mousePressEvent(self, event) -> None:
        self.row_selected.emit(self.curve_id)
        super().mousePressEvent(event)


class ExportPanel(QWidget):
    """Comprehensive export panel with preview plot and multiple export options."""

    export_csv_active = pyqtSignal()
    export_csv_wide = pyqtSignal()
    export_csv_checked = pyqtSignal(list)  # list of curve_ids
    export_json = pyqtSignal()
    copy_clipboard = pyqtSignal()
    refresh_requested = pyqtSignal()
    expert_debug_requested = pyqtSignal()
    points_edited = pyqtSignal(int, object, object)  # curve_id, new xs, new ys

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[int, ExportCurveRow] = {}
        self._curves: list[Curve] = []
        self._selected_curve_id: int | None = None
        self._editable_item: EditableCurveItem | None = None
        self._editable_curve_id: int | None = None

        layout = QVBoxLayout(self)

        # --- Preview Plot ---
        preview_box = QGroupBox("Curve Preview")
        preview_layout = QVBoxLayout(preview_box)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "X")
        self.plot_widget.setLabel("left", "Y")
        self.plot_widget.setMinimumHeight(200)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)
        preview_layout.addWidget(self.plot_widget)

        self._edit_cb = QCheckBox("Enable point editing for selected curve")
        self._edit_cb.setToolTip("Select a curve below first.")
        self._edit_cb.toggled.connect(self._refresh_plot)
        preview_layout.addWidget(self._edit_cb)

        self._edit_hint_lbl = QLabel("Drag = move point   |   Shift+Click = delete point   |   Click empty space = add point")
        self._edit_hint_lbl.setStyleSheet("QLabel { font-size: 11px; color: #666; }")
        self._edit_hint_lbl.setWordWrap(True)
        preview_layout.addWidget(self._edit_hint_lbl)

        layout.addWidget(preview_box, 2)

        # --- Quality analysis ---
        quality_box = QGroupBox("Curve Quality")
        quality_layout = QVBoxLayout(quality_box)
        self._quality_lbl = QLabel("Select a curve below to see quality stats.")
        self._quality_lbl.setWordWrap(True)
        quality_layout.addWidget(self._quality_lbl)
        layout.addWidget(quality_box)

        # --- Curve List ---
        list_box = QGroupBox("Curves to Export")
        ll = QVBoxLayout(list_box)

        top_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all)
        top_row.addWidget(self._select_all_btn)
        self._deselect_all_btn = QPushButton("Deselect All")
        self._deselect_all_btn.clicked.connect(self._deselect_all)
        top_row.addWidget(self._deselect_all_btn)
        self._refresh_btn = QPushButton("🔄 Refresh Preview")
        self._refresh_btn.clicked.connect(self._on_refresh)
        top_row.addWidget(self._refresh_btn)
        ll.addLayout(top_row)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        ll.addWidget(self.scroll_area)
        layout.addWidget(list_box, 1)

        # --- Export Buttons ---
        export_box = QGroupBox("Export Actions")
        el = QVBoxLayout(export_box)

        b_csv_checked = QPushButton("📁 Export Selected Curves → Individual CSVs")
        b_csv_checked.setToolTip("Exports each checked curve to its own CSV file in a chosen folder.")
        b_csv_checked.clicked.connect(self._on_export_checked)
        el.addWidget(b_csv_checked)

        b_csv_wide = QPushButton("📊 Export All Curves → Wide CSV")
        b_csv_wide.setToolTip("Exports all checked curves side-by-side into a single wide CSV.")
        b_csv_wide.clicked.connect(self.export_csv_wide)
        el.addWidget(b_csv_wide)

        b_json = QPushButton("💾 Export Full Session → JSON")
        b_json.setToolTip("Saves the entire session including calibration, HSV values, and all curve data.")
        b_json.clicked.connect(self.export_json)
        el.addWidget(b_json)

        b_clip = QPushButton("📋 Copy Selected Curve → Clipboard (TSV)")
        b_clip.setToolTip("Copies the currently selected curve as tab-separated values for pasting into Excel.")
        b_clip.clicked.connect(self.copy_clipboard)
        el.addWidget(b_clip)

        b_expert_debug = QPushButton("🔍 Show Expert Debug Plot")
        b_expert_debug.setToolTip("Opens a Matplotlib window with the original image, extracted curves, and OCR text overlay for debugging.")
        b_expert_debug.clicked.connect(self.expert_debug_requested)
        b_expert_debug.setStyleSheet("background-color: #6a0dad; color: white; font-weight: bold;")
        el.addWidget(b_expert_debug)

        layout.addWidget(export_box)

    def populate_curves(self, curves: list[Curve]) -> None:
        """Clear and rebuild the curve list and preview plot."""
        # Clear old rows
        for row in self._rows.values():
            self.scroll_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        self._curves = list(curves)
        existing_ids = {c.id for c in self._curves}
        if self._selected_curve_id not in existing_ids:
            self._selected_curve_id = None

        for curve in curves:
            row = ExportCurveRow(curve)
            row.checkbox.toggled.connect(self._refresh_plot)
            row.row_selected.connect(self._on_row_selected)
            row.set_selected(curve.id == self._selected_curve_id)
            self.scroll_layout.addWidget(row)
            self._rows[curve.id] = row

        sel_curve = next((c for c in self._curves if c.id == self._selected_curve_id), None)
        self._update_quality_panel(
            sel_curve.data_xs if sel_curve is not None else None,
            sel_curve.data_ys if sel_curve is not None else None,
        )
        self._refresh_plot()

    def _on_row_selected(self, curve_id: int) -> None:
        self._selected_curve_id = curve_id
        for cid, row in self._rows.items():
            row.set_selected(cid == curve_id)
        curve = next((c for c in self._curves if c.id == curve_id), None)
        self._update_quality_panel(
            curve.data_xs if curve is not None else None,
            curve.data_ys if curve is not None else None,
        )
        self._refresh_plot()

    def _update_quality_panel(self, xs: np.ndarray | None, ys: np.ndarray | None) -> None:
        if xs is None or len(xs) == 0:
            self._quality_lbl.setText("Select a curve with extracted points to see quality stats.")
            return
        stats = quality.curve_stats(xs, ys)
        outliers = quality.detect_outliers(xs, ys)
        x0, x1 = stats["x_range"]
        self._quality_lbl.setText(
            f"{stats['count']} points | X: {x0:.3g} to {x1:.3g} | "
            f"largest gap: {stats['largest_gap']:.3g} | {int(outliers.sum())} outlier(s) flagged"
        )

    def _on_points_edited(self) -> None:
        if self._editable_item is None or self._editable_curve_id is None:
            return
        xs, ys = self._editable_item.points()
        self.points_edited.emit(self._editable_curve_id, xs, ys)
        self._update_quality_panel(xs, ys)

    def _on_scene_clicked(self, ev) -> None:
        if not self._edit_cb.isChecked() or ev.button() != Qt.MouseButton.LeftButton:
            return
        if ev.isAccepted():
            return  # a point (drag or shift-click delete) already handled this click
        if self._editable_item is None or self._editable_curve_id is None:
            return
        vb = self.plot_widget.getPlotItem().vb
        view_pos = vb.mapSceneToView(ev.scenePos())
        x, y = float(view_pos.x()), float(view_pos.y())
        xs, ys = self._editable_item.points()
        idx = int(np.searchsorted(xs, x))
        xs = np.insert(xs, idx, x)
        ys = np.insert(ys, idx, y)
        self._editable_item.set_points(xs, ys)
        self._on_points_edited()

    def _refresh_plot(self) -> None:
        self.plot_widget.clear()
        self._editable_item = None
        self._editable_curve_id = None
        checked = set(self.checked_curve_ids())
        edit_on = self._edit_cb.isChecked()

        for curve in self._curves:
            if curve.id not in checked:
                continue
            if curve.data_xs.size == 0:
                continue
            pixel = np.uint8([[[curve.hsv_center[0], curve.hsv_center[1], curve.hsv_center[2]]]])
            rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0]
            color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            if edit_on and curve.id == self._selected_curve_id:
                item = EditableCurveItem()
                item.set_points(curve.data_xs, curve.data_ys)
                item.sigPointsEdited.connect(self._on_points_edited)
                self.plot_widget.addItem(item)
                self._editable_item = item
                self._editable_curve_id = curve.id
            else:
                pen = pg.mkPen(color=color, width=2)
                self.plot_widget.plot(
                    curve.data_xs, curve.data_ys,
                    pen=pen, name=curve.name,
                )

        sel_curve = next((c for c in self._curves if c.id == self._selected_curve_id), None)
        if sel_curve is not None and sel_curve.id in checked and sel_curve.data_xs.size >= 5:
            flags = quality.detect_outliers(sel_curve.data_xs, sel_curve.data_ys)
            if flags.any():
                outlier_item = pg.ScatterPlotItem(
                    x=sel_curve.data_xs[flags], y=sel_curve.data_ys[flags],
                    size=14, symbol="x", pen=pg.mkPen("r", width=2), brush=None,
                )
                outlier_item.setZValue(20)
                self.plot_widget.addItem(outlier_item)

        self.plot_widget.addLegend()

    def checked_curve_ids(self) -> list[int]:
        """Return IDs of all checked curves."""
        return [cid for cid, row in self._rows.items() if row.is_checked()]

    def _select_all(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(True)

    def _deselect_all(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(False)

    def _on_refresh(self) -> None:
        self.refresh_requested.emit()

    def _on_export_checked(self) -> None:
        ids = self.checked_curve_ids()
        if ids:
            self.export_csv_checked.emit(ids)
