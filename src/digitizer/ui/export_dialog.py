from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from digitizer.core.export import ExportOptions
from digitizer.ui.curve_panel import Curve


class ExportCurveRow(QFrame):
    """Single row in the export curve list: checkbox, colour, name, point count."""

    def __init__(self, curve: Curve, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.curve_id = curve.id
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)

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


class ExportPanel(QWidget):
    """Two jobs, kept separate: export data (CSV/TSV, with options) and save the project (JSON)."""

    export_csv_requested = pyqtSignal()
    copy_tsv_requested = pyqtSignal()
    save_project_requested = pyqtSignal()
    refresh_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[int, ExportCurveRow] = {}
        self._curves: list[Curve] = []

        layout = QVBoxLayout(self)

        # --- Curve list ---
        list_box = QGroupBox("Curves")
        ll = QVBoxLayout(list_box)
        top_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all)
        top_row.addWidget(self._select_all_btn)
        self._deselect_all_btn = QPushButton("Deselect All")
        self._deselect_all_btn.clicked.connect(self._deselect_all)
        top_row.addWidget(self._deselect_all_btn)
        self._refresh_btn = QPushButton("🔄 Refresh")
        self._refresh_btn.clicked.connect(self.refresh_requested)
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

        # --- Options ---
        opt_box = QGroupBox("Options")
        form = QFormLayout(opt_box)

        self._layout_combo = QComboBox()
        self._layout_combo.addItem("Wide (one file, name_x/name_y columns)", userData="wide")
        self._layout_combo.addItem("Individual files (one per curve)", userData="individual")
        form.addRow("Layout:", self._layout_combo)

        self._grid_spin = QDoubleSpinBox()
        self._grid_spin.setRange(0.0, 1e9)
        self._grid_spin.setDecimals(4)
        self._grid_spin.setValue(0.0)
        self._grid_spin.setToolTip("Resample to this uniform X step. 0 = keep raw extracted points.")
        form.addRow("Resample X step:", self._grid_spin)

        units_hint = "e.g. MPa, GPa, psi, %, ratio, mm, in, C, K, F — leave blank to skip"
        self._x_from, self._x_to = QLineEdit(), QLineEdit()
        self._x_from.setPlaceholderText("from")
        self._x_to.setPlaceholderText("to")
        form.addRow("Convert X:", self._unit_row(self._x_from, self._x_to, units_hint))
        self._y_from, self._y_to = QLineEdit(), QLineEdit()
        self._y_from.setPlaceholderText("from")
        self._y_to.setPlaceholderText("to")
        form.addRow("Convert Y:", self._unit_row(self._y_from, self._y_to, units_hint))

        self._uncertainty_cb = QCheckBox("Include per-point uncertainty (±dy) columns")
        form.addRow("", self._uncertainty_cb)
        layout.addWidget(opt_box)

        # --- Export data ---
        export_box = QGroupBox("Export data")
        el = QVBoxLayout(export_box)
        b_csv = QPushButton("📁 Export CSV (selected curves)")
        b_csv.setToolTip("Wide layout → one file. Individual → a file per curve (choose a folder).")
        b_csv.clicked.connect(self.export_csv_requested)
        el.addWidget(b_csv)
        b_clip = QPushButton("📋 Copy active curve → clipboard (TSV)")
        b_clip.clicked.connect(self.copy_tsv_requested)
        el.addWidget(b_clip)
        layout.addWidget(export_box)

        # --- Project ---
        proj_box = QGroupBox("Project")
        pl = QVBoxLayout(proj_box)
        b_json = QPushButton("💾 Save session (JSON)")
        b_json.setToolTip("Saves calibration, HSV settings and all curve data for the record.")
        b_json.clicked.connect(self.save_project_requested)
        pl.addWidget(b_json)
        layout.addWidget(proj_box)

    @staticmethod
    def _unit_row(w_from: QLineEdit, w_to: QLineEdit, tooltip: str) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        w_from.setToolTip(tooltip)
        w_to.setToolTip(tooltip)
        row.addWidget(w_from)
        row.addWidget(QLabel("→"))
        row.addWidget(w_to)
        return w

    # --- state read by main_window -----------------------------------------
    def export_options(self) -> ExportOptions:
        step = self._grid_spin.value()
        return ExportOptions(
            layout=str(self._layout_combo.currentData() or "wide"),
            x_grid_step=step if step > 0 else None,
            x_unit=self._unit_pair(self._x_from, self._x_to),
            y_unit=self._unit_pair(self._y_from, self._y_to),
            include_uncertainty=self._uncertainty_cb.isChecked(),
        )

    @staticmethod
    def _unit_pair(w_from: QLineEdit, w_to: QLineEdit) -> tuple[str, str] | None:
        a, b = w_from.text().strip(), w_to.text().strip()
        return (a, b) if a and b else None

    def populate_curves(self, curves: list[Curve]) -> None:
        for row in self._rows.values():
            self.scroll_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self._curves = list(curves)
        for curve in curves:
            row = ExportCurveRow(curve)
            self.scroll_layout.addWidget(row)
            self._rows[curve.id] = row

    def checked_curve_ids(self) -> list[int]:
        return [cid for cid, row in self._rows.items() if row.is_checked()]

    def _select_all(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(True)

    def _deselect_all(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(False)
