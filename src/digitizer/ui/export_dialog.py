from __future__ import annotations

import cv2
import numpy as np
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
    """Plain export panel: pick curves, write files.

    Preview, quality analysis and point editing all live in the Editing
    tab - this one stays deliberately simple.
    """

    export_csv_active = pyqtSignal()
    export_csv_wide = pyqtSignal()
    export_csv_checked = pyqtSignal(list)  # list of curve_ids
    export_json = pyqtSignal()
    copy_clipboard = pyqtSignal()
    refresh_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[int, ExportCurveRow] = {}
        self._curves: list[Curve] = []

        layout = QVBoxLayout(self)

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

        # --- Export Buttons ---
        export_box = QGroupBox("Export")
        el = QVBoxLayout(export_box)

        b_csv_checked = QPushButton("📁 Selected curves → individual CSVs")
        b_csv_checked.setToolTip("Exports each checked curve to its own CSV file in a chosen folder.")
        b_csv_checked.clicked.connect(self._on_export_checked)
        el.addWidget(b_csv_checked)

        b_csv_wide = QPushButton("📊 All curves → one wide CSV")
        b_csv_wide.setToolTip("Exports all curves side-by-side into a single wide CSV.")
        b_csv_wide.clicked.connect(self.export_csv_wide)
        el.addWidget(b_csv_wide)

        b_json = QPushButton("💾 Full session → JSON")
        b_json.setToolTip("Saves calibration, HSV settings and all curve data.")
        b_json.clicked.connect(self.export_json)
        el.addWidget(b_json)

        b_clip = QPushButton("📋 Active curve → clipboard (TSV)")
        b_clip.setToolTip("Copies the selected curve as tab-separated values for pasting into Excel.")
        b_clip.clicked.connect(self.copy_clipboard)
        el.addWidget(b_clip)

        layout.addWidget(export_box)

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
        """Return IDs of all checked curves."""
        return [cid for cid, row in self._rows.items() if row.is_checked()]

    def _select_all(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(True)

    def _deselect_all(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(False)

    def _on_export_checked(self) -> None:
        ids = self.checked_curve_ids()
        if ids:
            self.export_csv_checked.emit(ids)
