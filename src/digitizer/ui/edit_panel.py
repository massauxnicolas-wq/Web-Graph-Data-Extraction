from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from digitizer.core import quality
from digitizer.ui.curve_panel import Curve


class EditPanel(QWidget):
    """Controls for editing curve points *on the main canvas*.

    This panel owns no plot of its own - the canvas is the single editing
    surface, so what you edit is always shown over the original chart.
    """

    edit_mode_toggled = pyqtSignal(bool)
    curve_selected = pyqtSignal(int)
    image_opacity_changed = pyqtSignal(int)
    delete_point_requested = pyqtSignal()
    delete_outliers_requested = pyqtSignal()
    refresh_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curves: list[Curve] = []

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

        # --- Edit mode ---
        self._edit_cb = QCheckBox("✏️ Edit points on canvas")
        self._edit_cb.setStyleSheet("QCheckBox { font-size: 13px; font-weight: bold; padding: 4px; }")
        self._edit_cb.setToolTip(
            "While on, the selected curve's points become draggable on the main canvas\n"
            "and the other curves are hidden. While off, the canvas behaves normally."
        )
        self._edit_cb.toggled.connect(self.edit_mode_toggled)
        layout.addWidget(self._edit_cb)

        hint = QLabel(
            "Drag a point to move it  ·  Click a point to select it  ·  "
            "Click empty space to add a point"
        )
        hint.setStyleSheet("QLabel { font-size: 11px; color: #666; }")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # --- Image opacity ---
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Image opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setToolTip("Fade the original chart to see the extracted points more clearly.")
        self._opacity_slider.valueChanged.connect(self.image_opacity_changed)
        op_row.addWidget(self._opacity_slider)
        layout.addLayout(op_row)

        # --- Point actions ---
        act_box = QGroupBox("Points")
        act_layout = QVBoxLayout(act_box)

        self._sel_lbl = QLabel("No point selected.")
        self._sel_lbl.setWordWrap(True)
        act_layout.addWidget(self._sel_lbl)

        self._del_btn = QPushButton("🗑 Delete point")
        self._del_btn.setToolTip("Delete the selected point. Shortcut: Delete key.")
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self.delete_point_requested)
        act_layout.addWidget(self._del_btn)

        self._del_outliers_btn = QPushButton("🧹 Delete outliers")
        self._del_outliers_btn.setToolTip("Remove every point flagged as an outlier by the quality check.")
        self._del_outliers_btn.clicked.connect(self.delete_outliers_requested)
        act_layout.addWidget(self._del_outliers_btn)

        layout.addWidget(act_box)

        for seq in (QKeySequence.StandardKey.Delete, QKeySequence("Backspace")):
            QShortcut(seq, self, activated=self.delete_point_requested.emit)

        # --- Quality ---
        quality_box = QGroupBox("Curve Quality")
        ql = QVBoxLayout(quality_box)
        self._quality_lbl = QLabel("Pick a curve to see quality stats.")
        self._quality_lbl.setWordWrap(True)
        ql.addWidget(self._quality_lbl)
        layout.addWidget(quality_box)

        layout.addStretch(1)

    # --- population ---------------------------------------------------------
    def populate_curves(self, curves: list[Curve], selected_id: int | None = None) -> None:
        """Refresh the curve dropdown, keeping the current selection if possible."""
        self._curves = [c for c in curves if c.pixel_xs.size > 0]
        previous = selected_id if selected_id is not None else self.current_curve_id()

        self._curve_combo.blockSignals(True)
        self._curve_combo.clear()
        for c in self._curves:
            self._curve_combo.addItem(f"{c.name}  ({c.pixel_xs.size} pts)", userData=c.id)
        self._curve_combo.blockSignals(False)

        if not self._curves:
            self._quality_lbl.setText("No extracted curves yet - run an extraction first.")
            self.set_selected_point(-1, None, None)
            return

        ids = [c.id for c in self._curves]
        target = previous if previous in ids else ids[0]
        self._curve_combo.blockSignals(True)
        self._curve_combo.setCurrentIndex(ids.index(target))
        self._curve_combo.blockSignals(False)
        self.curve_selected.emit(target)

    def current_curve_id(self) -> int | None:
        data = self._curve_combo.currentData()
        return int(data) if data is not None else None

    def is_edit_mode(self) -> bool:
        return self._edit_cb.isChecked()

    def _on_curve_chosen(self, index: int) -> None:
        if index < 0:
            return
        self.curve_selected.emit(self._curve_combo.itemData(index))

    # --- readouts -----------------------------------------------------------
    def set_selected_point(self, index: int, x: float | None, y: float | None) -> None:
        self._del_btn.setEnabled(index >= 0)
        if index < 0 or x is None or y is None:
            self._sel_lbl.setText("No point selected.")
            return
        self._sel_lbl.setText(f"Selected point #{index + 1}:  x = {x:.4g}   y = {y:.4g}")

    def update_quality(self, xs: np.ndarray | None, ys: np.ndarray | None) -> None:
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
