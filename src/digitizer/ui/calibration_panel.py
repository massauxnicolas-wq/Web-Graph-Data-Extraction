from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CalibrationPanel(QWidget):
    """Three-point calibration UI.

    User clicks the chart's origin, X-axis max, and Y-axis max in that order, then
    enters the corresponding plot-axis values. The panel derives the three data
    points internally so the column-confusion failure mode (filling Dy where Dx
    was meant) is impossible.
    """

    start_capture = pyqtSignal()
    cancel_capture = pyqtSignal()
    solve_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    auto_calibrate_requested = pyqtSignal()
    debug_overlay_requested = pyqtSignal()

    LABELS = ("Origin", "X-axis max", "Y-axis max", "Top-Right (Optional)")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target_count = 3
        self._captured: list[tuple[float, float]] = []
        layout = QVBoxLayout(self)

        # --- Auto-Calibrate Section ---
        auto_box = QGroupBox("Automatic Calibration")
        auto_layout = QVBoxLayout(auto_box)
        auto_desc = QLabel(
            "Uses OCR to automatically detect the plot grid, axis values, and labels."
        )
        auto_desc.setWordWrap(True)
        auto_desc.setStyleSheet("QLabel { font-size: 12px; color: #555; }")
        auto_layout.addWidget(auto_desc)

        self._auto_btn = QPushButton("⚡ Auto-Calibrate (OCR)")
        self._auto_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; padding: 8px; "
            "background-color: #0078d4; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #106ebe; }"
            "QPushButton:disabled { background-color: #999; }"
        )
        self._auto_btn.clicked.connect(self.auto_calibrate_requested)

        self._debug_btn = QPushButton("\U0001f50d Debug")
        self._debug_btn.setStyleSheet(
            "QPushButton { font-size: 12px; padding: 8px; "
            "background-color: #444; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #666; }"
        )
        self._debug_btn.clicked.connect(self.debug_overlay_requested)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._auto_btn, stretch=3)
        btn_row.addWidget(self._debug_btn, stretch=1)
        auto_layout.addLayout(btn_row)

        self._auto_status = QLabel("")
        self._auto_status.setWordWrap(True)
        self._auto_status.setStyleSheet("QLabel { font-size: 11px; color: #666; }")
        auto_layout.addWidget(self._auto_status)
        layout.addWidget(auto_box)

        separator = QLabel("— OR calibrate manually —")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("QLabel { font-size: 12px; color: #999; margin: 6px 0; }")
        layout.addWidget(separator)

        # --- Manual Calibration Section ---
        instructions = QLabel(
            "<h3>Manual Calibration</h3>"
            "<ol>"
            "<li>Click <b>'Capture'</b> to begin.</li>"
            "<li>Click points on the graph: <b>Origin</b>, <b>X-max</b>, <b>Y-max</b>. If 4-point is enabled, click <b>Top-Right</b> last.</li>"
            "<li>Enter the exact plot-axis data values below.</li>"
            "<li>Press <b>'Solve calibration'</b> to finish.</li>"
            "</ol>"
        )
        instructions.setStyleSheet("QLabel { font-size: 13px; line-height: 1.5; }")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self._four_point_cb = QCheckBox("Enable 4-Point Calibration (Skewed Image)")
        self._four_point_cb.toggled.connect(self._on_four_point_toggled)
        layout.addWidget(self._four_point_cb)

        self._capture_btn = QPushButton("Capture calibration points")
        self._capture_btn.setCheckable(True)
        self._capture_btn.clicked.connect(self._on_capture_toggle)
        layout.addWidget(self._capture_btn)

        self._status = QLabel(f"Click points: 0/{self._target_count}")
        layout.addWidget(self._status)

        # Captured pixel coordinates (display only)
        pix_box = QGroupBox("Captured pixels")
        pix_grid = QGridLayout(pix_box)
        pix_grid.addWidget(QLabel("Point"), 0, 0)
        pix_grid.addWidget(QLabel("Pixel (x, y)"), 0, 1)
        self._pixel_labels: list[QLabel] = []
        for row, name in enumerate(self.LABELS, start=1):
            pix_grid.addWidget(QLabel(name), row, 0)
            lbl = QLabel("—")
            self._pixel_labels.append(lbl)
            pix_grid.addWidget(lbl, row, 1)
        layout.addWidget(pix_box)

        # Plot-axis values
        val_box = QGroupBox("Plot-axis values")
        form = QFormLayout(val_box)

        self._x_max = QDoubleSpinBox()
        self._x_max.setRange(-1e9, 1e9)
        self._x_max.setDecimals(6)
        self._x_max.setValue(1.0)
        self._x_max.setToolTip("Plot value on the X-axis at the second clicked point")
        form.addRow("X-axis max value:", self._x_max)

        self._y_max = QDoubleSpinBox()
        self._y_max.setRange(-1e9, 1e9)
        self._y_max.setDecimals(6)
        self._y_max.setValue(1.0)
        self._y_max.setToolTip("Plot value on the Y-axis at the third clicked point")
        form.addRow("Y-axis max value:", self._y_max)

        self._advanced_cb = QCheckBox("Override origin (advanced)")
        self._advanced_cb.toggled.connect(self._on_advanced_toggle)
        form.addRow(self._advanced_cb)

        self._x_origin = QDoubleSpinBox()
        self._x_origin.setRange(-1e9, 1e9)
        self._x_origin.setDecimals(6)
        self._x_origin.setValue(0.0)
        self._x_origin.setEnabled(False)
        form.addRow("X-axis origin value:", self._x_origin)

        self._y_origin = QDoubleSpinBox()
        self._y_origin.setRange(-1e9, 1e9)
        self._y_origin.setDecimals(6)
        self._y_origin.setValue(0.0)
        self._y_origin.setEnabled(False)
        form.addRow("Y-axis origin value:", self._y_origin)

        layout.addWidget(val_box)

        btns = QHBoxLayout()
        self._solve_btn = QPushButton("Solve calibration")
        self._solve_btn.clicked.connect(self.solve_requested)
        btns.addWidget(self._solve_btn)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.clicked.connect(self._on_reset)
        btns.addWidget(self._reset_btn)
        layout.addLayout(btns)

        self._error_label = QLabel("Calibration: not solved")
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)
        layout.addStretch(1)

    def _on_advanced_toggle(self, on: bool) -> None:
        self._x_origin.setEnabled(on)
        self._y_origin.setEnabled(on)

    def _on_capture_toggle(self, checked: bool) -> None:
        if checked:
            self._capture_btn.setText("Capturing... (cancel)")
            self.start_capture.emit()
        else:
            self._capture_btn.setText("Capture calibration points")
            self.cancel_capture.emit()

    def _on_reset(self) -> None:
        self._captured.clear()
        for lbl in self._pixel_labels:
            lbl.setText("—")
        self._status.setText(f"Click points: 0/{self._target_count}")
        self._error_label.setText("Calibration: not solved")
        self.reset_requested.emit()

    def add_pixel_point(self, x: float, y: float) -> bool:
        if len(self._captured) >= self._target_count:
            return True
        self._captured.append((x, y))
        self._update_ui_for_points()
        if len(self._captured) >= self._target_count:
            self._capture_btn.setChecked(False)
            self._capture_btn.setText("Capture calibration points")
            return True
        return False

    def _on_four_point_toggled(self, checked: bool) -> None:
        self._target_count = 4 if checked else 3
        if len(self._captured) > self._target_count:
            self._captured = self._captured[:self._target_count]
        self._update_ui_for_points()

    def set_pixel_points(self, pts: list[tuple[float, float]]) -> None:
        self._captured = list(pts)
        self._update_ui_for_points()

    def _update_ui_for_points(self):
        for i, lbl in enumerate(self._pixel_labels):
            if i < len(self._captured):
                lbl.setText(f"({self._captured[i][0]:.1f}, {self._captured[i][1]:.1f})")
            else:
                lbl.setText("—")
        self._status.setText(f"Click points: {len(self._captured)}/{self._target_count}")

    def pixel_points(self) -> list[tuple[float, float]]:
        return list(self._captured)

    def data_points(self) -> list[tuple[float, float]]:
        """Derive the data points from the X/Y range inputs.

        Origin is `(x_origin, y_origin)`; X-axis-max is `(x_max, y_origin)`;
        Y-axis-max is `(x_origin, y_max)`.
        4th point is `(x_max, y_max)`.
        """
        x0 = self._x_origin.value()
        y0 = self._y_origin.value()
        pts = [
            (x0, y0),
            (self._x_max.value(), y0),
            (x0, self._y_max.value()),
        ]
        if len(self._captured) == 4:
            pts.append((self._x_max.value(), self._y_max.value()))
        return pts

    def has_enough_pixel_points(self) -> bool:
        return len(self._captured) >= 3

    def set_solved_status(self, ok: bool, message: str) -> None:
        prefix = "Solved" if ok else "Error"
        self._error_label.setText(f"Calibration: {prefix} — {message}")

    # --- Auto-Calibrate helpers ---
    def set_auto_status(self, text: str) -> None:
        self._auto_status.setText(text)

    def set_auto_enabled(self, enabled: bool) -> None:
        self._auto_btn.setEnabled(enabled)
        if not enabled:
            self._auto_btn.setText("⏳ Detecting...")
        else:
            self._auto_btn.setText("⚡ Auto-Calibrate (OCR)")

    def set_axis_values(self, x_origin: float, y_origin: float, x_max: float, y_max: float) -> None:
        """Programmatically set the axis value spinboxes."""
        self._advanced_cb.setChecked(True)  # Enable origin override
        self._x_origin.setValue(x_origin)
        self._y_origin.setValue(y_origin)
        self._x_max.setValue(x_max)
        self._y_max.setValue(y_max)
