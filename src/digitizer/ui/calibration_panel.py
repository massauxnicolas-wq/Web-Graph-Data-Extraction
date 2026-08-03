from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
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
    points_changed = pyqtSignal(list)  # emitted when captured points are changed other than by a fresh click

    LABELS = ("Origin", "X-axis max", "Y-axis max", "Top-Right (Optional)")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target_count = 3
        self._captured: list[tuple[float, float]] = []
        layout = QVBoxLayout(self)

        # --- Manual Calibration Section ---
        manual_box = QGroupBox("Manual Calibration")
        manual_layout = QVBoxLayout(manual_box)

        instructions = QLabel(
            "Click <b>Capture</b>, then click Origin, X-max, Y-max on the graph "
            "(in that order), enter the axis values below, and press <b>Solve</b>."
        )
        instructions.setStyleSheet("QLabel { font-size: 12px; color: #555; }")
        instructions.setWordWrap(True)
        manual_layout.addWidget(instructions)

        self._four_point_cb = QCheckBox("Enable 4-Point Calibration (Skewed Image)")
        self._four_point_cb.toggled.connect(self._on_four_point_toggled)
        manual_layout.addWidget(self._four_point_cb)

        self._capture_btn = QPushButton("Capture calibration points")
        self._capture_btn.setCheckable(True)
        self._capture_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; padding: 8px; }"
        )
        self._capture_btn.clicked.connect(self._on_capture_toggle)
        manual_layout.addWidget(self._capture_btn)

        self._status = QLabel(f"Click points: 0/{self._target_count}")
        manual_layout.addWidget(self._status)

        layout.addWidget(manual_box)

        # Captured pixel coordinates (display only, collapsed by default)
        pix_box = QGroupBox("Captured pixels")
        pix_box.setCheckable(True)
        pix_box.setChecked(False)
        pix_outer = QVBoxLayout(pix_box)
        pix_content = QWidget()
        pix_content.setVisible(False)
        pix_box.toggled.connect(pix_content.setVisible)
        pix_grid = QGridLayout(pix_content)
        pix_grid.addWidget(QLabel("Point"), 0, 0)
        pix_grid.addWidget(QLabel("Pixel (x, y)"), 0, 1)
        self._pixel_labels: list[QLabel] = []
        for row, name in enumerate(self.LABELS, start=1):
            pix_grid.addWidget(QLabel(name), row, 0)
            lbl = QLabel("—")
            self._pixel_labels.append(lbl)
            pix_grid.addWidget(lbl, row, 1)
        pix_outer.addWidget(pix_content)
        layout.addWidget(pix_box)

        # Plot-axis values — min and max always visible for both axes
        val_box = QGroupBox("Plot-axis values")
        form = QFormLayout(val_box)

        self._x_origin = QDoubleSpinBox()
        self._x_origin.setRange(-1e9, 1e9)
        self._x_origin.setDecimals(6)
        self._x_origin.setValue(0.0)
        self._x_origin.setToolTip("Plot value on the X-axis at the Origin point")
        form.addRow("X-axis min value:", self._x_origin)

        self._x_max = QDoubleSpinBox()
        self._x_max.setRange(-1e9, 1e9)
        self._x_max.setDecimals(6)
        self._x_max.setValue(1.0)
        self._x_max.setToolTip("Plot value on the X-axis at the X-max point")
        form.addRow("X-axis max value:", self._x_max)

        self._y_origin = QDoubleSpinBox()
        self._y_origin.setRange(-1e9, 1e9)
        self._y_origin.setDecimals(6)
        self._y_origin.setValue(0.0)
        self._y_origin.setToolTip("Plot value on the Y-axis at the Origin point")
        form.addRow("Y-axis min value:", self._y_origin)

        self._y_max = QDoubleSpinBox()
        self._y_max.setRange(-1e9, 1e9)
        self._y_max.setDecimals(6)
        self._y_max.setValue(1.0)
        self._y_max.setToolTip("Plot value on the Y-axis at the Y-max point")
        form.addRow("Y-axis max value:", self._y_max)

        self._x_log_cb = QCheckBox("Logarithmic X-axis")
        self._x_log_cb.setToolTip("Treat the X-axis as base-10 logarithmic. X values must be > 0.")
        form.addRow("", self._x_log_cb)

        self._y_log_cb = QCheckBox("Logarithmic Y-axis")
        self._y_log_cb.setToolTip("Treat the Y-axis as base-10 logarithmic. Y values must be > 0.")
        form.addRow("", self._y_log_cb)

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

    def _on_capture_toggle(self, checked: bool) -> None:
        if checked:
            self._capture_btn.setText("Capturing... (cancel)")
            self.start_capture.emit()
        else:
            self._capture_btn.setText("Capture calibration points")
            self.cancel_capture.emit()

    def _on_reset(self) -> None:
        self.reset()
        self.reset_requested.emit()

    def reset(self) -> None:
        """Clear captured points and status. Does not emit reset_requested."""
        self._captured.clear()
        for lbl in self._pixel_labels:
            lbl.setText("—")
        self._status.setText(f"Click points: 0/{self._target_count}")
        self._error_label.setText("Calibration: not solved")

    def x_log(self) -> bool:
        return self._x_log_cb.isChecked()

    def y_log(self) -> bool:
        return self._y_log_cb.isChecked()

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
            self.points_changed.emit(list(self._captured))
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
        if len(self._captured) < self._target_count:
            next_label = self.LABELS[len(self._captured)]
            self._status.setText(f"Click point {len(self._captured) + 1}/{self._target_count}: {next_label}")
        else:
            self._status.setText("All points captured. Drag a marker to adjust, or Solve.")

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
