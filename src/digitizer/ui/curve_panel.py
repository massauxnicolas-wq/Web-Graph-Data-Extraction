from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame,
)


@dataclass
class Curve:
    id: int
    name: str
    hsv_center: tuple[int, int, int]
    hsv_tol: tuple[int, int, int]
    visible: bool = True
    pixel_xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    pixel_ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    data_xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    data_ys: np.ndarray = field(default_factory=lambda: np.empty(0))


class CurveCard(QFrame):
    delete_requested = pyqtSignal(int)
    visibility_toggled = pyqtSignal(int, bool)
    name_changed = pyqtSignal(int, str)
    select_requested = pyqtSignal(int)
    
    def __init__(self, curve: Curve, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.curve_id = curve.id
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        cid = curve.id  # capture for lambdas
        
        # Color block
        self.color_lbl = QLabel()
        self.color_lbl.setFixedSize(20, 20)
        self.update_color(curve.hsv_center)
        layout.addWidget(self.color_lbl)
        
        # Name
        self.name_edit = QLineEdit(curve.name)
        self.name_edit.textChanged.connect(lambda t: self.name_changed.emit(self.curve_id, t))
        layout.addWidget(self.name_edit, 1)
        
        # Visibility
        self.vis_btn = QPushButton("👁️")
        self.vis_btn.setCheckable(True)
        self.vis_btn.setChecked(curve.visible)
        self.vis_btn.setFixedSize(30, 24)
        self.vis_btn.clicked.connect(lambda _=None, c=cid: self.visibility_toggled.emit(c, self.vis_btn.isChecked()))
        layout.addWidget(self.vis_btn)
        
        # Delete
        self.del_btn = QPushButton("❌")
        self.del_btn.setFixedSize(30, 24)
        self.del_btn.clicked.connect(lambda _=None, c=cid: self.delete_requested.emit(c))
        layout.addWidget(self.del_btn)
        
    def update_color(self, hsv: tuple[int, int, int]) -> None:
        pixel = np.uint8([[[hsv[0], hsv[1], hsv[2]]]])
        rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0]
        self.color_lbl.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #888;")

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet(
                "QFrame { background-color: #d0e8ff; border: 1px solid #0066cc; }"
                "QLineEdit { background: white; color: #222; }"
                "QLabel { color: #222; }"
                "QPushButton { color: #222; }"
            )
        else:
            self.setStyleSheet("")

    def mousePressEvent(self, event) -> None:
        self.select_requested.emit(self.curve_id)
        super().mousePressEvent(event)


class CurvePanel(QWidget):
    """HSV picker + sliders + X-Step controls + rich curve manager."""

    sample_curves_requested = pyqtSignal(bool)  # Toggle sampling mode
    hsv_changed = pyqtSignal(tuple, tuple)  # (hsv_center, hsv_tol)
    overlay_toggled = pyqtSignal(bool)
    exclusion_toggled = pyqtSignal(bool)
    auto_detect_curves_requested = pyqtSignal()
    
    # Manager signals
    extract_curve_requested = pyqtSignal(int, int, bool, str, bool, int, int, int, int)  # curve_id, dx, fill, reducer, smooth, window, poly, passes, upscale
    extract_all_requested = pyqtSignal(int, bool, str, bool, int, int, int, int)
    delete_curve_requested = pyqtSignal(int)
    select_curve_changed = pyqtSignal(int)
    curve_visibility_changed = pyqtSignal(int, bool)
    curve_name_changed = pyqtSignal(int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[int, CurveCard] = {}
        layout = QVBoxLayout(self)

        self._auto_detect_btn = QPushButton("🎨 Auto-Detect Curve Colors")
        self._auto_detect_btn.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; padding: 6px; "
            "background-color: #107c10; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0e6b0e; }"
            "QPushButton:disabled { background-color: #999; }"
        )
        self._auto_detect_btn.clicked.connect(self.auto_detect_curves_requested)
        layout.addWidget(self._auto_detect_btn)

        self._sample_btn = QPushButton("Pick Curve Colors (Multi-Sample)")
        self._sample_btn.setCheckable(True)
        self._sample_btn.clicked.connect(lambda checked: self.sample_curves_requested.emit(checked))
        layout.addWidget(self._sample_btn)

        self._sample_lbl = QLabel("Active HSV Tolerance")
        layout.addWidget(self._sample_lbl)

        slider_box = QGroupBox("Global HSV Tolerance (Applies to Extraction)")
        sl = QVBoxLayout(slider_box)
        self._h_slider, self._h_lbl = self._make_slider("H ±", 0, 90, 2, sl)
        self._s_slider, self._s_lbl = self._make_slider("S ±", 0, 255, 15, sl)
        self._v_slider, self._v_lbl = self._make_slider("V ±", 0, 255, 15, sl)
        layout.addWidget(slider_box)

        self._overlay_cb = QCheckBox("Show mask overlay")
        self._overlay_cb.setChecked(True)
        self._overlay_cb.toggled.connect(self.overlay_toggled.emit)
        layout.addWidget(self._overlay_cb)

        self._exclusion_cb = QCheckBox("Show exclusion box (ignore legend)")
        self._exclusion_cb.toggled.connect(self.exclusion_toggled.emit)
        layout.addWidget(self._exclusion_cb)

        run_box = QGroupBox("Extraction Settings")
        run_layout = QVBoxLayout(run_box)

        reducer_row = QHBoxLayout()
        reducer_row.addWidget(QLabel("Reducer:"))
        self._reducer_combo = QComboBox()
        self._reducer_combo.addItem("Mean-Y per column", userData="mean")
        self._reducer_combo.addItem("Nearest run midpoint", userData="midpoint")
        self._reducer_combo.addItem("Centroid Tracking", userData="centroid")
        self._reducer_combo.addItem("Path Trace (Steep/Loops)", userData="trace")
        reducer_row.addWidget(self._reducer_combo, 1)
        run_layout.addLayout(reducer_row)

        dx_row = QHBoxLayout()
        dx_row.addWidget(QLabel("dx (px):"))
        self._dx_spin = QSpinBox()
        self._dx_spin.setRange(1, 50)
        self._dx_spin.setValue(2)
        dx_row.addWidget(self._dx_spin)

        dx_row.addWidget(QLabel("Upscale:"))
        self._upscale_spin = QSpinBox()
        self._upscale_spin.setRange(1, 10)
        self._upscale_spin.setValue(1)
        self._upscale_spin.setToolTip("Sub-pixel precision for high-frequency curves. 1=off, 2=double res.")
        dx_row.addWidget(self._upscale_spin)
        run_layout.addLayout(dx_row)

        self._fill_cb = QCheckBox("Linear Gap Fill")
        self._fill_cb.setChecked(True)
        run_layout.addWidget(self._fill_cb)

        self._smooth_cb = QCheckBox("Smooth curve (remove marker bumps)")
        self._smooth_cb.setToolTip("Applies Savitzky-Golay smoothing to remove sharp features\nlike diamond/star/circle markers on the line.")
        run_layout.addWidget(self._smooth_cb)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Window:"))
        self._smooth_window = QSpinBox()
        self._smooth_window.setRange(5, 201)
        self._smooth_window.setValue(21)
        self._smooth_window.setSingleStep(2)
        self._smooth_window.setToolTip("Smoothing window size (must be odd). Larger = more aggressive.")
        smooth_row.addWidget(self._smooth_window)

        smooth_row.addWidget(QLabel("Poly:"))
        self._smooth_poly = QSpinBox()
        self._smooth_poly.setRange(1, 5)
        self._smooth_poly.setValue(3)
        self._smooth_poly.setToolTip("Polynomial order. Lower = smoother (1=linear, 2=quadratic, 3=cubic).")
        smooth_row.addWidget(self._smooth_poly)

        smooth_row.addWidget(QLabel("Passes:"))
        self._smooth_passes = QSpinBox()
        self._smooth_passes.setRange(1, 10)
        self._smooth_passes.setValue(1)
        self._smooth_passes.setToolTip("Number of smoothing passes. More passes = stronger smoothing.\nUse 2-5 for large markers.")
        smooth_row.addWidget(self._smooth_passes)
        run_layout.addLayout(smooth_row)

        layout.addWidget(run_box)

        list_box = QGroupBox("Curve Manager")
        ll = QVBoxLayout(list_box)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        ll.addWidget(self.scroll_area)
        
        self.extract_all_btn = QPushButton("▶️ Extract All Visible Curves")
        self.extract_all_btn.clicked.connect(self._on_extract_all)
        ll.addWidget(self.extract_all_btn)
        
        layout.addWidget(list_box, 1)

        for s in (self._h_slider, self._s_slider, self._v_slider):
            s.valueChanged.connect(self._on_slider_change)

    def _make_slider(self, label: str, lo: int, hi: int, default: int, parent_layout: QVBoxLayout) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        lbl = QLabel(f"{label} {default}")
        row.addWidget(lbl)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        slider.valueChanged.connect(lambda v, l=lbl, t=label: l.setText(f"{t} {v}"))
        row.addWidget(slider)
        parent_layout.addLayout(row)
        return slider, lbl

    def _on_slider_change(self, _: int) -> None:
        self.hsv_changed.emit((0, 0, 0), self.hsv_tol())

    def hsv_tol(self) -> tuple[int, int, int]:
        return (self._h_slider.value(), self._s_slider.value(), self._v_slider.value())

    def add_curve_card(self, curve: Curve) -> None:
        card = CurveCard(curve)
        card.delete_requested.connect(self.delete_curve_requested.emit)
        card.visibility_toggled.connect(self.curve_visibility_changed.emit)
        card.name_changed.connect(self.curve_name_changed.emit)
        card.select_requested.connect(self.select_curve_changed.emit)
        self.scroll_layout.addWidget(card)
        self._cards[curve.id] = card
        self.select_curve_changed.emit(curve.id)

    def remove_curve_card(self, curve_id: int) -> None:
        if curve_id in self._cards:
            card = self._cards.pop(curve_id)
            self.scroll_layout.removeWidget(card)
            card.deleteLater()

    def set_card_selected(self, curve_id: int) -> None:
        for cid, card in self._cards.items():
            card.set_selected(cid == curve_id)

    def _on_extract_single(self, curve_id: int) -> None:
        reducer = self._reducer_combo.currentData() or "mean"
        self.extract_curve_requested.emit(
            curve_id,
            self._dx_spin.value(),
            self._fill_cb.isChecked(),
            str(reducer),
            self._smooth_cb.isChecked(),
            self._smooth_window.value(),
            self._smooth_poly.value(),
            self._smooth_passes.value(),
            self._upscale_spin.value()
        )

    def _on_extract_all(self) -> None:
        reducer = self._reducer_combo.currentData() or "mean"
        self.extract_all_requested.emit(
            self._dx_spin.value(),
            self._fill_cb.isChecked(),
            str(reducer),
            self._smooth_cb.isChecked(),
            self._smooth_window.value(),
            self._smooth_poly.value(),
            self._smooth_passes.value(),
            self._upscale_spin.value()
        )

    def uncheck_sample_button(self) -> None:
        self._sample_btn.setChecked(False)
