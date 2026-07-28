"""Offscreen construction smoke test for the PyQt UI.

Not a substitute for interactive testing -- catches import/attribute/crash-
level errors in the wiring between MainWindow, ImageView, CalibrationPanel
and CurvePanel without needing a real display.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox

from digitizer.ui.curve_panel import Curve
from digitizer.ui.export_dialog import EditableCurveItem, ExportPanel
from digitizer.ui.main_window import MainWindow, Mode


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    return MainWindow()


def _load_fake_image(window, size=(100, 100)):
    h, w = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (0, 0, 0), 2)
    window._image_rgb = img
    window.image_view.set_image(img)
    return img


def test_main_window_constructs(window):
    assert window._mode == Mode.IDLE


def test_mode_switch_sets_cursor(window):
    window._set_mode(Mode.CALIBRATING)
    window._set_mode(Mode.IDLE)


def test_plotbox_preview_on_load(window):
    _load_fake_image(window)
    window._preview_plot_box()
    assert window.image_view.plotbox_preview is not None


def test_calibration_marker_drag_updates_points(window):
    _load_fake_image(window)
    for x, y in [(10, 90), (90, 90), (10, 10)]:
        window.calib_panel.add_pixel_point(x, y)
    window.image_view.set_calibration_markers(
        *zip(*window.calib_panel.pixel_points()), list(type(window.calib_panel).LABELS)
    )
    assert len(window.image_view.calib_targets) == 3
    window._on_calib_marker_moved(0, 12, 88)
    assert window.calib_panel.pixel_points()[0] == (12, 88)


def test_calibration_reset_clears_points(window):
    _load_fake_image(window)
    window.calib_panel.add_pixel_point(1, 1)
    window.calib_panel.reset()
    assert window.calib_panel.pixel_points() == []


def test_curve_color_and_seed_point_flow(window):
    _load_fake_image(window)
    curve = Curve(id=1, name="c1", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))
    window._curves_dict[1] = curve
    window.curve_panel.add_curve_card(curve)

    window._on_curve_hsv_changed(1, (60, 200, 200))
    assert curve.hsv_center == (60, 200, 200)

    window._on_curve_display_color_changed(1, (10, 20, 30))
    assert curve.display_color == (10, 20, 30)

    window._on_set_seed_requested(1)
    assert window._mode == Mode.SETTING_SEED
    window._on_image_click(5, 5)
    assert curve.seed_point == (5, 5)
    assert window._mode == Mode.IDLE
    assert 1 in window.image_view.seed_markers

    window._on_set_end_requested(1)
    assert window._mode == Mode.SETTING_END
    window._on_image_click(80, 5)
    assert curve.end_point == (80, 5)
    assert window._mode == Mode.IDLE
    assert 1 in window.image_view.end_markers


def test_exclusion_thumbnail_updates(window):
    _load_fake_image(window, size=(200, 200))
    window.image_view.set_exclusion_roi_visible(True)
    window.image_view.exclusion_roi.setPos([20, 20])
    window.image_view.exclusion_roi.setSize([40, 40])
    assert not window.curve_panel._exclusion_thumb.pixmap().isNull()


def test_editable_curve_item_drag_and_delete(app):
    item = EditableCurveItem()
    item.set_points(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    xs, ys = item.points()
    assert xs.tolist() == [0.0, 1.0, 2.0]

    fired = []
    item.sigPointsEdited.connect(lambda: fired.append(True))

    # simulate what a completed drag does to the underlying data
    item.data["pos"][1] = [1.0, 99.0]
    item.updateGraph()
    _, ys = item.points()
    assert ys[1] == 99.0

    # simulate a delete (what shift-click triggers internally)
    item._push(np.delete(item.data["pos"], 0, axis=0))
    item.sigPointsEdited.emit()
    xs, _ = item.points()
    assert xs.tolist() == [1.0, 2.0]
    assert fired == [True]


def test_export_panel_edit_mode_and_quality_panel(app):
    panel = ExportPanel()
    curve = Curve(id=1, name="c1", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))
    curve.data_xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    curve.data_ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    panel.populate_curves([curve])

    panel._on_row_selected(1)
    panel._edit_cb.setChecked(True)
    assert panel._editable_item is not None

    edited = []
    panel.points_edited.connect(lambda cid, xs, ys: edited.append((cid, xs.copy(), ys.copy())))
    panel._editable_item._push(np.delete(panel._editable_item.data["pos"], 2, axis=0))
    panel._editable_item.sigPointsEdited.emit()

    assert len(edited) == 1
    assert edited[0][0] == 1
    assert "4 points" in panel._quality_lbl.text()


def test_curve_points_edited_syncs_pixel_coords_and_flags_manual_edit(window):
    _load_fake_image(window)
    window._calibration_M = np.eye(3)
    curve = Curve(id=1, name="c1", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))
    curve.data_xs = np.array([0.0, 1.0, 2.0])
    curve.data_ys = np.array([0.0, 1.0, 2.0])
    window._curves_dict[1] = curve

    window._on_curve_points_edited(1, np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 9.0]))
    assert curve.manually_edited is True
    assert curve.pixel_xs.tolist() == [0.0, 1.0, 5.0]


def test_manually_edited_curve_warns_before_reextraction(window, monkeypatch):
    _load_fake_image(window)
    window._calibration_M = np.eye(3)
    curve = Curve(id=1, name="c1", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))
    curve.manually_edited = True
    curve.data_xs = np.array([0.0, 1.0])

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    window._run_xstep_for_curve(curve, dx=2, fill=False, reducer="mean")
    assert curve.manually_edited is True  # declined -> guard left the flag untouched

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    window._run_xstep_for_curve(curve, dx=2, fill=False, reducer="mean")
    assert curve.manually_edited is False  # accepted -> proceeded and cleared the flag
