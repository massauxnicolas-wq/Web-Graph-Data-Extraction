"""Offscreen construction smoke test for the PyQt UI.

Not a substitute for interactive testing -- catches import/attribute/crash-
level errors in the wiring between MainWindow, ImageView, CalibrationPanel
and CurvePanel without needing a real display.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import pyqtgraph as pg
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from digitizer.ui.curve_panel import Curve
from digitizer.ui.edit_panel import EditPanel
from digitizer.ui.editable_curve_item import EditableCurveItem
from digitizer.ui.export_dialog import ExportPanel
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


def test_default_calibration_points_are_placed_and_cleared_by_manual_capture(window):
    _load_fake_image(window, size=(200, 200))  # draws a rect from (10,10) to (190,190)
    window._preview_plot_box()
    window._place_default_calibration_points()

    pts = window.calib_panel.pixel_points()
    assert len(pts) == 3
    origin, x_max, y_max = pts
    assert origin[1] == x_max[1]      # origin and X-max share the bottom edge
    assert origin[0] == y_max[0]      # origin and Y-max share the left edge
    assert x_max[0] > origin[0] and y_max[1] < origin[1]
    assert len(window.image_view.calib_targets) == 3

    window._enter_calibration_mode()   # manual clicking starts from scratch
    assert window.calib_panel.pixel_points() == []
    assert window.image_view.calib_targets == []


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

    # select + delete (the path the Delete button / Delete key uses)
    item.select_point(0)
    assert item.selected_index == 0
    assert item.delete_selected() is True
    xs, _ = item.points()
    assert xs.tolist() == [1.0, 2.0]
    assert item.selected_index == -1
    assert fired == [True]


def test_editable_curve_item_insert_keeps_x_order(app):
    item = EditableCurveItem()
    item.set_points(np.array([0.0, 2.0, 4.0]), np.array([0.0, 2.0, 4.0]))
    item.insert_point(3.0, 30.0)
    xs, ys = item.points()
    assert xs.tolist() == [0.0, 2.0, 3.0, 4.0]
    assert ys.tolist() == [0.0, 2.0, 30.0, 4.0]


def test_editable_curve_item_delete_mask_removes_outliers(app):
    item = EditableCurveItem()
    item.set_points(np.arange(5.0), np.array([0.0, 1.0, 99.0, 3.0, 4.0]))
    removed = item.delete_mask(np.array([False, False, True, False, False]))
    assert removed == 1
    _, ys = item.points()
    assert 99.0 not in ys.tolist()


def test_edit_panel_lists_only_extracted_curves(app):
    panel = EditPanel()
    extracted = Curve(id=1, name="c1", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))
    extracted.pixel_xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    extracted.pixel_ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    bare = Curve(id=2, name="c2", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))

    panel.populate_curves([extracted, bare])
    assert panel.current_curve_id() == 1
    assert panel._curve_combo.count() == 1  # the un-extracted curve is not offered

    panel.populate_curves([bare])
    assert "No extracted curves" in panel._quality_lbl.text()


def _curve_on_canvas(window, points=5):
    curve = Curve(id=1, name="c1", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))
    curve.pixel_xs = np.arange(float(points))
    curve.pixel_ys = np.arange(float(points))
    window._curves_dict[1] = curve
    window.image_view.set_curve_points(1, curve.pixel_xs, curve.pixel_ys, curve.hsv_center, True, None)
    return curve


def test_extraction_updates_canvas_while_edit_mode_is_on(window):
    """Regression: a previous revision made set_curve_points a silent no-op
    for the curve being edited, so extraction results never reached the canvas."""
    _load_fake_image(window)
    curve = _curve_on_canvas(window)
    window._edit_curve_id = 1
    window._apply_canvas_editing(True)
    assert window.image_view.editing_curve_id() == 1

    # new points arriving (as extraction would deliver them) must be visible
    new_xs, new_ys = np.array([10.0, 20.0, 30.0]), np.array([11.0, 21.0, 31.0])
    window.image_view.set_curve_points(1, new_xs, new_ys, curve.hsv_center, True, None)
    shown_xs, shown_ys = window.image_view.edited_points()
    assert shown_xs.tolist() == new_xs.tolist()
    assert shown_ys.tolist() == new_ys.tolist()


def test_canvas_edit_hides_other_curves_and_restores_them(window):
    _load_fake_image(window)
    _curve_on_canvas(window)
    other = Curve(id=2, name="c2", hsv_center=(60, 255, 255), hsv_tol=(2, 15, 15))
    other.pixel_xs = np.arange(5.0)
    other.pixel_ys = np.full(5, 9.0)
    window._curves_dict[2] = other
    window.image_view.set_curve_points(2, other.pixel_xs, other.pixel_ys, other.hsv_center, True, None)

    window._edit_curve_id = 1
    window._apply_canvas_editing(True)
    assert window.image_view.curve_scatters[2].isVisible() is False

    window._apply_canvas_editing(False)
    assert window.image_view.editing_curve_id() is None
    assert window.image_view.curve_scatters[2].isVisible() is True


def test_canvas_edit_syncs_back_to_data_coords(window):
    _load_fake_image(window)
    window._calibration_M = np.eye(3)
    curve = _curve_on_canvas(window)

    window._on_curve_canvas_edited(1, np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 9.0]))
    assert curve.manually_edited is True
    assert curve.pixel_xs.tolist() == [0.0, 1.0, 5.0]
    assert curve.data_ys.tolist() == [0.0, 1.0, 9.0]


def test_canvas_delete_and_insert_reach_the_curve(window):
    _load_fake_image(window)
    window._calibration_M = np.eye(3)
    curve = _curve_on_canvas(window)
    window._edit_curve_id = 1
    window._apply_canvas_editing(True)

    window.image_view._editable_item.select_point(2)
    window._on_delete_point()
    assert curve.pixel_xs.tolist() == [0.0, 1.0, 3.0, 4.0]

    window.image_view._editable_item.insert_point(2.0, 20.0)
    assert curve.pixel_xs.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert curve.pixel_ys.tolist() == [0.0, 1.0, 20.0, 3.0, 4.0]


def test_image_opacity_round_trip(window):
    _load_fake_image(window)
    window.image_view.set_image_opacity(0.35)
    assert window.image_view.image_item.opacity() == pytest.approx(0.35)


# --- real mouse interaction ------------------------------------------------
# These drive Qt's actual event path rather than calling methods directly,
# which is the only way to catch "the canvas stopped responding" regressions.

def _drag(widget, start, end, steps=4):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for i in range(1, steps + 1):
        QTest.mouseMove(
            widget,
            QPoint(
                start.x() + (end.x() - start.x()) * i // steps,
                start.y() + (end.y() - start.y()) * i // steps,
            ),
        )
        QApplication.processEvents()
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    QApplication.processEvents()


@pytest.mark.parametrize("edit_mode", [False, True])
def test_canvas_still_pans_with_a_real_drag(window, edit_mode):
    """The canvas must keep panning whether or not point editing is active."""
    _load_fake_image(window, size=(200, 200))
    _curve_on_canvas(window)
    window._edit_curve_id = 1
    window._apply_canvas_editing(edit_mode)

    # wide enough that the canvas gets real estate next to the side panel,
    # otherwise the drag lands outside the (tiny) viewport and proves nothing
    window.resize(1600, 700)
    window.show()
    QApplication.processEvents()

    viewport = window.image_view.viewport()
    assert viewport.width() > 300, "canvas too small for this test to mean anything"

    vb = window.image_view.view
    before = list(vb.viewRange()[0])
    # start well away from the curve (which lies along y == x near the origin)
    _drag(viewport, QPoint(400, 300), QPoint(470, 300))
    after = list(vb.viewRange()[0])

    assert before != after, f"canvas did not pan (edit_mode={edit_mode})"


def test_real_drag_on_a_point_moves_it_without_panning(window):
    _load_fake_image(window, size=(200, 200))
    window._calibration_M = np.eye(3)
    curve = Curve(id=1, name="c1", hsv_center=(0, 255, 255), hsv_tol=(2, 15, 15))
    curve.pixel_xs = np.array([50.0, 100.0, 150.0])
    curve.pixel_ys = np.array([100.0, 100.0, 100.0])
    window._curves_dict[1] = curve
    window.image_view.set_curve_points(1, curve.pixel_xs, curve.pixel_ys, curve.hsv_center, True, None)
    window._edit_curve_id = 1
    window._apply_canvas_editing(True)

    window.resize(1600, 700)
    window.show()
    QApplication.processEvents()

    iv = window.image_view
    vb = iv.view
    target = iv.mapFromScene(vb.mapViewToScene(pg.Point(100.0, 100.0)))
    view_before = list(vb.viewRange()[0])

    _drag(iv.viewport(), target, QPoint(target.x(), target.y() - 50), steps=5)

    assert curve.pixel_ys[1] != 100.0, "dragging a point did not move it"
    assert curve.pixel_ys[0] == 100.0 and curve.pixel_ys[2] == 100.0, "wrong point moved"
    assert curve.manually_edited is True
    assert list(vb.viewRange()[0]) == view_before, "dragging a point should not pan the view"


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
