import numpy as np

from digitizer.core.calibration import solve_calibration
from digitizer.core.uncertainty import point_uncertainty


def _linear_cal(data_per_pixel):
    # pixel origin bottom-left; y grows up in data. 1 pixel of Y == data_per_pixel data units.
    pixel_pts = np.array([[0, 100], [100, 100], [0, 0]], dtype=float)
    data_pts = np.array([[0, 0], [100, 0], [0, 100 * data_per_pixel]], dtype=float)
    return solve_calibration(pixel_pts, data_pts)


def test_dy_scales_with_calibration_resolution():
    fine = _linear_cal(1.0)
    coarse = _linear_cal(5.0)
    xs = np.array([10.0, 20.0, 30.0])
    ys = np.array([50.0, 50.0, 50.0])
    np.testing.assert_allclose(point_uncertainty(fine, xs, ys), 1.0)
    np.testing.assert_allclose(point_uncertainty(coarse, xs, ys), 5.0)


def test_log_axis_dy_grows_with_magnitude():
    # Y logarithmic: pixel y 100(bottom)->1, 0(top)->1000.
    pixel_pts = np.array([[0, 100], [100, 100], [0, 0]], dtype=float)
    data_pts = np.array([[1, 1], [100, 1], [1, 1000]], dtype=float)
    cal = solve_calibration(pixel_pts, data_pts, y_log=True)
    xs = np.array([0.0, 0.0])
    ys_pixel = np.array([100.0, 0.0])  # low-magnitude end, high-magnitude end
    dy = point_uncertainty(cal, xs, ys_pixel)
    assert dy[1] > dy[0]


def test_empty_input():
    cal = _linear_cal(1.0)
    assert point_uncertainty(cal, np.empty(0), np.empty(0)).size == 0
