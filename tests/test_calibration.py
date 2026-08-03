import numpy as np
import pytest

from digitizer.core.calibration import (
    Calibration,
    affine_from_points,
    calibration_error,
    round_trip_error,
    solve_calibration,
)
from digitizer.core.transform import pixel_to_data


def test_affine_recovers_known_transform():
    # Pixel origin at (10, 500), x-axis maps 1:1, y-axis flips (image y grows down).
    pixel_pts = np.array([[10, 500], [510, 500], [10, 0]])
    data_pts = np.array([[0, 0], [500, 0], [0, 500]])
    M = affine_from_points(pixel_pts, data_pts)

    homo = np.hstack([pixel_pts, np.ones((3, 1))])
    recovered = homo @ M.T
    np.testing.assert_allclose(recovered[:, :2], data_pts, atol=1e-9)


def test_affine_rejects_collinear_points():
    pixel_pts = np.array([[0, 0], [10, 0], [20, 0]])
    data_pts = np.array([[0, 0], [1, 0], [2, 0]])
    with pytest.raises(ValueError, match="collinear"):
        affine_from_points(pixel_pts, data_pts)


def test_affine_rejects_too_few_points():
    with pytest.raises(ValueError, match=">= 3"):
        affine_from_points(np.array([[0, 0], [1, 1]]), np.array([[0, 0], [1, 1]]))


def test_affine_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        affine_from_points(np.array([[0, 0], [1, 0], [0, 1]]), np.array([[0, 0], [1, 0]]))


def test_four_point_homography_is_exact():
    pixel_pts = np.array([[0, 100], [100, 100], [100, 0], [0, 0]], dtype=float)
    data_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    M = affine_from_points(pixel_pts, data_pts)
    err = round_trip_error(M, pixel_pts, data_pts)
    assert err < 1e-6


def test_round_trip_error_zero_for_exact_fit():
    pixel_pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    data_pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    M = affine_from_points(pixel_pts, data_pts)
    assert round_trip_error(M, pixel_pts, data_pts) == pytest.approx(0.0, abs=1e-9)


def test_solve_calibration_linear_matches_affine():
    pixel_pts = np.array([[10, 500], [510, 500], [10, 0]], dtype=float)
    data_pts = np.array([[0, 0], [500, 0], [0, 500]], dtype=float)
    cal = solve_calibration(pixel_pts, data_pts)
    assert isinstance(cal, Calibration)
    assert not cal.x_log and not cal.y_log
    np.testing.assert_allclose(cal.M, affine_from_points(pixel_pts, data_pts), atol=1e-9)


def test_solve_calibration_log_axis_round_trips_a_decade():
    pixel_pts = np.array([[100, 50], [300, 50], [100, 0]], dtype=float)
    data_pts = np.array([[1.0, 0.0], [1000.0, 0.0], [1.0, 10.0]], dtype=float)
    cal = solve_calibration(pixel_pts, data_pts, x_log=True)
    assert cal.x_log and not cal.y_log
    # Calibration points must map back to their data values.
    out = pixel_to_data(pixel_pts, cal)
    np.testing.assert_allclose(out, data_pts, atol=1e-6)
    assert calibration_error(cal, pixel_pts, data_pts) == pytest.approx(0.0, abs=1e-9)


def test_solve_calibration_log_axis_rejects_nonpositive():
    pixel_pts = np.array([[100, 50], [300, 50], [100, 0]], dtype=float)
    data_pts = np.array([[0.0, 0.0], [1000.0, 0.0], [1.0, 10.0]], dtype=float)  # X-min = 0
    with pytest.raises(ValueError, match="logarithmic X-axis"):
        solve_calibration(pixel_pts, data_pts, x_log=True)
