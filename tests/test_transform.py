import numpy as np

from digitizer.core.calibration import Calibration, affine_from_points, solve_calibration
from digitizer.core.transform import data_to_pixel, pixel_to_data


def _known_calibration():
    pixel_pts = np.array([[10, 500], [510, 500], [10, 0]], dtype=float)
    data_pts = np.array([[0, 0], [500, 0], [0, 500]], dtype=float)
    return Calibration(affine_from_points(pixel_pts, data_pts)), pixel_pts, data_pts


def test_pixel_to_data_matches_calibration_points():
    cal, pixel_pts, data_pts = _known_calibration()
    out = pixel_to_data(pixel_pts, cal)
    np.testing.assert_allclose(out, data_pts, atol=1e-9)


def test_data_to_pixel_is_inverse_of_pixel_to_data():
    cal, pixel_pts, _ = _known_calibration()
    data = pixel_to_data(pixel_pts, cal)
    recovered_pixels = data_to_pixel(data, cal)
    np.testing.assert_allclose(recovered_pixels, pixel_pts, atol=1e-6)


def test_pixel_to_data_handles_1d_input():
    cal, pixel_pts, data_pts = _known_calibration()
    out = pixel_to_data(pixel_pts[0], cal)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], data_pts[0], atol=1e-9)


def test_pixel_to_data_empty_input():
    cal, _, _ = _known_calibration()
    out = pixel_to_data(np.empty((0, 2)), cal)
    assert out.shape == (0, 2)


def test_log_axis_round_trip():
    # X-axis spans a decade: pixels 100..300 map to data 1..1000 logarithmically.
    pixel_pts = np.array([[100, 50], [300, 50], [100, 0]], dtype=float)
    data_pts = np.array([[1.0, 0.0], [1000.0, 0.0], [1.0, 10.0]], dtype=float)
    cal = solve_calibration(pixel_pts, data_pts, x_log=True)

    # The midpoint pixel (x=200) is halfway in log space -> 10^1.5 ~= 31.62.
    mid = pixel_to_data(np.array([[200.0, 50.0]]), cal)
    np.testing.assert_allclose(mid[0, 0], 10 ** 1.5, rtol=1e-6)

    # Round-trip data -> pixel -> data recovers the calibration points.
    recovered = pixel_to_data(data_to_pixel(data_pts, cal), cal)
    np.testing.assert_allclose(recovered, data_pts, atol=1e-6)
