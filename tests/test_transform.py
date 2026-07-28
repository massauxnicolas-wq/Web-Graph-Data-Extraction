import numpy as np

from digitizer.core.calibration import affine_from_points
from digitizer.core.transform import data_to_pixel, pixel_to_data


def _known_matrix():
    pixel_pts = np.array([[10, 500], [510, 500], [10, 0]], dtype=float)
    data_pts = np.array([[0, 0], [500, 0], [0, 500]], dtype=float)
    return affine_from_points(pixel_pts, data_pts), pixel_pts, data_pts


def test_pixel_to_data_matches_calibration_points():
    M, pixel_pts, data_pts = _known_matrix()
    out = pixel_to_data(pixel_pts, M)
    np.testing.assert_allclose(out, data_pts, atol=1e-9)


def test_data_to_pixel_is_inverse_of_pixel_to_data():
    M, pixel_pts, _ = _known_matrix()
    data = pixel_to_data(pixel_pts, M)
    recovered_pixels = data_to_pixel(data, M)
    np.testing.assert_allclose(recovered_pixels, pixel_pts, atol=1e-6)


def test_pixel_to_data_handles_1d_input():
    M, pixel_pts, data_pts = _known_matrix()
    out = pixel_to_data(pixel_pts[0], M)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], data_pts[0], atol=1e-9)


def test_pixel_to_data_empty_input():
    M, _, _ = _known_matrix()
    out = pixel_to_data(np.empty((0, 2)), M)
    assert out.shape == (0, 2)
