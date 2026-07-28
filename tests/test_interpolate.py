import numpy as np
import pytest

from digitizer.core.interpolate import fill_gaps, fill_gaps_parametric, polynomial_best_fit, uniform_grid


def test_fill_gaps_linear_interpolation():
    xs = np.array([0, 10])
    ys = np.array([0, 100])
    out = fill_gaps(xs, ys, x_grid=[0, 5, 10])
    np.testing.assert_allclose(out, [0, 50, 100])


def test_fill_gaps_out_of_range_is_nan():
    out = fill_gaps([0, 10], [0, 100], x_grid=[-5, 15])
    assert np.isnan(out).all()


def test_fill_gaps_sorts_unsorted_input():
    xs = [10, 0, 5]
    ys = [100, 0, 50]
    out = fill_gaps(xs, ys, x_grid=[2])
    np.testing.assert_allclose(out, [20])


def test_fill_gaps_requires_two_points():
    with pytest.raises(ValueError, match=">= 2"):
        fill_gaps([1], [1], x_grid=[1])


def test_fill_gaps_dedups_duplicate_x():
    xs = [0, 0, 10]
    ys = [0, 5, 100]
    out = fill_gaps(xs, ys, x_grid=[5])
    np.testing.assert_allclose(out, [50])


def test_polynomial_best_fit_recovers_a_parabola():
    xs = np.linspace(0, 10, 50)
    ys = 2 * xs**2 - 3 * xs + 1
    out_x, out_y = polynomial_best_fit(xs, ys, degree=2)
    expected = 2 * out_x**2 - 3 * out_x + 1
    np.testing.assert_allclose(out_y, expected, atol=1e-6)


def test_polynomial_best_fit_smooths_noisy_line():
    xs = np.linspace(0, 10, 100)
    ys = 3 * xs + 5 + np.array([(-1) ** i * 0.5 for i in range(100)])  # small alternating noise
    out_x, out_y = polynomial_best_fit(xs, ys, degree=1)
    expected = 3 * out_x + 5
    assert np.max(np.abs(out_y - expected)) < 0.5


def test_polynomial_best_fit_rejects_too_few_points():
    with pytest.raises(ValueError, match=">="):
        polynomial_best_fit([0, 1], [0, 1], degree=3)


def test_uniform_grid_spans_min_to_max():
    grid = uniform_grid([3, 1, 7], step=2.0)
    assert grid[0] == 1
    assert grid[-1] >= 7


def test_fill_gaps_parametric_resamples_evenly_spaced():
    xs = np.array([0, 1, 2, 3])
    ys = np.zeros(4)
    out_x, out_y = fill_gaps_parametric(xs, ys, step=1.0)
    np.testing.assert_allclose(out_x, [0, 1, 2, 3])
    np.testing.assert_allclose(out_y, [0, 0, 0, 0])


def test_fill_gaps_parametric_handles_vertical_line():
    # x constant, y increasing: a plain x-interpolation would fail here.
    xs = np.array([5.0, 5.0, 5.0])
    ys = np.array([0.0, 1.0, 2.0])
    out_x, out_y = fill_gaps_parametric(xs, ys, step=1.0)
    np.testing.assert_allclose(out_x, [5.0, 5.0, 5.0])
    np.testing.assert_allclose(out_y, [0.0, 1.0, 2.0])


def test_fill_gaps_parametric_drops_duplicate_points():
    xs = np.array([0.0, 0.0, 1.0])
    ys = np.array([0.0, 0.0, 0.0])
    out_x, out_y = fill_gaps_parametric(xs, ys, step=1.0)
    assert len(out_x) == 2


def test_fill_gaps_parametric_too_few_points_returns_input():
    xs = np.array([1.0])
    ys = np.array([2.0])
    out_x, out_y = fill_gaps_parametric(xs, ys)
    np.testing.assert_allclose(out_x, xs)
    np.testing.assert_allclose(out_y, ys)
