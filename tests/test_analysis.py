import numpy as np
import pytest

from digitizer.core.analysis import (
    area,
    curve_metrics,
    derivative,
    initial_slope,
    secant_modulus,
)


def test_area_triangle_and_rectangle():
    assert area(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
    xs = np.linspace(0, 10, 11)
    assert area(xs, np.full(11, 5.0)) == pytest.approx(50.0)


def test_derivative_of_line_is_constant_slope():
    xs = np.linspace(0, 10, 11)
    np.testing.assert_allclose(derivative(xs, 3 * xs + 2), 3.0)


def test_initial_slope_recovers_youngs_modulus():
    xs = np.linspace(0, 10, 101)
    assert initial_slope(xs, 4 * xs) == pytest.approx(4.0)


def test_secant_modulus_and_zero_x():
    np.testing.assert_allclose(
        secant_modulus(np.array([1.0, 2.0, 4.0]), np.array([2.0, 4.0, 8.0])), [2.0, 2.0, 2.0]
    )
    s = secant_modulus(np.array([0.0, 1.0]), np.array([5.0, 3.0]))
    assert np.isnan(s[0]) and s[1] == pytest.approx(3.0)


def test_curve_metrics_peak_and_area():
    m = curve_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 0.0]))
    assert m["peak_x"] == pytest.approx(1.0)
    assert m["peak_y"] == pytest.approx(10.0)
    assert m["area"] == pytest.approx(10.0)
