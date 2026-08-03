import numpy as np
import pytest

from digitizer.core.units import convert


def test_stress_scale():
    np.testing.assert_allclose(convert([1.0], "GPa", "MPa"), [1000.0])
    np.testing.assert_allclose(convert([1.0], "MPa", "Pa"), [1e6])
    np.testing.assert_allclose(convert([1.0], "ksi", "psi"), [1000.0], rtol=1e-6)


def test_strain_percent():
    np.testing.assert_allclose(convert([5.0], "%", "ratio"), [0.05])


def test_length_and_accel():
    np.testing.assert_allclose(convert([1.0], "in", "mm"), [25.4])
    np.testing.assert_allclose(convert([1.0], "g", "m/s^2"), [9.80665])


def test_temperature_affine():
    np.testing.assert_allclose(convert([0.0], "C", "K"), [273.15])
    np.testing.assert_allclose(convert([32.0], "F", "C"), [0.0], atol=1e-9)
    np.testing.assert_allclose(convert([100.0], "C", "F"), [212.0], atol=1e-9)


def test_identity_passthrough():
    np.testing.assert_allclose(convert([1, 2, 3], "MPa", "MPa"), [1, 2, 3])


def test_cross_family_raises():
    with pytest.raises(ValueError):
        convert([1.0], "MPa", "mm")


def test_temp_to_scale_raises():
    with pytest.raises(ValueError):
        convert([1.0], "C", "MPa")


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        convert([1.0], "furlong", "m")
