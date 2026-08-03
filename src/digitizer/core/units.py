"""Small built-in unit conversions for export. Qt-free, no external dependency.

Scale families convert through a base unit by a factor; temperature is affine and handled
separately. Cross-family or unknown conversions raise ValueError. Extend the dicts to add units.
"""
from __future__ import annotations

import numpy as np

# Scale families: unit -> multiplier to the family's base unit.
_STRESS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
           "psi": 6894.757293, "ksi": 6894757.293, "bar": 1e5}
_STRAIN = {"ratio": 1.0, "mm/mm": 1.0, "%": 0.01, "percent": 0.01}
_LENGTH = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "in": 0.0254, "inch": 0.0254}
_ACCEL = {"m/s^2": 1.0, "m/s2": 1.0, "g": 9.80665}
_SCALE_FAMILIES = (_STRESS, _STRAIN, _LENGTH, _ACCEL)

# Temperature (affine): symbol sets that mean the same unit.
_TEMP_C = {"C", "°C", "degC"}
_TEMP_K = {"K", "kelvin"}
_TEMP_F = {"F", "°F", "degF"}
_TEMP = _TEMP_C | _TEMP_K | _TEMP_F


def _scale_family(unit: str) -> dict | None:
    for fam in _SCALE_FAMILIES:
        if unit in fam:
            return fam
    return None


def _to_kelvin(v: np.ndarray, unit: str) -> np.ndarray:
    if unit in _TEMP_C:
        return v + 273.15
    if unit in _TEMP_K:
        return v
    return (v - 32.0) * 5.0 / 9.0 + 273.15  # Fahrenheit


def _from_kelvin(v: np.ndarray, unit: str) -> np.ndarray:
    if unit in _TEMP_C:
        return v - 273.15
    if unit in _TEMP_K:
        return v
    return (v - 273.15) * 9.0 / 5.0 + 32.0  # Fahrenheit


def convert(values, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert an array of values between two units of the same family."""
    values = np.asarray(values, dtype=float)
    if from_unit == to_unit:
        return values

    if from_unit in _TEMP or to_unit in _TEMP:
        if not (from_unit in _TEMP and to_unit in _TEMP):
            raise ValueError(f"cannot convert between {from_unit!r} and {to_unit!r}")
        return _from_kelvin(_to_kelvin(values, from_unit), to_unit)

    fam_from, fam_to = _scale_family(from_unit), _scale_family(to_unit)
    if fam_from is None or fam_to is None:
        raise ValueError(f"unknown unit: {from_unit!r} or {to_unit!r}")
    if fam_from is not fam_to:
        raise ValueError(f"cannot convert between {from_unit!r} and {to_unit!r} (different families)")
    return values * (fam_from[from_unit] / fam_to[to_unit])
