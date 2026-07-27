from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_payload(
    image_path: str | None,
    calibration_matrix: np.ndarray | None,
    calibration_points: list[dict[str, Any]] | None,
    curves: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "digitizer/0.1",
        "image_path": image_path,
        "calibration": {
            "matrix": calibration_matrix.tolist() if calibration_matrix is not None else None,
            "points": calibration_points or [],
        },
        "curves": curves,
    }


def write_payload(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def serialize_curve(
    name: str,
    hsv_center: tuple[int, int, int],
    hsv_tol: tuple[int, int, int],
    pixel_xs: np.ndarray,
    pixel_ys: np.ndarray,
    data_xs: np.ndarray,
    data_ys: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": name,
        "hsv_center": list(hsv_center),
        "hsv_tol": list(hsv_tol),
        "pixel": {"x": np.asarray(pixel_xs).tolist(), "y": np.asarray(pixel_ys).tolist()},
        "data": {"x": np.asarray(data_xs).tolist(), "y": np.asarray(data_ys).tolist()},
    }
