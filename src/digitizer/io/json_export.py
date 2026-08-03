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
    x_log: bool = False,
    y_log: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "digitizer/0.2",
        "image_path": image_path,
        "calibration": {
            "matrix": calibration_matrix.tolist() if calibration_matrix is not None else None,
            "points": calibration_points or [],
            "x_log": x_log,
            "y_log": y_log,
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
    seed_point: tuple[float, float] | None = None,
    end_point: tuple[float, float] | None = None,
    display_color: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "hsv_center": list(hsv_center),
        "hsv_tol": list(hsv_tol),
        "seed_point": list(seed_point) if seed_point is not None else None,
        "end_point": list(end_point) if end_point is not None else None,
        "display_color": list(display_color) if display_color is not None else None,
        "pixel": {"x": np.asarray(pixel_xs).tolist(), "y": np.asarray(pixel_ys).tolist()},
        "data": {"x": np.asarray(data_xs).tolist(), "y": np.asarray(data_ys).tolist()},
    }
