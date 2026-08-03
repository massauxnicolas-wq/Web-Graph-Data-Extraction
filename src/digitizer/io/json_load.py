"""Read a session JSON back into structured, Qt-free data (the inverse of json_export).

This is the backend half of project reload: it parses and validates the file and reconstructs
a Calibration. Rebuilding UI objects (curves, markers, widgets) from the returned data is the
frontend's job — this layer stays UI-agnostic so FastAPI and the React app can share it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from digitizer.core.calibration import Calibration


@dataclass
class LoadedSession:
    image_path: str | None
    calibration: Calibration | None
    calibration_points: list[dict[str, Any]] = field(default_factory=list)
    curves: list[dict[str, Any]] = field(default_factory=list)


def load_payload(path: str | Path) -> LoadedSession:
    """Parse a digitizer session file. Accepts schemas digitizer/0.1 and digitizer/0.2."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema", "")
    if not isinstance(schema, str) or not schema.startswith("digitizer/"):
        raise ValueError(f"not a digitizer session file (schema={schema!r})")

    calib = payload.get("calibration") or {}
    matrix = calib.get("matrix")
    calibration = None
    if matrix is not None:
        calibration = Calibration(
            M=np.array(matrix, dtype=float),
            x_log=bool(calib.get("x_log", False)),   # absent in 0.1 -> linear
            y_log=bool(calib.get("y_log", False)),
        )

    return LoadedSession(
        image_path=payload.get("image_path"),
        calibration=calibration,
        calibration_points=calib.get("points", []),
        curves=payload.get("curves", []),
    )


def load_profile(path: str | Path):
    """Parse an extraction profile (digitizer-profile/0.1) into a core.profile.Profile."""
    from digitizer.core.pipeline import ExtractionParams
    from digitizer.core.profile import Profile, ProfileCurve

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema", "") != "digitizer-profile/0.1":
        raise ValueError(f"not a digitizer profile (schema={payload.get('schema')!r})")

    cal = payload.get("calibration", {})
    curves = [
        ProfileCurve(
            name=c["name"],
            hsv_center=tuple(c["hsv_center"]),
            hsv_tol=tuple(c["hsv_tol"]),
            seed=tuple(c["seed"]) if c.get("seed") is not None else None,
            end=tuple(c["end"]) if c.get("end") is not None else None,
        )
        for c in payload.get("curves", [])
    ]
    return Profile(
        calibration_pixel_pts=[tuple(p) for p in cal.get("pixel", [])],
        calibration_data_pts=[tuple(p) for p in cal.get("data", [])],
        x_log=bool(cal.get("x_log", False)),
        y_log=bool(cal.get("y_log", False)),
        params=ExtractionParams(**payload.get("params", {})),
        curves=curves,
    )
