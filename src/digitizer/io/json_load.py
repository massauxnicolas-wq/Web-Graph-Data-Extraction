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
