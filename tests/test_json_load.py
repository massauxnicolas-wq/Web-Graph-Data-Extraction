import json

import numpy as np
import pytest

from digitizer.io.json_export import build_payload, serialize_curve, write_payload
from digitizer.io.json_load import load_payload


def test_round_trip_preserves_calibration_and_curves(tmp_path):
    M = np.array([[2.0, 0, 1], [0, 3.0, 4], [0, 0, 1]])
    curve = serialize_curve(
        "c1", hsv_center=(1, 2, 3), hsv_tol=(4, 5, 6),
        pixel_xs=[0, 1], pixel_ys=[2, 3], data_xs=[0.0, 1.0], data_ys=[0.0, 2.0],
        seed_point=(1.5, 2.5), end_point=(9.0, 8.0), display_color=(10, 20, 30),
    )
    payload = build_payload("img.png", M, [{"pixel": [0, 0], "data": [0, 0]}],
                            [curve], x_log=True, y_log=False)
    path = tmp_path / "session.json"
    write_payload(path, payload)

    loaded = load_payload(path)
    assert loaded.image_path == "img.png"
    assert loaded.calibration is not None
    np.testing.assert_allclose(loaded.calibration.M, M)
    assert loaded.calibration.x_log is True
    assert loaded.calibration.y_log is False

    c = loaded.curves[0]
    assert c["name"] == "c1"
    assert c["seed_point"] == [1.5, 2.5]
    assert c["end_point"] == [9.0, 8.0]
    assert c["display_color"] == [10, 20, 30]
    assert c["data"] == {"x": [0.0, 1.0], "y": [0.0, 2.0]}


def test_missing_matrix_yields_no_calibration(tmp_path):
    payload = build_payload(None, None, None, curves=[])
    path = tmp_path / "s.json"
    write_payload(path, payload)
    assert load_payload(path).calibration is None


def test_legacy_0_1_loads_with_linear_defaults(tmp_path):
    # A pre-log-axis file: schema 0.1, no x_log/y_log keys.
    legacy = {
        "schema": "digitizer/0.1",
        "image_path": "old.png",
        "calibration": {"matrix": np.eye(3).tolist(), "points": []},
        "curves": [],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = load_payload(path)
    assert loaded.calibration is not None
    assert loaded.calibration.x_log is False
    assert loaded.calibration.y_log is False


def test_rejects_non_digitizer_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something/else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a digitizer session"):
        load_payload(path)
