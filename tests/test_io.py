import csv
import json

import numpy as np
import pytest

from digitizer.core.image_io import load_image
from digitizer.io.csv_export import write_curve_csv, write_curves_wide
from digitizer.io.json_export import build_payload, serialize_curve, write_payload


def test_write_curve_csv_roundtrip(tmp_path):
    path = tmp_path / "curve.csv"
    write_curve_csv(path, xs=[0.0, 1.0, 2.0], ys=[10.0, 20.0, 30.0])
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["x", "y"]
    assert rows[1:] == [["0.0", "10.0"], ["1.0", "20.0"], ["2.0", "30.0"]]


def test_write_curve_csv_rejects_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        write_curve_csv(tmp_path / "curve.csv", xs=[0, 1], ys=[0])


def test_write_curves_wide_pads_uneven_lengths(tmp_path):
    path = tmp_path / "wide.csv"
    curves = [
        ("a", np.array([1, 2, 3]), np.array([10, 20, 30])),
        ("b", np.array([1]), np.array([100])),
    ]
    write_curves_wide(path, curves)
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["a_x", "a_y", "b_x", "b_y"]
    assert rows[1] == ["1", "10", "1", "100"]
    assert rows[2] == ["2", "20", "", ""]


def test_write_curves_wide_rejects_empty_list(tmp_path):
    with pytest.raises(ValueError, match="no curves"):
        write_curves_wide(tmp_path / "wide.csv", [])


def test_build_payload_serializes_matrix():
    M = np.eye(3)
    payload = build_payload("img.png", M, [{"x": 1, "y": 2}], curves=[])
    assert payload["schema"] == "digitizer/0.2"
    assert payload["calibration"]["matrix"] == M.tolist()


def test_build_payload_handles_missing_calibration():
    payload = build_payload(None, None, None, curves=[])
    assert payload["calibration"]["matrix"] is None
    assert payload["calibration"]["points"] == []


def test_serialize_curve_shape():
    curve = serialize_curve(
        "c1", hsv_center=(1, 2, 3), hsv_tol=(4, 5, 6),
        pixel_xs=[0, 1], pixel_ys=[0, 1], data_xs=[0.0, 1.0], data_ys=[0.0, 2.0],
    )
    assert curve["name"] == "c1"
    assert curve["pixel"] == {"x": [0, 1], "y": [0, 1]}
    assert curve["data"] == {"x": [0.0, 1.0], "y": [0.0, 2.0]}


def test_write_payload_roundtrip(tmp_path):
    path = tmp_path / "session.json"
    payload = build_payload("img.png", np.eye(3), [], curves=[])
    write_payload(path, payload)
    assert json.loads(path.read_text())["schema"] == "digitizer/0.2"


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")
