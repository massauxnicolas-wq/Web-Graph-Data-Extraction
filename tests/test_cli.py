import cv2
import numpy as np

from digitizer.cli import main
from digitizer.core.pipeline import ExtractionParams
from digitizer.core.profile import Profile, ProfileCurve
from digitizer.io.json_export import write_profile


def _red_chart(path):
    bgr = np.full((100, 100, 3), 255, dtype=np.uint8)  # white
    bgr[45:55, 10:90] = (0, 0, 255)                     # red band (BGR)
    cv2.imwrite(str(path), bgr)


def _profile(path):
    write_profile(path, Profile(
        calibration_pixel_pts=[(10.0, 90.0), (90.0, 90.0), (10.0, 10.0)],
        calibration_data_pts=[(0.0, 0.0), (80.0, 0.0), (0.0, 100.0)],
        params=ExtractionParams(dx=5, reducer="mean"),
        curves=[ProfileCurve("red", (0, 255, 255), (10, 80, 80))],
    ))


def test_cli_single_image_wide(tmp_path):
    img, prof, out = tmp_path / "chart.png", tmp_path / "p.json", tmp_path / "data.csv"
    _red_chart(img)
    _profile(prof)
    assert main([str(img), "--profile", str(prof), "--out", str(out)]) == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith("red_x")


def test_cli_folder_batch(tmp_path):
    charts = tmp_path / "charts"
    charts.mkdir()
    _red_chart(charts / "a.png")
    _red_chart(charts / "b.png")
    prof, out = tmp_path / "p.json", tmp_path / "out"
    _profile(prof)
    assert main([str(charts), "--profile", str(prof), "--out", str(out)]) == 0
    assert (out / "a.csv").exists() and (out / "b.csv").exists()
