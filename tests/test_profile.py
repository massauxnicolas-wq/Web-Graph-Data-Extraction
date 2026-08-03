import numpy as np

from digitizer.core.pipeline import ExtractionParams
from digitizer.core.profile import Profile, ProfileCurve, apply_profile
from digitizer.io.json_export import write_profile
from digitizer.io.json_load import load_profile


def test_profile_save_load_round_trip(tmp_path):
    prof = Profile(
        calibration_pixel_pts=[(10.0, 90.0), (90.0, 90.0), (10.0, 10.0)],
        calibration_data_pts=[(1.0, 1.0), (80.0, 1.0), (1.0, 100.0)],
        x_log=False,
        y_log=True,
        params=ExtractionParams(dx=2, reducer="mean", smooth=True, smooth_window=7),
        curves=[ProfileCurve("red", (0, 255, 255), (5, 40, 40), seed=(1.0, 2.0), end=(9.0, 8.0))],
    )
    path = tmp_path / "p.json"
    write_profile(path, prof)
    loaded = load_profile(path)

    assert loaded.x_log is False and loaded.y_log is True
    assert loaded.params.dx == 2 and loaded.params.smooth is True and loaded.params.smooth_window == 7
    assert loaded.calibration_pixel_pts == [(10.0, 90.0), (90.0, 90.0), (10.0, 10.0)]
    c = loaded.curves[0]
    assert c.name == "red" and c.hsv_center == (0, 255, 255) and c.hsv_tol == (5, 40, 40)
    assert c.seed == (1.0, 2.0) and c.end == (9.0, 8.0)


def test_apply_profile_extracts_a_curve():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)  # white RGB
    img[45:55, 10:90] = (255, 0, 0)                     # red band

    prof = Profile(
        calibration_pixel_pts=[(10.0, 90.0), (90.0, 90.0), (10.0, 10.0)],
        calibration_data_pts=[(0.0, 0.0), (80.0, 0.0), (0.0, 100.0)],
        params=ExtractionParams(dx=5, reducer="mean"),
        curves=[ProfileCurve("red", (0, 255, 255), (10, 80, 80))],
    )
    series = apply_profile(img, prof)
    assert len(series) == 1
    assert series[0].xs.size >= 2
    assert np.isfinite(series[0].ys).all()
