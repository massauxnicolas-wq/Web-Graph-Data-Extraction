import numpy as np

from digitizer.core.quality import curve_stats, detect_outliers


def test_curve_stats_empty():
    stats = curve_stats(np.array([]), np.array([]))
    assert stats == {"count": 0, "x_range": (0.0, 0.0), "largest_gap": 0.0}


def test_curve_stats_basic():
    xs = np.array([0, 1, 2, 5, 6])
    ys = np.zeros(5)
    stats = curve_stats(xs, ys)
    assert stats["count"] == 5
    assert stats["x_range"] == (0.0, 6.0)
    assert stats["largest_gap"] == 3.0


def test_curve_stats_unsorted_input():
    xs = np.array([5, 0, 2])
    ys = np.zeros(3)
    stats = curve_stats(xs, ys)
    assert stats["x_range"] == (0.0, 5.0)


def test_detect_outliers_too_few_points():
    xs = np.arange(4)
    ys = np.array([0, 0, 100, 0])
    assert not detect_outliers(xs, ys).any()


def test_detect_outliers_flat_curve_no_false_positives():
    xs = np.arange(20)
    ys = np.full(20, 5.0)
    assert not detect_outliers(xs, ys).any()


def test_detect_outliers_flags_single_spike():
    xs = np.arange(30)
    ys = np.zeros(30)
    ys[15] = 50.0  # one clear isolated spike
    flagged = detect_outliers(xs, ys)
    assert flagged[15]
    assert flagged.sum() < 5  # neighbors should not also be flagged


def test_detect_outliers_does_not_overflag_a_sustained_step():
    xs = np.arange(40)
    ys = np.concatenate([np.zeros(20), np.full(20, 50.0)])  # a real step, not a spike
    flagged = detect_outliers(xs, ys)
    # a handful of points right at the transition may trip the residual test,
    # but the sustained step itself must not be flagged wholesale
    assert flagged.sum() < 10
