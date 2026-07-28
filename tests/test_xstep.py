import numpy as np
import pytest

from digitizer.core.xstep import extract_curve


def _line_mask(h=50, w=100, slope=0.2, intercept=10):
    mask = np.zeros((h, w), dtype=bool)
    for x in range(w):
        y = int(intercept + slope * x)
        if 0 <= y < h:
            mask[y, x] = True
    return mask


def test_rejects_non_2d_mask():
    with pytest.raises(ValueError, match="2D"):
        extract_curve(np.zeros((3, 3, 3), dtype=bool))


def test_rejects_dx_below_one():
    with pytest.raises(ValueError, match="dx"):
        extract_curve(np.zeros((3, 3), dtype=bool), dx=0)


def test_rejects_unknown_reducer():
    with pytest.raises(ValueError, match="reducer"):
        extract_curve(np.zeros((3, 3), dtype=bool), reducer="bogus")


def test_empty_mask_returns_empty_arrays():
    xs, ys = extract_curve(np.zeros((10, 10), dtype=bool))
    assert xs.size == 0
    assert ys.size == 0


@pytest.mark.parametrize("reducer", ["mean", "midpoint", "centroid", "trace"])
def test_reducer_follows_a_straight_line(reducer):
    mask = _line_mask(slope=0.2, intercept=10)
    xs, ys = extract_curve(mask, dx=2, reducer=reducer)
    assert len(xs) > 10
    # every recovered point should sit close to the true line y = 10 + 0.2x
    expected_y = 10 + 0.2 * xs
    assert np.max(np.abs(ys - expected_y)) < 2.0


def test_bbox_restricts_scan_region():
    mask = _line_mask()
    xs, ys = extract_curve(mask, dx=2, bbox=(0, 0, 49, 49))
    assert xs.max() <= 49


def test_midpoint_picks_run_closest_to_seed():
    # Two separate horizontal runs in the same column; midpoint should track the seed.
    mask = np.zeros((30, 5), dtype=bool)
    mask[5, :] = True
    mask[25, :] = True
    xs, ys = extract_curve(mask, dx=1, reducer="midpoint", seed_y=6)
    assert ys[0] == pytest.approx(5.0)


@pytest.mark.parametrize("reducer", ["trace", "centroid"])
def test_seed_x_picks_correct_branch_over_a_decoy_blob(reducer):
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 2:18] = True   # the real curve, a flat line at y=10
    mask[3, 2:6] = True     # a decoy blob near the left edge (e.g. a marker/legend swatch)
    xs, ys = extract_curve(mask, reducer=reducer, seed_x=10, seed_y=10, max_jump=5)
    assert xs.size > 0
    assert abs(ys[0] - 10) < 1


def test_upscale_factor_returns_same_pixel_scale():
    mask = _line_mask()
    xs1, ys1 = extract_curve(mask, dx=2, reducer="mean", upscale_factor=1)
    xs2, ys2 = extract_curve(mask, dx=2, reducer="mean", upscale_factor=2)
    # Same underlying line, so recovered y-values should agree within a pixel.
    common_x = np.intersect1d(xs1, xs2)
    assert len(common_x) > 5
    for x in common_x[:5]:
        y1 = ys1[xs1 == x][0]
        y2 = ys2[xs2 == x][0]
        assert abs(y1 - y2) < 1.5
