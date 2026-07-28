import numpy as np

from digitizer.core.masking import hsv_mask, mask_overlay_rgba, rgb_pixel_to_hsv


def _solid_rgb(rgb, h=20, w=20):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = rgb
    return img


def test_hsv_mask_matches_solid_color_image():
    img = _solid_rgb((255, 0, 0))  # pure red
    h, s, v = rgb_pixel_to_hsv(img, 0, 0)
    mask = hsv_mask(img, (h, s, v), tol=(5, 30, 30))
    assert mask.all()


def test_hsv_mask_excludes_far_color():
    red = _solid_rgb((255, 0, 0))
    h, s, v = rgb_pixel_to_hsv(red, 0, 0)
    blue = _solid_rgb((0, 0, 255))
    mask = hsv_mask(blue, (h, s, v), tol=(5, 30, 30))
    assert not mask.any()


def test_hsv_mask_hue_wraparound_near_zero():
    # Hue near 0/180 boundary (red) must match across the wrap.
    img = np.zeros((1, 2, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)
    img[0, 1] = (255, 10, 10)
    mask = hsv_mask(img, (0, 255, 255), tol=(3, 200, 200))
    assert mask[0, 0]


def test_hsv_mask_wide_hue_tolerance_ignores_hue():
    img = _solid_rgb((0, 255, 0))
    mask = hsv_mask(img, (0, 255, 255), tol=(90, 255, 255))
    assert mask.all()


def test_mask_overlay_rgba_shape_and_alpha():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    out = mask_overlay_rgba(mask, color=(1, 2, 3), alpha=140)
    assert out.shape == (5, 5, 4)
    assert tuple(out[2, 2]) == (1, 2, 3, 140)
    assert tuple(out[0, 0]) == (0, 0, 0, 0)
