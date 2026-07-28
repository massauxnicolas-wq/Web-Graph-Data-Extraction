import cv2
import numpy as np

from digitizer.core.auto_detect import PlotBox, detect_curve_colors, detect_plot_box


def _chart_with_axes(size=300, margin=40):
    """White image with a black plot-box rectangle, like a simple chart's axes."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (margin, margin), (size - margin, size - margin), (0, 0, 0), 2)
    return img


def test_detect_plot_box_finds_the_axes_rectangle():
    img = _chart_with_axes(size=300, margin=40)
    box = detect_plot_box(img)
    assert isinstance(box, PlotBox)
    # detected box should roughly match the drawn rectangle, not the full image
    assert 20 < box.x < 60
    assert 20 < box.y < 60
    assert box.w > 150
    assert box.h > 150


def test_detect_plot_box_falls_back_to_full_image_when_no_axes():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    box = detect_plot_box(img)
    assert (box.x, box.y, box.w, box.h) == (0, 0, 200, 100)


def test_detect_curve_colors_finds_dominant_saturated_color():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)  # white background (BGR)
    img[40:60, :] = (0, 0, 255)  # solid red band (BGR)
    box = PlotBox(x=0, y=0, w=100, h=100, origin_x=0, origin_y=100)
    colors = detect_curve_colors(img, box)
    assert len(colors) >= 1
    # returned as (R, G, B): red band should dominate
    assert any(r > 200 and g < 60 and b < 60 for r, g, b in colors)


def test_detect_curve_colors_empty_box_returns_empty_list():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    box = PlotBox(x=0, y=0, w=0, h=0, origin_x=0, origin_y=0)
    assert detect_curve_colors(img, box) == []
