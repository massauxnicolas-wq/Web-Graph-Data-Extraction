import numpy as np

from digitizer.core.export import (
    ExportOptions,
    NamedSeries,
    build_tables,
    serialize_delimited,
)


def _two_series():
    return [
        NamedSeries("a", np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 20.0])),
        NamedSeries("b", np.array([0.0, 1.0]), np.array([5.0, 15.0])),
    ]


def test_wide_layout_pads_shorter_series():
    (t,) = build_tables(_two_series(), ExportOptions(layout="wide"))
    assert t.headers == ["a_x", "a_y", "b_x", "b_y"]
    assert t.rows[0] == [0.0, 0.0, 0.0, 5.0]
    assert t.rows[2] == [2.0, 20.0, "", ""]  # b is padded


def test_individual_layout_one_table_per_series():
    tables = build_tables(_two_series(), ExportOptions(layout="individual"))
    assert len(tables) == 2
    assert tables[0].headers == ["a_x", "a_y"]
    assert tables[1].headers == ["b_x", "b_y"]


def test_serialize_csv_and_tsv():
    (t,) = build_tables([NamedSeries("a", np.array([0.0, 1.0]), np.array([2.0, 3.0]))])
    assert serialize_delimited(t, ",").splitlines()[0] == "a_x,a_y"
    assert serialize_delimited(t, "\t").splitlines()[0] == "a_x\ta_y"


def test_resample_to_uniform_grid():
    s = [NamedSeries("a", np.array([0.0, 2.0, 4.0]), np.array([0.0, 20.0, 40.0]))]
    (t,) = build_tables(s, ExportOptions(x_grid_step=1.0))
    xs = [r[0] for r in t.rows]
    ys = [r[1] for r in t.rows]
    np.testing.assert_allclose(xs, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(ys, [0.0, 10.0, 20.0, 30.0, 40.0])


def test_unit_conversion_on_y():
    s = [NamedSeries("a", np.array([1.0, 2.0]), np.array([1.0, 2.0]))]  # y in GPa
    (t,) = build_tables(s, ExportOptions(y_unit=("GPa", "MPa")))
    np.testing.assert_allclose([r[1] for r in t.rows], [1000.0, 2000.0])


def test_uncertainty_column_included_only_when_requested():
    s = [NamedSeries("a", np.array([0.0, 1.0]), np.array([0.0, 10.0]), dy=np.array([0.5, 0.5]))]
    assert "a_dy" not in build_tables(s)[0].headers
    (t,) = build_tables(s, ExportOptions(include_uncertainty=True))
    assert t.headers == ["a_x", "a_y", "a_dy"]
