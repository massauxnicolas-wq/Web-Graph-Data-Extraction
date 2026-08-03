"""The single tabular export model. Qt-free.

Everything that leaves the app as a table — CSV file, TSV clipboard, batch CLI output — goes
through here: a list of NamedSeries + ExportOptions -> Table(s) -> a delimited string. Sinks
(write a file, push to the clipboard) stay thin and live with their caller.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

from digitizer.core import interpolate, units


@dataclass
class NamedSeries:
    name: str
    xs: np.ndarray
    ys: np.ndarray
    dy: np.ndarray | None = None  # optional per-point uncertainty (Phase 2 / #3)


@dataclass
class ExportOptions:
    layout: str = "wide"                    # "wide" (name_x/name_y columns) | "individual"
    x_grid_step: float | None = None        # resample X to a uniform step (#5); None = raw points
    x_unit: tuple[str, str] | None = None    # (from, to) unit conversion for X (#5)
    y_unit: tuple[str, str] | None = None    # (from, to) unit conversion for Y (#5)
    include_uncertainty: bool = False        # emit name_dy columns when dy is present (#3)


@dataclass
class Table:
    headers: list[str]
    rows: list[list]


def build_tables(series: list[NamedSeries], opts: ExportOptions | None = None) -> list[Table]:
    """Apply resampling + unit conversion, then assemble into one or many tables."""
    opts = opts or ExportOptions()
    prepared = [_prepare(s, opts) for s in series]
    if opts.layout == "individual":
        return [_table_from_cols(_cols_for(s)) for s in prepared]
    cols: list[tuple[str, np.ndarray]] = []
    for s in prepared:
        cols.extend(_cols_for(s))
    return [_table_from_cols(cols)]


def serialize_delimited(table: Table, delimiter: str = ",") -> str:
    """Render a Table as CSV (delimiter=',') or TSV (delimiter='\\t')."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    writer.writerow(table.headers)
    for row in table.rows:
        writer.writerow(row)
    return buf.getvalue()


# --- internals -------------------------------------------------------------

def _prepare(s: NamedSeries, opts: ExportOptions) -> NamedSeries:
    xs = np.asarray(s.xs, dtype=float)
    ys = np.asarray(s.ys, dtype=float)
    dy = None if s.dy is None else np.asarray(s.dy, dtype=float)

    if opts.x_grid_step and xs.size >= 2:
        grid = interpolate.uniform_grid(xs, opts.x_grid_step)
        ys = interpolate.fill_gaps(xs, ys, grid)
        if dy is not None:
            dy = interpolate.fill_gaps(xs, dy, grid)
        xs = grid

    if opts.x_unit:
        xs = units.convert(xs, *opts.x_unit)
    if opts.y_unit:
        if dy is not None:
            # Convert the interval exactly (correct for affine units too).
            dy = np.abs(units.convert(ys + dy, *opts.y_unit) - units.convert(ys, *opts.y_unit))
        ys = units.convert(ys, *opts.y_unit)

    return NamedSeries(s.name, xs, ys, dy if opts.include_uncertainty else None)


def _cols_for(s: NamedSeries) -> list[tuple[str, np.ndarray]]:
    cols = [(f"{s.name}_x", s.xs), (f"{s.name}_y", s.ys)]
    if s.dy is not None:
        cols.append((f"{s.name}_dy", s.dy))
    return cols


def _table_from_cols(cols: list[tuple[str, np.ndarray]]) -> Table:
    headers = [h for h, _ in cols]
    n = max((len(v) for _, v in cols), default=0)
    rows = [[(v[i] if i < len(v) else "") for _, v in cols] for i in range(n)]
    return Table(headers, rows)
