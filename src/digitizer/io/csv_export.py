from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def write_curve_csv(path: str | Path, xs: np.ndarray, ys: np.ndarray, header: tuple[str, str] = ("x", "y")) -> None:
    path = Path(path)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must share shape")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for x, y in zip(xs.tolist(), ys.tolist()):
            writer.writerow([x, y])


def write_curves_wide(
    path: str | Path,
    curves: Iterable[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """Write multiple curves into a single wide CSV: name_x, name_y, ... columns.

    Curves with different lengths are padded with empty cells.
    """
    path = Path(path)
    curves = list(curves)
    if not curves:
        raise ValueError("no curves to write")
    max_len = max(len(xs) for _, xs, _ in curves)
    headers: list[str] = []
    for name, _, _ in curves:
        headers.extend([f"{name}_x", f"{name}_y"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for i in range(max_len):
            row: list[str] = []
            for _, xs, ys in curves:
                if i < len(xs):
                    row.extend([str(xs[i]), str(ys[i])])
                else:
                    row.extend(["", ""])
            writer.writerow(row)
