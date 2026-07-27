from __future__ import annotations

import io

import numpy as np
from PyQt6.QtGui import QGuiApplication


def copy_curve_tsv(xs: np.ndarray, ys: np.ndarray, header: tuple[str, str] = ("x", "y")) -> str:
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    buf = io.StringIO()
    buf.write(f"{header[0]}\t{header[1]}\n")
    for x, y in zip(xs.tolist(), ys.tolist()):
        buf.write(f"{x}\t{y}\n")
    text = buf.getvalue()
    cb = QGuiApplication.clipboard()
    if cb is not None:
        cb.setText(text)
    return text
