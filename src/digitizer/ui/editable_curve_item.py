from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt


class EditableCurveItem(pg.GraphItem):
    """Draggable + shift-click-deletable curve points.

    One GraphItem (wrapping a single internal ScatterPlotItem) per curve,
    not one item per point - scales fine to hundreds of points. Pattern
    adapted from pyqtgraph's own examples/CustomGraphItem.py. Works in
    whatever coordinate space its points are given (pixel or data).
    """

    sigPointsEdited = pyqtSignal()

    def __init__(self) -> None:
        self.dragPoint = None
        self.dragOffset = None
        super().__init__()
        self.scatter.sigClicked.connect(self._on_scatter_clicked)

    def set_points(self, xs: np.ndarray, ys: np.ndarray) -> None:
        pos = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        self._push(pos)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.get("pos") if self.data else None
        if pos is None or len(pos) == 0:
            return np.empty(0), np.empty(0)
        return pos[:, 0].copy(), pos[:, 1].copy()

    def _push(self, pos: np.ndarray) -> None:
        n = len(pos)
        adj = np.column_stack([np.arange(n - 1), np.arange(1, n)]) if n > 1 else np.empty((0, 2), dtype=int)
        self.setData(
            pos=pos, adj=adj, size=8, symbol="o", pxMode=True,
            brush=pg.mkBrush(0, 120, 255, 200), pen=pg.mkPen(0, 120, 255, width=2),
        )

    def setData(self, **kwds) -> None:
        self.data = kwds
        self.updateGraph()

    def updateGraph(self) -> None:
        pg.GraphItem.setData(self, **self.data)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            pts = self.scatter.pointsAt(ev.buttonDownPos())
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            self.dragOffset = self.data["pos"][pts[0].index()] - ev.buttonDownPos()
        elif ev.isFinish():
            self.dragPoint = None
            self.sigPointsEdited.emit()
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        ind = self.dragPoint.index()
        self.data["pos"][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()

    def _on_scatter_clicked(self, _scatter, points, ev) -> None:
        if not (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier) or len(points) == 0:
            return
        idx = points[0].index()
        self._push(np.delete(self.data["pos"], idx, axis=0))
        self.sigPointsEdited.emit()
