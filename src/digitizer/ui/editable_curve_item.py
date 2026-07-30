from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt

BASE_BRUSH = pg.mkBrush(0, 120, 255, 200)
SELECTED_BRUSH = pg.mkBrush(255, 60, 0, 255)


class EditableCurveItem(pg.GraphItem):
    """Draggable curve points with a selectable point for deletion.

    One GraphItem (wrapping a single internal ScatterPlotItem) per curve,
    not one item per point - scales fine to hundreds of points. Pattern
    adapted from pyqtgraph's own examples/CustomGraphItem.py.

    Deletion is deliberately NOT dependent on a modifier+click gesture:
    touching a point (either by starting a drag on it, or plain-clicking
    it) selects it, and the owning panel deletes the selection via a
    button or the Delete key. Shift+click still works as a shortcut.
    """

    sigPointsEdited = pyqtSignal()
    sigPointSelected = pyqtSignal(int)  # index, or -1 for "nothing selected"

    def __init__(self) -> None:
        self.dragPoint = None
        self.dragOffset = None
        self.selected_index = -1
        super().__init__()
        self.scatter.sigClicked.connect(self._on_scatter_clicked)

    # --- data in/out --------------------------------------------------------
    def set_points(self, xs: np.ndarray, ys: np.ndarray, keep_selection: bool = False) -> None:
        pos = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        if not keep_selection:
            self.selected_index = -1
        if self.selected_index >= len(pos):
            self.selected_index = -1
        self._push(pos)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.get("pos") if self.data else None
        if pos is None or len(pos) == 0:
            return np.empty(0), np.empty(0)
        return pos[:, 0].copy(), pos[:, 1].copy()

    def _push(self, pos: np.ndarray) -> None:
        n = len(pos)
        adj = np.column_stack([np.arange(n - 1), np.arange(1, n)]) if n > 1 else np.empty((0, 2), dtype=int)
        brushes = [BASE_BRUSH] * n
        if 0 <= self.selected_index < n:
            brushes[self.selected_index] = SELECTED_BRUSH
        self.setData(
            pos=pos, adj=adj, size=11, symbol="o", pxMode=True,
            symbolBrush=brushes, pen=pg.mkPen(0, 120, 255, width=2),
        )

    def setData(self, **kwds) -> None:
        self.data = kwds
        self.updateGraph()

    def updateGraph(self) -> None:
        pg.GraphItem.setData(self, **self.data)

    # --- selection / deletion ----------------------------------------------
    def select_point(self, index: int) -> None:
        xs, _ = self.points()
        self.selected_index = index if 0 <= index < len(xs) else -1
        if self.data.get("pos") is not None:
            self._push(self.data["pos"])
        self.sigPointSelected.emit(self.selected_index)

    def delete_selected(self) -> bool:
        """Delete the currently selected point. Returns True if one was removed."""
        return self.delete_index(self.selected_index)

    def delete_index(self, index: int) -> bool:
        pos = self.data.get("pos")
        if pos is None or not (0 <= index < len(pos)):
            return False
        self.selected_index = -1
        self._push(np.delete(pos, index, axis=0))
        self.sigPointSelected.emit(-1)
        self.sigPointsEdited.emit()
        return True

    def delete_mask(self, mask: np.ndarray) -> int:
        """Delete every point where mask is True. Returns how many were removed."""
        pos = self.data.get("pos")
        mask = np.asarray(mask, dtype=bool)
        if pos is None or mask.size != len(pos) or not mask.any():
            return 0
        self.selected_index = -1
        self._push(pos[~mask])
        self.sigPointSelected.emit(-1)
        self.sigPointsEdited.emit()
        return int(mask.sum())

    def insert_point(self, x: float, y: float) -> None:
        xs, ys = self.points()
        idx = int(np.searchsorted(xs, x))
        self.selected_index = -1
        self._push(np.column_stack([np.insert(xs, idx, x), np.insert(ys, idx, y)]))
        self.sigPointsEdited.emit()

    # --- mouse --------------------------------------------------------------
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
            index = pts[0].index()
            self.dragOffset = self.data["pos"][index] - ev.buttonDownPos()
            # Starting a drag also selects the point - this path works even if
            # the scatter's click signal never reaches us.
            self.selected_index = index
            self.sigPointSelected.emit(index)
        elif ev.isFinish():
            self.dragPoint = None
            self._push(self.data["pos"])  # repaint selection highlight
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
        if len(points) == 0:
            return
        index = points[0].index()
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.delete_index(index)
            return
        self.select_point(index)
