"""The system-clipboard sink. The one sanctioned Qt user in io/ (clipboard is inherently GUI)."""
from __future__ import annotations

from PyQt6.QtGui import QGuiApplication


def set_clipboard(text: str) -> None:
    cb = QGuiApplication.clipboard()
    if cb is not None:
        cb.setText(text)
