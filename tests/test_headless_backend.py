"""Guard: the backend layer must import without PyQt.

Everything under ``core/`` plus the ``io`` serializers is what FastAPI will import
server-side once PyQt is replaced by a React frontend (backend stays Python). If any
of them pulls Qt into ``sys.modules``, that headless split silently breaks.

``io/clipboard.py`` is the one sanctioned Qt user — clipboard is inherently GUI and has
no server-side equivalent — so it is excluded here.

Runs in a clean subprocess: the Qt UI tests import PyQt6 into the shared pytest process,
which would mask a real leak if we just checked ``sys.modules`` here.
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_backend_layer_imports_without_pyqt():
    core_dir = SRC / "digitizer" / "core"
    modules = [
        f"digitizer.core.{p.stem}"
        for p in sorted(core_dir.glob("*.py"))
        if p.stem != "__init__"
    ]
    modules += ["digitizer.io.json_export", "digitizer.io.json_load", "digitizer.cli"]

    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
        "leaked = sorted(n for n in sys.modules "
        "if n.split('.')[0] in ('PyQt6', 'PySide6', 'PySide2'))\n"
        "assert not leaked, 'Qt leaked into backend layer via: ' + ', '.join(leaked)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(SRC),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
