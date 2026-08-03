# AGENTS.md

Canonical working notes for any coding agent (Claude Code, Cursor, Copilot, …) on this
repository. Read this before touching code — several rules here were learned by breaking the
app, and the reasons are recorded so they don't get re-broken.

---

## 1. What this project is

A **plot digitizer**: load a raster image of a chart (PNG/JPG), calibrate its axes, then
automatically trace the plotted curves and export them as real numeric data (CSV / JSON /
clipboard TSV).

Current state: a working PyQt6 desktop application, ~3.7k lines of source, 92 tests.

**Where it is going.** This standalone app is a staging ground. The intent is to fold the
engine into a larger desktop application built on **Tauri + Vite + React + Tailwind, with a
FastAPI/Python backend**. That migration target is the single most important architectural
constraint in this repo — see §3.

---

## 2. Quick start

```bash
pip install -e ".[dev]"
```

```bash
python -m digitizer
```

```bash
python -m pytest -q
```

Requires Python ≥ 3.10. Tesseract OCR auto-calibration lives on the `feature/ocr`
branch, not on main; main has no external binary dependency.

Sample charts for manual testing live in `graphs/`.

---

## 3. Architecture — the one rule that matters

```
src/digitizer/
  core/     pure numpy / OpenCV / SciPy.  ZERO Qt imports.        <- survives the rewrite
  io/       CSV / JSON writers, Qt-free.                          <- survives the rewrite
            EXCEPT clipboard.py, which needs QGuiApplication.
  ui/       PyQt6 + pyqtgraph.                                    <- thrown away later
```

**`core/` must never import Qt.** It is the future FastAPI backend. Any new logic that is
conceptually computation — an algorithm, a fit, a metric, a transform — belongs in `core/`,
even when it would be quicker to inline it in a panel.

`io/csv_export.py` and `io/json_export.py` are Qt-free and port as-is.
`io/clipboard.py` imports `QGuiApplication` because the system clipboard is inherently a GUI
service — it is the one deliberate exception, and it will not port to the backend (the
browser handles clipboard in the React version). Keep new file-format writers Qt-free so
they stay portable.

**`ui/` is deliberately disposable.** It gets replaced by React. Invest in making it *usable*,
not beautiful or future-proof. Do not build abstractions in `ui/` for reuse that will never
happen.

Practical test when unsure where code goes: *would a FastAPI endpoint need this?* If yes →
`core/`. If it only exists to move pixels around a widget → `ui/`.

---

## 4. Module map

### `core/` — the engine

| Module | Public API | Notes |
|---|---|---|
| `image_io.py` | `load_image(path)` | Returns RGB ndarray. Unicode-safe on Windows (`np.fromfile` + `imdecode`). |
| `calibration.py` | `affine_from_points`, `round_trip_error` (linear primitives); `Calibration(M, x_log, y_log)`, `solve_calibration`, `calibration_error` | 3 pts → affine, 4 → perspective, >4 → homography. `solve_calibration` wraps that with per-axis **log10** handling (log axes calibrated in log space; positive values required). |
| `transform.py` | `pixel_to_data(pts, cal)`, `data_to_pixel(pts, cal)` | The only sanctioned way to move between coordinate spaces (§5). Take a `Calibration`; apply `M` then `10**` on log axes. |
| `masking.py` | `rgb_pixel_to_hsv`, `hsv_mask`, `mask_overlay_rgba` | HSV colour selection; handles hue wraparound at 180. |
| `xstep.py` | `extract_curve(mask, …)` | The tracer. Four reducers — see §6. |
| `interpolate.py` | `fill_gaps`, `fill_gaps_parametric`, `polynomial_best_fit`, `polynomial_best_fit_through_points`, `uniform_grid` | Gap filling and regression. `fill_gaps` is currently unused. |
| `quality.py` | `curve_stats`, `detect_outliers` | Point count / x-range / largest gap, and MAD-based spike flagging. |
| `pipeline.py` | `run_pipeline`, `apply_postprocessing`, `ExtractionParams`, `PipelineResult` | Qt-free extraction pipeline: extract → **smooth → fill** → best-fit (order matters). The one function the UI and a future FastAPI both call. Composes `xstep` + `interpolate`. |
| `export.py` | `NamedSeries`, `ExportOptions`, `Table`, `build_tables`, `serialize_delimited` | The single tabular-export model: series + options (resample / unit-convert / uncertainty) → Table(s) → CSV/TSV string. All CSV/TSV output routes through here. |
| `units.py` | `convert(values, from, to)` | Built-in unit conversions (stress / temperature / strain / length / accel). No dependency; raises on unknown or cross-family. |
| `auto_detect.py` | `detect_plot_box`, `detect_curve_colors` | Pure OpenCV (**no OCR**). Tesseract axis-label reading is on the `feature/ocr` branch. |

### `io/`
`json_export.build_payload` / `serialize_curve` / `write_payload` (schema `digitizer/0.2`) and
`json_load.load_payload` → `LoadedSession` (the inverse; accepts `0.1` and `0.2`) — all Qt-free.
Tabular data export (CSV/TSV) lives in `core/export.py`, not here. `clipboard.set_clipboard(text)`
is the pure Qt sink (the one sanctioned Qt user in `io/`).

### `ui/`

| Module | Role |
|---|---|
| `main_window.py` | Owns all state, wires every signal. The big one (~990 lines). |
| `image_view.py` | The main canvas: image, mask overlay, curve points, calibration markers, grid. |
| `editable_curve_item.py` | `pg.GraphItem` subclass — draggable/selectable/deletable curve points. |
| `calibration_panel.py` | Tab 1 controls. |
| `curve_panel.py` | Tab 2 controls **and** the `Curve` dataclass (see §5). |
| `edit_panel.py` | Tab 3 controls only — owns no plot. |
| `export_dialog.py` | Tab 4 — curve checklist + export buttons. |

---

## 5. Data model and coordinate spaces

The `Curve` dataclass lives in **`ui/curve_panel.py`** (it is UI-layer state, not engine state):

```python
id, name, hsv_center, hsv_tol, visible
pixel_xs, pixel_ys      # image pixel coordinates
data_xs,  data_ys       # calibrated real-world values
display_color           # on-canvas dot colour override (None = auto complement)
seed_point, end_point   # forced start/end, in PIXEL coords
manually_edited         # True once a human moved/added/deleted a point
```

**Every curve exists in two coordinate spaces simultaneously and they must stay in sync.**

- Extraction produces **pixel** coords → `data_*` derived via `transform.pixel_to_data`.
- Canvas editing produces **pixel** coords → same direction.
- Export always uses **`data_*`**.

Whenever you mutate one space, recompute the other, guarded on `_calibration_M is not None`
(`data_*` only exists once calibration is solved). `MainWindow._recompute_curve_data()` does
this in bulk.

`manually_edited` gates a confirmation dialog in `_run_xstep_for_curve` before re-extraction
discards hand edits. That guard lives in the single shared function both extract-one and
extract-all route through — do not duplicate it into the callers.

---

## 6. The extraction algorithm

`xstep.extract_curve(mask, dx, bbox, seed_y, seed_x, end_x, end_y, reducer, max_jump,
window_size, upscale_factor)` scans a boolean HSV mask and returns pixel `xs, ys`.

Four reducers, dispatched to private functions:

| Reducer | Strategy | `seed_x` honoured? |
|---|---|---|
| `mean` | mean Y of all mask pixels per column | no (scans every column) |
| `midpoint` | midpoint of the run nearest the last Y | no |
| `centroid` | arc-length stepping along local centroid direction | **yes** |
| `trace` | greedy nearest-neighbour KD-tree walk | **yes** |

- `seed_x`/`seed_y` force where tracing *starts* (defeats decoy blobs, legend swatches and
  markers near the origin). Documented no-op for `mean`/`midpoint` — the UI tooltip says so.
- `end_x` crops the result (`xs <= end_x`) and therefore works for **all** reducers.
- If both a seed and an end point are given, the exact points are **inserted verbatim** into
  the output so the curve provably starts and ends where the user pinned it, and
  `polynomial_best_fit_through_points` is used so the fit passes exactly through both.

### `quality.detect_outliers` — the non-obvious bit

Rolling-median residual + modified z-score. Plain MAD **collapses to zero** whenever outliers
are sparse (a lone spike has a residual median of 0), which would silently flag nothing — so
there is a deliberate fallback to a small fraction of the curve's own value range. Do not
"simplify" that away. Documented ceiling: it flags isolated spikes; a sustained step
discontinuity is *not* flagged, by design.

---

## 7. UI structure

Left: `ImageView` canvas (shared across all tabs). Right: a 4-tab panel.

| Tab | Panel | Purpose |
|---|---|---|
| 1. Calibrate | `calibration_panel.py` | Place/drag Origin, X-max, Y-max (+ optional 4th for skew), enter axis values, solve. |
| 2. Curves | `curve_panel.py` | Sample curve colours, tune HSV, choose reducer, extract. |
| 3. Editing | `edit_panel.py` | Toggle canvas point editing, image opacity, delete point / outliers, quality stats. |
| 4. Export | `export_dialog.py` | Check curves, write CSV/JSON/clipboard. |

Signal flow is strictly **panel → MainWindow → ImageView**. Panels never talk to each other
or to the canvas directly. `MainWindow` holds all state (`_curves_dict`, `_calibration_M`,
`_selected_curve_id`, `_edit_curve_id`, `_mode`).

`Mode` enum (`IDLE`, `CALIBRATING`, `PICKING_COLOR`, `SETTING_SEED`, `SETTING_END`) routes
canvas clicks in `_on_image_click`. Note that calibration-marker dragging and point editing
deliberately **bypass** `Mode` — they have their own signals, because they are not modal
click-capture states.

### Canvas behaviour by tab
- Calibration grid: bold blue dashed, **Calibrate tab only**.
- Other tabs: same geometry restyled to 30% black hairlines with darker X/Y axis lines, so
  the plot stays readable at full image opacity.
- Grid only appears after calibration is solved (it is drawn in calibrated data steps).

---

## 8. Invariants — break these and the app dies

> **`ImageView.set_curve_points()` must always draw. Never early-return from it.**
>
> It is the *only* path anything takes to the canvas — extraction results, visibility
> toggles, colour changes, curve creation. A previous revision added
> `if curve_id == self._editable_curve_id: return` and the canvas silently stopped
> updating for the edited curve; extraction appeared to do nothing and the app looked
> dead. It now *routes* (editable item vs plain scatter) but always renders.

Others:

- **Point editing is behind an explicit toggle.** With it off, no `EditableCurveItem` is in
  the scene at all, so the canvas runs exactly its original code path.
- **`core/` stays Qt-free.** Adding a Qt import there breaks the FastAPI migration.
- **Panels own no duplicate view of the same data.** Tab 3 has no plot precisely so there is
  one editing surface (the canvas) and nothing to keep in sync.
- **Fix root causes, not call sites.** Where several callers converge on one function, the
  guard belongs in that function (see `manually_edited`).

---

## 9. pyqtgraph notes (verified against the installed 0.14.0, not the docs)

`pyproject.toml` pins `pyqtgraph>=0.13` but 0.14.0 is installed and differs. Verify against
the installed source, not documentation or memory.

- `ScatterPlotItem.sigClicked` emits **`(self, points, ev)`** — three args. Older examples
  show two; a two-arg slot silently receives only the scatter.
- `TargetItem` signals are `sigPositionChanged` / **`sigPositionChangeFinished`**. There is
  no `sigDragged`. Use the *Finished* variant — the other fires every drag frame.
- `ScatterPlotItem` has **no** `mouseDragEvent`. For draggable points subclass `GraphItem`
  and override `mouseDragEvent`, hit-testing via `self.scatter.pointsAt(ev.buttonDownPos())`
  — this is pyqtgraph's own `examples/CustomGraphItem.py` recipe. One `GraphItem` per curve
  wraps a single internal scatter, so it scales to hundreds of points; one `TargetItem` per
  point does not (that is only right for the 1–4 calibration/seed/end markers).
- `pointsAt()` needs a real `QPointF`/`Point` **and** a rendered view — it raises `TypeError`
  on a raw tuple, and returns nothing if the widget was never shown/sized.
- A `GraphItem` with `mouseDragEvent` **does not block ViewBox panning** when it calls
  `ev.ignore()` — measured with synthetic events, panning is identical with it present,
  absent, and present-but-empty. Do not "fix" panning problems by removing it.
- `QColor` hue is 0–359; OpenCV `uint8` HSV hue is 0–179. Always round-trip through RGB with
  `cv2.cvtColor`, never via `QColor`'s hue getters.
- Wrapping a numpy crop in `QImage` does **not** copy. Call `np.ascontiguousarray()` then
  `.copy()` on the `QImage` before building a `QPixmap`, or the buffer can be freed under it.
- `QGroupBox.setCheckable(True)` + `toggled → inner.setVisible` gives collapsible sections
  natively. Don't write a custom collapsible widget.

---

## 10. Testing

```bash
python -m pytest -q
```

92 tests. `core/` and `io/` are covered conventionally. UI tests run **offscreen**:

```python
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # BEFORE the first PyQt6 import
```

Two tiers of UI test, and the distinction matters:

1. **Construction/wiring tests** — call methods directly. Catch import, attribute and crash
   errors. Cheap, but they *cannot* catch event-routing regressions.
2. **Real interaction tests** — drive Qt's actual event path with `QTest.mousePress/
   mouseMove/mouseRelease` on `image_view.viewport()`. These are the ones that catch "the
   canvas stopped responding". Existing examples assert panning still works with edit mode
   both off and on, and that dragging a point moves that point without panning.

Gotcha when writing interaction tests: **size the window generously (e.g. 1600×700)**. The
side panel has a ~608px minimum width, so in a small window the canvas collapses to a few
dozen pixels and your synthetic clicks land outside it, passing or failing for the wrong
reason. Assert `viewport().width() > 300` first.

**What automated tests cannot verify here:** how interaction *feels* — drag precision, whether
clicking reliably hits the intended point, whether a colour or opacity range is usable. There
is no way to drive a real interactive desktop window in the agent environment (browser
automation does not reach native apps). Always say so explicitly and list what the human
needs to click through.

CI (`.github/workflows/tests.yml`) runs the same suite on Ubuntu with
`QT_QPA_PLATFORM=offscreen` and the extra system libs Qt's offscreen plugin needs
(`libgl1`, `libegl1`, `libxkbcommon0`, `libxkbcommon-x11-0`, `libdbus-1-3`, `libxcb-cursor0`).

---

## 11. Conventions

- **Commit messages: no AI attribution.** No "Generated with…", no `Co-Authored-By: Claude`.
  Explain *why*, not just what. Small, focused commits — the user often asks for one commit
  per change.
- Match the surrounding style; this codebase uses plain functions, `np.asarray(..., dtype=float)`
  guards at boundaries, and `ValueError` for degenerate input.
- Comments explain *why*, especially for non-obvious numerics (see `detect_outliers`).
- Don't commit run output: `/*.csv`, `/session.json`, `scratch/` are gitignored. `session.json`
  embeds an absolute local path — it was purged from history once already.
- Licence: Apache 2.0.

---

## 12. Known limitations / deferred

- `main_window.py` (~780 lines) is a god object, left alone deliberately — it is the layer
  being replaced by React. Extraction/processing logic has been evacuated to `core/pipeline.py`
  (§4); what remains is Qt wiring. `extract_curve_requested` / `extract_all_requested` now carry
  a single `ExtractionParams` object (previously 11 positional params).
- `interpolate.fill_gaps` is unused (superseded by `fill_gaps_parametric`).
- Tesseract OCR auto-calibration is parked on the `feature/ocr` branch (best-effort, needs
  Tesseract on PATH). Main keeps only deterministic OpenCV plot-box / curve-color detection.
- Session JSON now has a Qt-free loader (`json_load.load_payload`), but **restoring it into the
  running app** (rebuilding curves/markers/widgets) is not wired — deferred to the React frontend.
