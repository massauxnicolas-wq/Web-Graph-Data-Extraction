# Plot Digitizer

Automated Python plot digitizer — interactive PyQtGraph GUI on top of the
deterministic **X-Step** extraction algorithm described in [project.md](project.md).

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10 and a desktop environment for Qt.

## Run

```bash
python -m digitizer
# or, after install:
digitizer
```

## Workflow

1. **Open Image** — toolbar button, pick a graph PNG/JPG.
2. **Calibrate** — open the *Calibrate* panel, click "Click 3 calibration
   points", then click on the chart's origin, X-axis maximum, and Y-axis
   maximum (in that order). Type the corresponding data values into the
   `Data X` / `Data Y` fields, then "Solve calibration". The status line
   shows the affine round-trip error.
3. **Curves** — open the *Curves* panel, click "Pick curve color", click on
   one pixel of the curve. Tune the H/S/V tolerance sliders until the red
   mask overlay covers only the curve. Press "Run X-Step". Inspect the
   green scatter overlay. Press "Add as new curve" to commit.
4. **Repeat** step 3 for each additional curve (multi-curve supported).
5. **Export** — open the *Export* panel: save active curve as CSV, all
   curves as a wide CSV, the full session as JSON (includes calibration
   matrix + per-curve HSV settings), or copy active curve as TSV to the
   clipboard for direct paste into Excel.

## Sample inputs

`graphs/` ships with four reference plots used during validation:

- `stress_strain.png` — single clean curve (good first test)
- `ixef1521_polyarylamide.png` — multi-curve, good for testing multi-curve flow
- `ixef1521_polyarylamide_secantmodulus.png`
- `permabond_hotstrength.png`

## Architecture

- `src/digitizer/core/` — pure-numeric pipeline (image I/O, affine
  calibration, HSV masking, X-Step scanner, cubic-spline gap fill,
  pixel↔data transform). No Qt dependency.
- `src/digitizer/ui/` — PyQt6 + pyqtgraph: `ImageView`, calibration /
  curve / export panels, and `MainWindow` (mode state machine + signal
  router).
- `src/digitizer/io/` — CSV, JSON, clipboard exporters.

## Out of scope (MVP)

Log-axis calibration, automatic deskew beyond what affine handles,
curve fitting / regression, headless batch CLI, and session reload. The
JSON schema (`digitizer/0.1`) is forward-compatible with adding session
reload later.

## License

No license file yet — all rights reserved by default until one is added.
