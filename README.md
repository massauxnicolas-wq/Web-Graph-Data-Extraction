# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# Plot Digitizer

Automated Python plot digitizer — interactive PyQtGraph GUI on top of the
deterministic **X-Step** extraction algorithm described in `project.md`.

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
