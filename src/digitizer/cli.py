"""Headless batch digitizing: apply a saved profile to an image (or a folder) and write CSVs.

Qt-free — composes the same core the GUI uses. The entry FastAPI will later wrap as an endpoint.

    digitizer-batch chart.png --profile recipe.json --out data.csv
    digitizer-batch charts/   --profile recipe.json --out out/ --grid 0.1 --y-unit MPa:GPa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from digitizer.core.export import ExportOptions, NamedSeries, build_tables, serialize_delimited
from digitizer.core.image_io import load_image
from digitizer.core.profile import apply_profile
from digitizer.io.json_load import load_profile

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _unit(spec: str) -> tuple[str, str] | None:
    if not spec:
        return None
    if ":" not in spec:
        raise SystemExit(f"unit must look like FROM:TO, got {spec!r}")
    a, b = spec.split(":", 1)
    return (a.strip(), b.strip())


def _options(args: argparse.Namespace) -> ExportOptions:
    return ExportOptions(
        layout=args.layout,
        x_grid_step=args.grid if args.grid > 0 else None,
        x_unit=_unit(args.x_unit),
        y_unit=_unit(args.y_unit),
        include_uncertainty=args.uncertainty,
    )


def _write(series: list[NamedSeries], opts: ExportOptions, out: Path) -> int:
    series = [s for s in series if s.xs.size > 0]
    if not series:
        return 0
    if opts.layout == "individual":
        out.mkdir(parents=True, exist_ok=True)
        for s in series:
            (table,) = build_tables([s], opts)
            (out / f"{s.name}.csv").write_text(serialize_delimited(table), encoding="utf-8")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        (table,) = build_tables(series, opts)
        out.write_text(serialize_delimited(table), encoding="utf-8")
    return len(series)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="digitizer-batch",
                                description="Digitize an image (or folder) with a saved profile.")
    p.add_argument("input", help="image file, or a folder of images")
    p.add_argument("--profile", required=True, help="extraction profile JSON (see 'Save profile')")
    p.add_argument("--out", required=True, help="CSV file (wide) or folder (individual / batch)")
    p.add_argument("--layout", choices=["wide", "individual"], default="wide")
    p.add_argument("--grid", type=float, default=0.0, help="resample to a uniform X step (0 = raw)")
    p.add_argument("--x-unit", default="", help="convert X units, e.g. %%:ratio")
    p.add_argument("--y-unit", default="", help="convert Y units, e.g. MPa:GPa")
    p.add_argument("--uncertainty", action="store_true", help="add per-point ±dy columns")
    args = p.parse_args(argv)

    profile = load_profile(args.profile)
    opts = _options(args)
    inp, out = Path(args.input), Path(args.out)

    if inp.is_dir():
        images = sorted(q for q in inp.iterdir() if q.suffix.lower() in _EXTS)
        if not images:
            print(f"no images in {inp}", file=sys.stderr)
            return 1
        out.mkdir(parents=True, exist_ok=True)
        for img in images:
            dest = (out / img.stem) if opts.layout == "individual" else (out / f"{img.stem}.csv")
            n = _write(apply_profile(load_image(img), profile), opts, dest)
            print(f"{img.name}: {n} curve(s)")
    else:
        n = _write(apply_profile(load_image(inp), profile), opts, out)
        print(f"{inp.name}: {n} curve(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
