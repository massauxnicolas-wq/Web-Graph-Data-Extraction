from __future__ import annotations

import numpy as np
import cv2
from scipy.spatial import KDTree

REDUCERS = ("mean", "midpoint", "centroid", "trace")


def _runs_in_column(col: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_inclusive) index ranges where col is True."""
    if not col.any():
        return []
    diff = np.diff(col.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _extract_trace(
    mask: np.ndarray,
    x_min: int, y_min: int, x_max: int, y_max: int,
    seed_y: float | None, seed_x: float | None, dx: int, max_jump: float | None,
) -> tuple[list[float], list[float]]:
    """Path-trace via greedy nearest-neighbor walk. Best for steep/vertical/looping curves."""
    y_idx, x_idx = np.where(mask[y_min:y_max + 1, x_min:x_max + 1])
    if len(x_idx) == 0:
        return [], []

    x_idx = x_idx + x_min
    y_idx = y_idx + y_min
    points = np.column_stack((x_idx, y_idx))

    if seed_x is not None:
        # Forced start: nearest mask point to the user-picked (seed_x, seed_y),
        # instead of always starting at the leftmost column (which a decoy
        # blob/marker/legend swatch near the edge could otherwise hijack).
        sy = seed_y if seed_y is not None else (y_min + y_max) / 2.0
        dists = np.hypot(points[:, 0] - seed_x, points[:, 1] - sy)
        curr = int(np.argmin(dists))
    elif seed_y is not None:
        left_points = points[points[:, 0] == np.min(points[:, 0])]
        start_idx = np.argmin(np.abs(left_points[:, 1] - seed_y))
        start_point = left_points[start_idx]
        curr = np.where((points == start_point).all(axis=1))[0][0]
    else:
        curr = np.argmin(x_idx)

    tree = KDTree(points)
    unvisited = set(range(len(points)))

    path_x, path_y = [], []
    step_radius = max(1.0, float(dx))

    while unvisited:
        path_x.append(float(points[curr][0]))
        path_y.append(float(points[curr][1]))

        # Clear nearby points to avoid getting stuck zigzagging in local thickness.
        nearby = tree.query_ball_point(points[curr], r=step_radius)
        unvisited.difference_update(nearby)

        if not unvisited:
            break

        k_search = min(int(4 * step_radius ** 2) + 50, len(points))
        distances, indices = tree.query(points[curr], k=k_search)

        next_curr = None
        for dist, idx in zip(distances, indices):
            if idx in unvisited:
                if points[idx][0] < points[curr][0] - step_radius:
                    continue  # no doubling back horizontally past the step radius
                if max_jump is not None and dist > max_jump:
                    continue
                next_curr = idx
                break

        if next_curr is None:
            # Gap larger than the initial search radius but maybe still within max_jump.
            k_search_wide = min(k_search * 5, len(points))
            distances, indices = tree.query(points[curr], k=k_search_wide)
            for dist, idx in zip(distances, indices):
                if idx in unvisited:
                    if points[idx][0] < points[curr][0] - step_radius:
                        continue
                    if max_jump is not None and dist > max_jump:
                        continue
                    next_curr = idx
                    break

        if next_curr is None:
            break  # path broke or gap too large
        curr = next_curr

    return path_x, path_y


def _extract_centroid(
    mask: np.ndarray,
    x_min: int, y_min: int, x_max: int, y_max: int,
    seed_y: float | None, seed_x: float | None, dx: int, window_size: int,
) -> tuple[list[float], list[float]]:
    """Fixed arc-length stepping along the local centroid direction."""
    xs: list[float] = []
    ys: list[float] = []

    if seed_x is not None:
        cur_x = float(min(max(seed_x, x_min), x_max))
        if seed_y is not None:
            cur_y = float(seed_y)
        else:
            col = mask[y_min:y_max + 1, int(round(cur_x))]
            cur_y = float(np.median(np.where(col)[0])) + y_min if col.any() else (y_min + y_max) / 2.0
    elif seed_y is not None:
        cur_y = float(seed_y)
        cur_x = float(x_min)
    else:
        cur_y = None
        cur_x = float(x_min)
        for x in range(x_min, x_max + 1):
            col = mask[y_min:y_max + 1, x]
            if col.any():
                cur_x = float(x)
                cur_y = float(np.median(np.where(col)[0])) + y_min
                break
        if cur_y is None:
            return [], []

    dir_x, dir_y = 1.0, 0.0

    while x_min <= cur_x <= x_max and y_min <= cur_y <= y_max:
        x0, x1 = int(max(x_min, cur_x - window_size // 2)), int(min(x_max, cur_x + window_size // 2))
        y0, y1 = int(max(y_min, cur_y - window_size // 2)), int(min(y_max, cur_y + window_size // 2))

        window = mask[y0:y1 + 1, x0:x1 + 1]
        if not window.any():
            break

        y_indices, x_indices = np.where(window)
        if y_indices.size == 0:
            break

        centroid_y = float(np.median(y_indices)) + y0
        centroid_x = float(np.median(x_indices)) + x0

        xs.append(centroid_x)
        ys.append(centroid_y)

        if x_indices.size > 1:
            cov = np.cov(x_indices, y_indices)
            if np.count_nonzero(cov) > 0:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                vx, vy = eigenvectors[:, np.argmax(eigenvalues)]

                if (vx * dir_x + vy * dir_y) < 0:
                    vx, vy = -vx, -vy

                dir_x = 0.5 * dir_x + 0.5 * vx
                dir_y = 0.5 * dir_y + 0.5 * vy
                norm = np.hypot(dir_x, dir_y)
                if norm > 0:
                    dir_x /= norm
                    dir_y /= norm

                # Step along the full 2D direction vector, not just fixed-X increments,
                # so steep/near-vertical segments still advance.
                step_x = dir_x * dx
                step_y = dir_y * dx
            else:
                step_x, step_y = float(dx), 0.0
        else:
            step_x, step_y = float(dx), 0.0

        cur_x += step_x
        cur_y += step_y

        if abs(step_x) < 0.1 and abs(step_y) < 0.1:
            break  # stuck in place

    return xs, ys


def _extract_column_scan(
    mask: np.ndarray,
    x_min: int, y_min: int, x_max: int, y_max: int,
    seed_y: float | None, dx: int, reducer: str,
) -> tuple[list[float], list[float]]:
    """Mean or midpoint reducer: one Y value per scanned X column."""
    xs: list[float] = []
    ys: list[float] = []
    last_y: float | None = float(seed_y) if seed_y is not None else None

    for x in range(x_min, x_max + 1, dx):
        col = mask[y_min:y_max + 1, x]
        if not col.any():
            continue

        if reducer == "mean":
            y_indices = np.where(col)[0]
            y = float(y_indices.mean()) + y_min
        else:  # midpoint
            runs = _runs_in_column(col)
            midpoints = [(s + e) / 2.0 + y_min for s, e in runs]
            target = last_y if last_y is not None else (y_min + y_max) / 2.0
            y = min(midpoints, key=lambda m, t=target: abs(m - t))

        xs.append(float(x))
        ys.append(y)
        last_y = y

    return xs, ys


def extract_curve(
    mask: np.ndarray,
    dx: int = 2,
    bbox: tuple[int, int, int, int] | None = None,
    seed_y: int | None = None,
    seed_x: int | None = None,
    end_x: float | None = None,
    end_y: float | None = None,
    reducer: str = "mean",
    max_jump: float | None = 50.0,
    window_size: int = 20,
    upscale_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Column-scan (or path-trace) a boolean mask and return pixel xs, ys for one curve."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if dx < 1:
        raise ValueError("dx must be >= 1")
    if reducer not in REDUCERS:
        raise ValueError(f"reducer must be one of {REDUCERS}, got {reducer!r}")

    if upscale_factor > 1:
        # Upscaling gives sub-pixel tracking on tight/noisy curves before any reducer runs.
        # Nearest-neighbor keeps the mask strictly binary (no interpolated edge opacity).
        h, w = mask.shape
        mask = cv2.resize(
            mask.astype(np.uint8),
            (w * upscale_factor, h * upscale_factor),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        if bbox is not None:
            bbox = tuple(v * upscale_factor for v in bbox)
        dx *= upscale_factor
        window_size *= upscale_factor
        if seed_y is not None:
            seed_y *= upscale_factor
        if seed_x is not None:
            seed_x *= upscale_factor
        if max_jump is not None:
            max_jump *= upscale_factor

    h, w = mask.shape
    if bbox is None:
        x_min, y_min, x_max, y_max = 0, 0, w - 1, h - 1
    else:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)

    if reducer == "trace":
        xs, ys = _extract_trace(mask, x_min, y_min, x_max, y_max, seed_y, seed_x, dx, max_jump)
    elif reducer == "centroid":
        xs, ys = _extract_centroid(mask, x_min, y_min, x_max, y_max, seed_y, seed_x, dx, window_size)
    else:
        xs, ys = _extract_column_scan(mask, x_min, y_min, x_max, y_max, seed_y, dx, reducer)

    xs_out = np.asarray(xs, dtype=float)
    ys_out = np.asarray(ys, dtype=float)

    if upscale_factor > 1:
        xs_out /= upscale_factor
        ys_out /= upscale_factor

    if end_x is not None:
        keep = xs_out <= end_x
        xs_out = xs_out[keep]
        ys_out = ys_out[keep]

    # Force the output to literally touch the user-picked start/end points,
    # rather than whichever mask pixel happened to be nearest - a seed/end
    # point is meant to pin exactly where the curve begins/ends.
    if seed_x is not None and seed_y is not None:
        if xs_out.size == 0 or xs_out[0] != seed_x or ys_out[0] != seed_y:
            xs_out = np.insert(xs_out, 0, float(seed_x))
            ys_out = np.insert(ys_out, 0, float(seed_y))

    if end_x is not None and end_y is not None:
        if xs_out.size == 0 or xs_out[-1] != end_x or ys_out[-1] != end_y:
            xs_out = np.append(xs_out, float(end_x))
            ys_out = np.append(ys_out, float(end_y))

    return xs_out, ys_out
