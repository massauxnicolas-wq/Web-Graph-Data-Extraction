from __future__ import annotations

import numpy as np
import cv2
from scipy.spatial import KDTree

def _runs_in_column(col: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_inclusive) index ranges where col is True."""
    if not col.any():
        return []
    diff = np.diff(col.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


# Added 'trace' to the allowed reducers
REDUCERS = ("mean", "midpoint", "centroid", "trace")


def extract_curve(
    mask: np.ndarray,
    dx: int = 2,
    bbox: tuple[int, int, int, int] | None = None,
    seed_y: int | None = None,
    reducer: str = "mean",
    max_jump: float | None = 50.0,
    window_size: int = 20,
    upscale_factor: int = 1,  # NEW: Sub-pixel precision for high-frequency curves
) -> tuple[np.ndarray, np.ndarray]:
    """Column-scan (or path-trace) a boolean mask and return pixel xs, ys for one curve."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if dx < 1:
        raise ValueError("dx must be >= 1")
    if reducer not in REDUCERS:
        raise ValueError(f"reducer must be one of {REDUCERS}, got {reducer!r}")

    # --- HIGHER FREQUENCY HANDLING ---
    # If the curve is tight/noisy, upscaling the mask gives us sub-pixel 
    # tracking capability before we run any of the reducers.
    if upscale_factor > 1:
        h, w = mask.shape
        # Nearest neighbor perfectly preserves the 100% binary nature of the mask without opacity fading
        mask = cv2.resize(
            mask.astype(np.uint8), 
            (w * upscale_factor, h * upscale_factor), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Scale bounding box and parameters to match
        if bbox is not None:
            bbox = tuple(v * upscale_factor for v in bbox)
        dx *= upscale_factor
        window_size *= upscale_factor
        if seed_y is not None:
            seed_y *= upscale_factor
        if max_jump is not None:
            max_jump *= upscale_factor

    h, w = mask.shape
    if bbox is None:
        x_min, y_min, x_max, y_max = 0, 0, w - 1, h - 1
    else:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)

    xs: list[float] = []
    ys: list[float] = []

    # =========================================================
    # REDUCER: TRACE (Best for Steep / Vertical / Loops)
    # =========================================================
    if reducer == "trace":
        # Extract all points in the ROI
        y_idx, x_idx = np.where(mask[y_min:y_max+1, x_min:x_max+1])
        if len(x_idx) == 0:
            return np.asarray([]), np.asarray([])
            
        x_idx = x_idx + x_min
        y_idx = y_idx + y_min
        
        points = np.column_stack((x_idx, y_idx))
        
        # Determine starting point (left-most point, matching seed if provided)
        if seed_y is not None:
            # Find the point closest to the seed on the left edge
            left_points = points[points[:, 0] == np.min(points[:, 0])]
            start_idx = np.argmin(np.abs(left_points[:, 1] - seed_y))
            # Get original index
            start_point = left_points[start_idx]
            curr = np.where((points == start_point).all(axis=1))[0][0]
        else:
            curr = np.argmin(x_idx)

        # Build a KDTree for rapid nearest-neighbor pathfinding
        tree = KDTree(points)
        unvisited = set(range(len(points)))
        
        path_x, path_y = [], []
        step_radius = max(1.0, float(dx))
        
        while unvisited:
            path_x.append(float(points[curr][0]))
            path_y.append(float(points[curr][1]))
            
            # Clear nearby points to prevent getting stuck in local thickness (zigzag loop)
            nearby = tree.query_ball_point(points[curr], r=step_radius)
            unvisited.difference_update(nearby)
            
            if not unvisited:
                break
                
            # Query enough neighbors to look past the cleared ball
            k_search = min(int(4 * step_radius**2) + 50, len(points))
            distances, indices = tree.query(points[curr], k=k_search)
            
            next_curr = None
            for dist, idx in zip(distances, indices):
                if idx in unvisited:
                    # Enforce forward progression: do not double-back horizontally by more than the step_radius
                    if points[idx][0] < points[curr][0] - step_radius:
                        continue
                        
                    if max_jump is not None and dist > max_jump:
                        continue 
                    next_curr = idx
                    break
                    
            if next_curr is None:
                # Fallback: search wider if gap is large but within max_jump
                k_search_wide = min(k_search * 5, len(points))
                distances, indices = tree.query(points[curr], k=k_search_wide)
                for dist, idx in zip(distances, indices):
                    if idx in unvisited:
                        # Enforce forward progression on fallback too
                        if points[idx][0] < points[curr][0] - step_radius:
                            continue
                            
                        if max_jump is not None and dist > max_jump:
                            continue
                        next_curr = idx
                        break
                        
            if next_curr is None:
                # Path broke or gap too large
                break
            curr = next_curr
            
        xs = path_x
        ys = path_y

    # =========================================================
    # REDUCER: CENTROID (Fixed Arc-Length Stepping)
    # =========================================================
    elif reducer == "centroid":
        cur_y = None
        if seed_y is not None:
            cur_y = float(seed_y)
            cur_x = float(x_min)
        else:
            for x in range(x_min, x_max + 1):
                col = mask[y_min:y_max + 1, x]
                if col.any():
                    cur_x = float(x)
                    cur_y = float(np.median(np.where(col)[0])) + y_min
                    break
            if cur_y is None:
                return np.asarray([]), np.asarray([])
        
        dir_x, dir_y = 1.0, 0.0
        
        while x_min <= cur_x <= x_max and y_min <= cur_y <= y_max:
            x0, x1 = int(max(x_min, cur_x - window_size//2)), int(min(x_max, cur_x + window_size//2))
            y0, y1 = int(max(y_min, cur_y - window_size//2)), int(min(y_max, cur_y + window_size//2))
            
            window = mask[y0:y1+1, x0:x1+1]
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
                        
                    # BUGFIX: Step along the 2D vector, not just forcing X to increment.
                    step_x = dir_x * dx
                    step_y = dir_y * dx
                else:
                    step_x, step_y = float(dx), 0.0
            else:
                step_x, step_y = float(dx), 0.0
            
            # This allows steep ascents because if dir_x is 0 (vertical), step_x is 0.
            cur_x += step_x
            cur_y += step_y
            
            # Break condition to prevent getting stuck in a single spot
            if abs(step_x) < 0.1 and abs(step_y) < 0.1:
                break

    # =========================================================
    # REDUCER: MEAN & MIDPOINT
    # =========================================================
    else:
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
                best_y = min(midpoints, key=lambda m, t=target: abs(m - t))
                y = best_y

            xs.append(float(x))
            ys.append(y)
            last_y = y

    xs_out = np.asarray(xs, dtype=float)
    ys_out = np.asarray(ys, dtype=float)

    # --- RETURN TO ORIGINAL SCALE ---
    if upscale_factor > 1:
        xs_out /= upscale_factor
        ys_out /= upscale_factor

    return xs_out, ys_out