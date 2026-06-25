# Cube Detection — File Split Design

**Date:** 2026-06-25
**Status:** Approved

## Goal

Split the monolithic `cube-detection-01.py` (~924 lines) into a package of focused modules under `cube-detection-01/`. No behavioral changes — purely structural.

## Target Structure

```
cube-detection-01/
├── config.py      # All module-level constants
├── state.py       # Global drawing element storage + _reset_drawing_globals()
├── geometry.py    # Pure geometry helpers (no global I/O)
├── color.py       # Lab thresholding + morphological processing
├── contours.py    # Contour detection + min-area box computation
├── lines.py       # Line extend/merge/intersect/assign logic
└── pipeline.py    # pipeline_detection, drawing, composite, process_image, main, self-checks
```

No `__init__.py` — Python 3.3+ namespace package.

## File Contents

### `config.py`
All constants from original lines 14–40: `PHOTOS_DIR`, `OUTPUT_DIR`, `LINES_DIR`, `LAB_LOWER`, `LAB_UPPER`, `MORPH_KERNEL_SIZE`, `MORPH_ITERATIONS`, `HOUGH_RHO`, `HOUGH_THETA`, `HOUGH_THRESHOLD`, `HOUGH_MIN_LINE_LENGTH`, `HOUGH_MAX_LINE_GAP`, `LINE_EXTEND_FACTOR`, `RECT_EXTEND_PX`, `EXTEND_LENGTH`, `ANGLE_THRESHOLD`, `DIST_THRESHOLD`.

### `state.py`
- Global variable declarations: `_raw_lines`, `_extended_lines`, `_merged_lines`, `_intersection_points`, `_merged_points`, `_red_segments`, `_extension_lines`, `_edges`, `_box`, `_image_size`, `_excluded_points`, `_red_endpoints`
- `_reset_drawing_globals()` function
- Module-level `_reset_drawing_globals()` call (runs on import)

### `geometry.py`
Pure functions, no global I/O:
- `_line_intersection_pair(p1, p2, p3, p4)`
- `_segment_intersection_pair(p1, p2, p3, p4)`
- `point_on_segment(px, py, x1, y1, x2, y2, tol=2)`
- `point_to_line_distance(px, py, x1, y1, x2, y2)`
- `clip_ray_to_box(origin, direction, box)`

### `color.py`
- `lab_threshold(image)` — LAB color space thresholding
- `morphological_processing(mask)` — open/close/dilate pipeline

### `contours.py`
- `find_contours(image, morphed_mask)` — contour detection + filtering
- `_compute_min_area_box(contours, extend_px=0)` — min-area rotated rect
- `draw_min_area_rects(image, contours, color, thickness, extend_px)` — drawing utility

### `lines.py`
All line processing stages:
- `extend_lines()` — extend raw Hough lines from midpoint
- `merge_lines()` — polar-coordinate clustering merge
- `find_all_intersections()` — pairwise segment intersections
- `merge_points()` — 10px centroid clustering
- `assign_red_segments()` — connect intersection points along merged lines
- `filter_intersection_points()` — remove excluded + out-of-box points
- `extend_independent_points()` — extend rays from orphan points toward box center
- `_is_independent_blue(blue_line)` — check if blue line has no red coverage

### `pipeline.py`
Orchestration + entry point:
- `create_output_dir()`
- `pipeline_detection(image, morphed_mask, box=None)` — main detection pipeline
- `draw_all_elements(image, box=None)` — draw all detected elements
- `create_composite(image, mask, morphed, edges, lines_img, contours_img)` — 2x3 visualization
- `process_image(image_path, idx)` — single image processing
- `main()` — entry point
- `_self_check_global_flow()`, `_self_check_clip_ray()`, `_self_check_intersections()` — smoke tests

## Import Graph

```
config.py       ← (no internal imports)
state.py        ← (no internal imports)
geometry.py     ← config
color.py        ← config, state
contours.py     ← config, state
lines.py        ← config, state, geometry
pipeline.py     ← config, state, geometry, color, contours, lines
```

## Migration Rules

1. Each function moves verbatim — no logic changes
2. Cross-file references become explicit imports (`from .config import ...`)
3. No `from x import *`
4. Global state pattern preserved exactly — `state.py` centralizes it
5. Original `cube-detection-01.py` kept as backup
6. Self-checks still run before `main()` in `if __name__ == "__main__"`

## Verification

1. Syntax check all 7 files with `py_compile`
2. Self-check functions produce same PASS output
3. Pipeline on known image produces pixel-identical output (no logic change)
4. `import cube_detection_01` works (namespace package)
