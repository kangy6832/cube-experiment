# Separate Detection From Drawing

## Goal

Refactor `cube-detection-01.py` so that all drawing elements (green lines, blue lines, red segments, red dots, extension lines) are stored in module-level data structures during detection/computation, then drawn in a single unified function at the end. The yellow bounding box remains drawn separately (unchanged).

## Motivation

Currently `pipeline_detection()` interleaves computation with drawing — green lines are drawn before merging, blue lines after merging, red segments after intersection computation, etc. This makes it hard to:
- Test detection logic without rendering
- Change drawing order or add new element types
- Reuse detection results with different visualizations

Separating "what to draw" from "how to draw" clarifies the pipeline.

## Architecture

### Module-Level Globals

```python
_raw_lines = []            # green: ((x1,y1), (x2,y2)) from HoughLinesP
_merged_lines = []         # blue: ((x1,y1), (x2,y2)) after extend+merge
_red_segments = []         # red: ((x1,y1), (x2,y2)) between intersection points
_intersection_points = []   # red dots: (x, y)
_excluded_points = set()   # intermediate collinear points to skip drawing
_red_endpoints = set()     # endpoints of red segments (skip their extension)
_extension_lines = []      # red extension: ((x1,y1), (x2,y2))
```

Reset at the start of each `pipeline_detection()` call to prevent cross-image contamination.

### Function Changes

| Function | Before | After |
|----------|--------|-------|
| `pipeline_detection()` | Detects + draws in-place, returns `(result, raw_lines_img, edges, line_count, raw_lines, intersection_count, merged_count)` | Pure computation, populates globals, returns `(edges, line_count, merged_count, intersection_count)` |
| `draw_all_elements(image, box=None)` | Does not exist | Draws all elements from globals in correct order |
| `process_image()` | Calls pipeline_detection (gets drawn image) → draws yellow box → saves | Calls pipeline_detection → draws yellow box → calls `draw_all_elements` → saves |

### Drawing Order in `draw_all_elements`

1. Green raw lines (`_raw_lines`)
2. Blue merged lines (`_merged_lines`)
3. Red segments (`_red_segments`)
4. Red dots (`_intersection_points` minus `_excluded_points`, clipped to box)
5. Red extension lines (`_extension_lines`)

Order matters: later draws overlay earlier ones, matching current visual output.

### Removed

- `raw_lines_img` output (separate green-line-only image in `LINES_RAW_DIR`) — green lines are now part of the final result
- `LINES_RAW_DIR` constant and its directory creation

### Unchanged

- All detection/thresholding/morphology/merging logic
- `find_contours`, `_compute_min_area_box`, `draw_min_area_rects`
- Output files: mask, morphed, edges, lines, contours, composite
- Composite image generation

## Data Flow in `process_image`

```
image
  → lab_threshold → mask
  → morphological_processing → morphed
  → find_contours → contours_list
  → _compute_min_area_box → box
  → pipeline_detection(image, morphed, box)    # fills globals, returns metadata
  → result = image.copy()
  → draw_min_area_rects(result, contours)      # yellow box
  → draw_all_elements(result, box)             # all line/point elements
  → save outputs
```

## Error Handling

- If `pipeline_detection` finds no lines, globals contain empty lists; `draw_all_elements` handles empty data gracefully (no-op draws).
- Box=None fallback preserved in `draw_all_elements` (draws without clipping).

## Testing

- Run against existing photos, verify output images are visually identical to pre-refactor.
- `_self_check_clip_ray()` still passes (unchanged).
