# Global Data Flow Refactoring Design

**Date:** 2026-06-25
**Status:** Approved

## Goal

Refactor tool functions to read inputs from and write outputs to global arrays directly, eliminating the intermediate collection logic in `pipeline_detection`. Tool functions and drawing functions both read from globals.

## Architecture

### Data Flow

```
process_image
  │
  ├─ set _box, _image_size
  │
  └─ pipeline_detection (thin orchestrator)
       │
       ├─ cv2.HoughLinesP → _raw_lines
       │
       ├─ extend_lines()
       │     reads:    _raw_lines
       │     writes:   _extended_lines
       │
       ├─ merge_lines()
       │     reads:    _extended_lines
       │     writes:   _merged_lines
       │
       ├─ find_all_intersections()
       │     reads:    _merged_lines
       │     writes:   _intersection_points
       │
       ├─ merge_points()
       │     reads:    _intersection_points
       │     writes:   _merged_points
       │
       ├─ assign_red_segments()
       │     reads:    _merged_lines, _merged_points
       │     writes:   _red_segments
       │
       └─ extend_independent_points()
             reads:    _merged_points, _merged_lines, _box
             writes:   _extension_lines

draw_all_elements:
  reads: _raw_lines, _merged_lines, _red_segments,
         _intersection_points, _merged_points, _extension_lines
```

### Global State

```python
_raw_lines = []           # list of ((x1,y1), (x2,y2)) from HoughLinesP
_extended_lines = []     # list of ((x1,y1), (x2,y2)) after extension
_merged_lines = []       # list of ((x1,y1), (x2,y2)) after merging
_intersection_points = [] # list of (x, y) raw intersections
_merged_points = []      # list of (x, y) merged intersections
_red_segments = []       # list of ((x1,y1), (x2,y2)) between intersection points
_extension_lines = []     # list of ((x1,y1), (x2,y2)) extension lines
_edges = []               # Canny edge image (debug)
_box = None              # np.intp 4x2 or None
_image_size = (0, 0)     # (w, h)
```

### Function Changes

| Function | Before | After |
|----------|--------|-------|
| `extend_line_nx(p1, p2, factor)` | Takes params, returns pair | `extend_lines()` no params, reads `_raw_lines`, writes `_extended_lines` |
| `merge_adjacent_lines(lines, ...)` | Takes params, returns list | `merge_lines()` no params, reads `_extended_lines`, writes `_merged_lines` |
| `segment_intersection(p1,p2,p3,p4)` | Pure function, single pair | `find_all_intersections()` no params, reads `_merged_lines`, writes `_intersection_points` |
| `merge_intersection_points(points, ...)` | Takes params, returns list | `merge_points()` no params, reads `_intersection_points`, writes `_merged_points` |
| (new) `assign_red_segments()` | Logic in `pipeline_detection` | Reads `_merged_lines` + `_merged_points`, writes `_red_segments` |
| (new) `extend_independent_points()` | Logic in `pipeline_detection` | Reads `_merged_points` + `_merged_lines` + `_box`, writes `_extension_lines` |
| `pipeline_detection` | Collects returns, writes globals | Thin orchestrator: reset globals, call stages in order, return stats |
| `draw_all_elements` | Reads globals | Unchanged (already reads globals) |
| `point_on_segment`, `line_intersection`, `point_to_line_distance`, `clip_ray_to_line` | Pure helpers | Unchanged (no global I/O) |

### Return Values

Tool functions become void (no return). `pipeline_detection` returns `(line_count, merged_count, intersection_count)` for debug printing.

## Error Handling

- `_reset_drawing_globals()` called at start of `pipeline_detection` (prevents cross-image contamination)
- Each tool function guards empty input: `if not _raw_lines: return`
- `_box is None` skips box-filtering logic (current behavior preserved)
- `_image_size` set in `process_image` before pipeline runs

## Testing

- Keep existing `_self_check_clip_ray()`
- Add `_self_check_global_flow()`: populate fake `_raw_lines` → call `extend_lines()` → assert `_extended_lines` has correct count
- Add `_self_check_merge_flow()`: populate fake `_extended_lines` → call `merge_lines()` → assert `_merged_lines` populated

## Scope

- Single file: `cube-detection-01.py`
- No new dependencies
- No changes to `process_image`, `find_contours`, `draw_min_area_rects`, `create_composite`, `main`
