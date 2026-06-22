# Filter Collinear Red Points

**Date:** 2026-06-22
**Status:** Approved

## Problem

When a merged line has 3 or more collinear red intersection points, all of them are drawn as red dots. The user wants only the two endpoint dots (at the ends of the red segment) to be drawn — intermediate collinear points should be hidden.

## Goal

When 3+ red intersection points are collinear on a merged line, only keep the two dots at the endpoints of the segment (the farthest-apart pair). Intermediate points are globally excluded from drawing.

## Approach

Collect intermediate collinear points into an exclusion set during the per-line loop, then skip them in the global dot-drawing loop.

## Design

### 1. New local variable: `excluded_points`

Initialize an empty `set()` at the top of `pipeline_detection()`, after `merged_points` is computed.

### 2. Modify the per-line loop (lines 416-443)

Inside the existing `for ext_line in merged_lines:` loop, after finding `p_a, p_b` (the farthest-apart pair):

```python
# If 3+ points on this line, collect intermediate points for exclusion
if len(points_on_line) >= 3:
    for p in points_on_line:
        if p != p_a and p != p_b:
            excluded_points.add(p)
```

This runs only when there are 3+ collinear points. The `p_a, p_b` pair are the endpoints that define the red segment — they are kept; all others on that line are excluded.

### 3. Modify the global dot-drawing loop (lines 446-449)

Add an exclusion check before drawing each dot:

```python
for x, y in merged_points:
    if (x, y) in excluded_points:
        continue
    if box is not None and cv2.pointPolygonTest(box, (x, y), False) < 0:
        continue
    cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
    cv2.circle(result, (x, y), 5, (0, 0, 255), 2)
```

### 4. Edge cases

| Scenario | Behavior |
|----------|----------|
| 2 points on a line | No exclusion; both dots drawn (unchanged) |
| 3+ points on a line | Only the two endpoint dots drawn |
| Point is intermediate on line A but endpoint on line B | Globally excluded (user confirmed) |
| No cube contour (box=None) | Exclusion logic unaffected |

## Files Changed

- `cube-detection-01.py` — all changes in this file

## Test Plan

- Run the pipeline on existing photos in `photos/`
- Verify: on lines with 3+ collinear intersection points, only the two endpoint red dots are drawn
- Verify: on lines with exactly 2 intersection points, both dots are drawn (unchanged)
- Verify: when no cube contour is found (box=None), filtering still works correctly
