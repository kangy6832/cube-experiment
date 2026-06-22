# Filter Red Intersection Points by Yellow Bounding Box

**Date:** 2026-06-22
**Status:** Approved

## Problem

`pipeline_detection()` draws all merged intersection points as red dots on the result image, regardless of whether they fall inside the yellow bounding rectangle. The yellow rectangle is drawn afterward by `draw_min_area_rects()` in `process_image()`, so the red points are computed before the box is known.

## Goal

Only draw red intersection points (and red connecting lines between them) that fall inside the yellow rotated bounding rectangle.

## Approach

Compute the box once in `process_image()`, pass it into `pipeline_detection()`, and filter at draw time using `cv2.pointPolygonTest`.

## Design

### 1. New helper: `_compute_min_area_box(contours, extend_px=0)`

Extract the pure-computation portion of `draw_min_area_rects` into a standalone function:

- Find the largest contour by area
- Compute `cv2.minAreaRect(largest)` → `(cx, cy), (w, h), angle`
- Compute `cv2.boxPoints(rect)` → 4x2 float array
- If `extend_px > 0`, scale each vertex outward from `(cx, cy)` by `1 + extend_px / dist`
- Return `np.intp(box)` or `None` if no contours

### 2. Modify `draw_min_area_rects`

- Call `_compute_min_area_box(contours, extend_px)` internally
- Draw with `cv2.drawContours` as before
- Return the box (so callers can reuse it if needed)

### 3. Modify `pipeline_detection(image, morphed_mask, box=None)`

New optional parameter `box` (4x2 `np.intp` vertices of the yellow rectangle, or `None`).

**Red point drawing (lines 436–438):**
```python
for x, y in merged_points:
    if box is not None and cv2.pointPolygonTest(box, (x, y), False) < 0:
        continue
    cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
    cv2.circle(result, (x, y), 5, (0, 0, 255), 2)
```

**Red line drawing (lines 418–433):**
```python
# Only draw if both endpoints are inside the box
if box is not None:
    if cv2.pointPolygonTest(box, p_a, False) < 0 or \
       cv2.pointPolygonTest(box, p_b, False) < 0:
        continue  # skip this line
cv2.line(result, p_a, p_b, (0, 0, 255), 2)
```

When `box is None`, behavior is unchanged (backward compatible).

### 4. Wire in `process_image()`

```python
# Step 4: Contour detection (unchanged)
result_contours, candidate_count, contours_list = find_contours(image, morphed)

# Step 5: Compute box BEFORE drawing red points
box = _compute_min_area_box(contours_list, extend_px=RECT_EXTEND_PX)

# Step 6: Pass box to pipeline_detection for filtering
result_lines, raw_lines_img, edges, line_count, raw_lines, \
    intersection_count, merged_count = pipeline_detection(
        image, morphed, box=box)

# Step 7: Draw yellow box on result_lines (unchanged)
draw_min_area_rects(result_lines, contours_list, extend_px=RECT_EXTEND_PX)
```

`find_contours` result is reused — no extra contour detection pass.

## Files Changed

- `cube-detection-01.py` — all changes in this file

## Test Plan

- Run the pipeline on existing photos in `photos/`
- Verify: red dots only appear inside the yellow rectangle
- Verify: red lines only connect dots inside the yellow rectangle
- Verify: when no cube contour is found (box=None), all red dots are drawn as before
