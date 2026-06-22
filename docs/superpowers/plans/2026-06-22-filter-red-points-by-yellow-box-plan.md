# Filter Red Intersection Points by Yellow Bounding Box — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Only draw red intersection points and red connecting lines that fall inside the yellow rotated bounding rectangle.

**Architecture:** Extract box computation into a shared helper, pass the box into `pipeline_detection`, and filter at draw time with `cv2.pointPolygonTest`. Single file change.

**Tech Stack:** Python, OpenCV (`cv2.pointPolygonTest`, `cv2.minAreaRect`, `cv2.boxPoints`)

## Global Constraints

- Single file: `cube-detection-01.py`
- `box=None` default preserves backward compatibility — all existing callers unchanged
- `pointPolygonTest(box, pt, False) >= 0` is the inside-or-on-boundary check
- Red points AND red connecting lines are both filtered
- No new dependencies

---

### Task 1: Add `_compute_min_area_box` helper

**Files:**
- Modify: `cube-detection-01.py` (insert before `draw_min_area_rects`, around line 468)

**Interfaces:**
- Consumes: `contours` (list from `cv2.findContours`), `extend_px` (int)
- Produces: `np.intp` 4x2 array of box vertices, or `None` if no contours

- [ ] **Step 1: Write a quick inline self-check**

Add at the bottom of the file, inside `if __name__ == "__main__":` block (temporary, removed after):

```python
# TEMP: sanity check _compute_min_area_box
import numpy as np
# Synthetic contour: a 100x50 rectangle at (200, 150)
pts = np.array([[[200,150]], [[300,150]], [[300,200]], [[200,200]]], dtype=np.int32)
box = _compute_min_area_box([pts], extend_px=0)
assert box is not None, "box should not be None"
assert box.shape == (4, 2), f"expected (4,2), got {box.shape}"
print("PASS: _compute_min_area_box basic case")
```

- [ ] **Step 2: Run the self-check**

```bash
cd /home/kangy/MyProjects/cube-experiment && python cube-detection-01.py 2>&1 | tail -5
```

Expected: either "PASS" or "Error: No images found" (both acceptable — the function is called only if images exist). If `NameError: _compute_min_area_box` → function not yet defined, expected.

- [ ] **Step 3: Implement `_compute_min_area_box`**

Insert before `draw_min_area_rects` (before line 468):

```python
def _compute_min_area_box(contours, extend_px=0):
    """
    Compute the minimum-area rotated bounding rectangle of the largest contour.

    Args:
        contours: list of contours from cv2.findContours
        extend_px: pixels to extend the rectangle outward from each edge

    Returns:
        np.intp 4x2 array of box vertices, or None if no contours
    """
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    try:
        rect = cv2.minAreaRect(largest)
        (cx, cy), _, _ = rect
        box = cv2.boxPoints(rect).astype(np.float64)

        if extend_px > 0:
            for i in range(4):
                px, py = box[i]
                dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                if dist > 1e-10:
                    scale = 1 + extend_px / dist
                    box[i] = (cx + scale * (px - cx), cy + scale * (py - cy))

        return np.intp(box)
    except Exception as e:
        print(f"    Warning: minAreaRect failed for contour: {e}")
        return None
```

- [ ] **Step 4: Run self-check again**

```bash
cd /home/kangy/MyProjects/cube-experiment && python cube-detection-01.py 2>&1 | tail -5
```

Expected: "PASS: _compute_min_area_box basic case" (if photos exist and pipeline runs) or at minimum no `NameError`.

- [ ] **Step 5: Commit**

```bash
git add cube-detection-01.py
git commit -m "feat: add _compute_min_area_box helper"
```

---

### Task 2: Refactor `draw_min_area_rects` to use the helper

**Files:**
- Modify: `cube-detection-01.py` (the `draw_min_area_rects` function, around line 480)

**Interfaces:**
- Consumes: same as before (`image`, `contours`, `color`, `thickness`, `extend_px`)
- Produces: same as before (image reference) — no return value change

- [ ] **Step 1: Replace the body of `draw_min_area_rects`**

Current body (lines 485–504) computes `largest`, `rect`, `box`, `extend` inline. Replace with a call to `_compute_min_area_box`:

```python
def draw_min_area_rects(image, contours, color=(0, 255, 255), thickness=2, extend_px=0):
    """
    Draw the minimum-area rotated bounding rectangle around the largest contour.
    ...
    """
    box = _compute_min_area_box(contours, extend_px)
    if box is None:
        return image

    cv2.drawContours(image, [box], 0, color, thickness)
    return image
```

- [ ] **Step 2: Run the pipeline to verify no regression**

```bash
cd /home/kangy/MyProjects/cube-experiment && python cube-detection-01.py 2>&1 | tail -20
```

Expected: same output as before — yellow rectangles drawn, no errors. Visually spot-check `output/01/*_04_lines.png` — yellow boxes should appear in the same positions.

- [ ] **Step 3: Commit**

```bash
git add cube-detection-01.py
git commit -m "refactor: draw_min_area_rects uses _compute_min_area_box"
```

---

### Task 3: Add `box` parameter to `pipeline_detection` and filter red points

**Files:**
- Modify: `cube-detection-01.py` (function signature at line 344, red point drawing at lines 436–438, red line drawing at lines 418–433)

**Interfaces:**
- Consumes: new `box=None` parameter (np.intp 4x2 or None)
- Produces: same return tuple — no change to callers when `box=None`

- [ ] **Step 1: Update function signature**

```python
def pipeline_detection(image, morphed_mask, box=None):
```

- [ ] **Step 2: Filter red connecting lines (lines 418–433)**

Wrap the `cv2.line(...)` calls inside the red-line block with a box check. The block currently is:

```python
if len(points_on_line) >= 2:
    if len(points_on_line) == 2:
        cv2.line(result, points_on_line[0], points_on_line[1], (0, 0, 255), 2)
    else:
        # ... find farthest pair p_a, p_b ...
        cv2.line(result, p_a, p_b, (0, 0, 255), 2)
```

Replace with:

```python
if len(points_on_line) >= 2:
    if len(points_on_line) == 2:
        p_a, p_b = points_on_line[0], points_on_line[1]
    else:
        # ... find farthest pair p_a, p_b (unchanged) ...
    # Only draw if both endpoints inside box (or no box)
    if box is None or (
        cv2.pointPolygonTest(box, p_a, False) >= 0 and
        cv2.pointPolygonTest(box, p_b, False) >= 0
    ):
        cv2.line(result, p_a, p_b, (0, 0, 255), 2)
```

- [ ] **Step 3: Filter red dots (lines 436–438)**

Replace:

```python
for x, y in merged_points:
    cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
    cv2.circle(result, (x, y), 5, (0, 0, 255), 2)
```

With:

```python
for x, y in merged_points:
    if box is not None and cv2.pointPolygonTest(box, (x, y), False) < 0:
        continue
    cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
    cv2.circle(result, (x, y), 5, (0, 0, 255), 2)
```

- [ ] **Step 4: Run the pipeline and verify**

```bash
cd /home/kangy/MyProjects/cube-experiment && python cube-detection-01.py 2>&1 | tail -20
```

Expected: no errors. Check `output/01/*_04_lines.png` — red dots should only appear inside yellow rectangles. Dots outside should be gone.

- [ ] **Step 5: Commit**

```bash
git add cube-detection-01.py
git commit -m "feat: filter red intersection points by yellow bounding box"
```

---

### Task 4: Wire `process_image` — compute box before `pipeline_detection`

**Files:**
- Modify: `cube-detection-01.py` (`process_image` function, around lines 534–549)

**Interfaces:**
- `find_contours` result already returns `contours_list` — reuse it
- `_compute_min_area_box` is called once, result passed to `pipeline_detection`

- [ ] **Step 1: Reorder and wire in `process_image`**

Current order (simplified):
```
Step 3: pipeline_detection → result_lines
Step 4: find_contours → contours_list
Step 5: draw_min_area_rects(result_lines, contours_list)
```

New order:
```
Step 3: find_contours → contours_list
Step 4: box = _compute_min_area_box(contours_list, extend_px=RECT_EXTEND_PX)
Step 5: pipeline_detection(image, morphed, box=box) → result_lines
Step 6: draw_min_area_rects(result_lines, contours_list, extend_px=RECT_EXTEND_PX)
```

Replace the relevant section in `process_image` (around lines 534–549):

```python
    # Step 4: Contour detection for cube candidates
    result_contours, candidate_count, contours_list = find_contours(image, morphed)
    print(f"  Cube candidates: {candidate_count}")

    # Step 5: Compute bounding box before drawing red points
    box = _compute_min_area_box(contours_list, extend_px=RECT_EXTEND_PX)

    # Step 6: Pipeline (Hough Line) detection — pass box to filter red points
    result_lines, raw_lines_img, edges, line_count, raw_lines, \
        intersection_count, merged_count = pipeline_detection(
            image, morphed, box=box)
    print(f"  Hough Lines detected: {line_count}")
    print(f"  Lines after merging: {merged_count}")
    print(f"  Intersection points (merged): {intersection_count}")

    # Step 7: Draw yellow minimum-area bounding boxes on result_lines
    draw_min_area_rects(result_lines, contours_list, extend_px=RECT_EXTEND_PX)
```

Note: `result_contours` is still computed (it's saved to disk at line 559) but `contours_list` is now obtained from Step 4 instead of being computed separately. The `find_contours` call moves from after `pipeline_detection` to before it.

- [ ] **Step 2: Run the full pipeline end-to-end**

```bash
cd /home/kangy/MyProjects/cube-experiment && python cube-detection-01.py 2>&1 | tail -30
```

Expected: no errors. Visually verify `output/01/*_04_lines.png`:
- Red dots only inside yellow boxes
- Yellow boxes unchanged
- Blue/green lines unchanged

- [ ] **Step 3: Remove the temporary self-check from Task 1**

Delete the `if __name__ == "__main__":` block's TEMP sanity check (it was only for Task 1 verification). If `main()` is already in that block, only remove the TEMP lines, keep `main()`.

- [ ] **Step 4: Final run to confirm clean**

```bash
cd /home/kangy/MyProjects/cube-experiment && python cube-detection-01.py 2>&1 | tail -30
```

Expected: clean output, no TEMP prints, no errors.

- [ ] **Step 5: Commit**

```bash
git add cube-detection-01.py
git commit -m "feat: wire process_image to filter red points by yellow box"
```
