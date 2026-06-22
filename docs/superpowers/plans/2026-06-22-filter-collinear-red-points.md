# Filter Collinear Red Points Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When 3+ red intersection points are collinear on a merged line, only draw the two endpoint dots.

**Architecture:** Add an `excluded_points` set populated during the per-line loop (when 3+ points are collinear), then skip those points in the global dot-drawing loop.

**Tech Stack:** Python, OpenCV (cv2), numpy

## Global Constraints

- All changes in `cube-detection-01.py` only
- No new dependencies
- Backward compatible: when no 3+ collinear points exist, behavior is unchanged

---

### Task 1: Implement collinear red point filtering

**Files:**
- Modify: `cube-detection-01.py` (function `pipeline_detection`)

**Interfaces:**
- Consumes: existing `merged_points`, `merged_lines`, `box` variables in `pipeline_detection()`
- Produces: same return signature — no change to function signature

- [ ] **Step 1: Add `excluded_points` set initialization**

After line 412 (`merged_points = merge_intersection_points(raw_points, merge_radius=10)`), add:

```python
# Track intermediate collinear points to exclude from drawing
excluded_points = set()
```

- [ ] **Step 2: Add exclusion logic in per-line loop**

Inside the existing `for ext_line in merged_lines:` loop (line 416), after the farthest-pair `p_a, p_b` are found and before the `cv2.line(...)` call, add:

```python
# If 3+ points on this line, collect intermediate points for exclusion
if len(points_on_line) >= 3:
    for p in points_on_line:
        if p != p_a and p != p_b:
            excluded_points.add(p)
```

- [ ] **Step 3: Add skip check in global dot-drawing loop**

Modify the loop at lines 446-449 from:

```python
for x, y in merged_points:
    if box is not None and cv2.pointPolygonTest(box, (x, y), False) < 0:
        continue
    cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
    cv2.circle(result, (x, y), 5, (0, 0, 255), 2)
```

to:

```python
for x, y in merged_points:
    if (x, y) in excluded_points:
        continue
    if box is not None and cv2.pointPolygonTest(box, (x, y), False) < 0:
        continue
    cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
    cv2.circle(result, (x, y), 5, (0, 0, 255), 2)
```

- [ ] **Step 4: Run the pipeline to verify**

```bash
cd /home/kangy/MyProjects/cube-experiment && python cube-detection-01.py
```

Expected output: Processing completes without errors. Visually inspect any output images where 3+ collinear intersection points exist — only the two endpoint red dots should be drawn.

- [ ] **Step 5: Commit**

```bash
git add cube-detection-01.py
git commit -m "feat: filter intermediate collinear red points"
```
