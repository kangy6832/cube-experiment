# Merge Adjacent Blue Lines Design

**Date:** 2026-06-22
**Status:** Approved

## Overview

Merge adjacent/near-parallel blue lines (extended Hough lines) into single lines in `cube-detection-01.py`. When Hough Line Transform detects multiple small segments belonging to the same edge, their extended versions appear as multiple near-parallel blue lines. This feature merges them into one clean line.

## Requirements

- **R1:** Merge blue lines that are near-parallel (angle difference < threshold) AND close in distance (perpendicular distance < threshold)
- **R2:** Merged line endpoints = the two extreme endpoints among all constituent segments (their union)
- **R3:** Red intersection points and red segments must be recalculated based on merged lines
- **R4:** Thresholds must be configurable constants with defaults: angle < 10 degrees, distance < 20 pixels

## Architecture

### New Configuration Constants

```python
# Line merging thresholds
ANGLE_THRESHOLD = 10        # degrees
DIST_THRESHOLD = 20         # pixels
```

Added in the Configuration section, alongside existing Hough parameters.

### New Function: `merge_adjacent_lines(extended_lines, angle_thresh, dist_thresh)`

**Input:** List of extended line tuples `[(ext1, ext2), ...]`

**Algorithm (Polar Coordinate Clustering):**

1. **Polar Transform** — For each line `(p1, p2)`:
   - Compute direction angle: `theta = atan2(dy, dx)` normalized to `[0, pi)` (lines are undirected)
   - Compute normal distance from origin: `rho = |x1*y2 - x2*y1| / sqrt(dx^2 + dy^2)`

2. **Primary Clustering by Angle:**
   - Sort lines by `theta`
   - Group consecutive lines where adjacent angle difference < `angle_thresh`
   - Handle wrap-around: `theta` near 0 and near pi represent nearly parallel lines; treat `|theta_a - theta_b|` with pi-wrapping: `min(diff, pi - diff)`

3. **Secondary Clustering by Distance:**
   - Within each angle group, sort by `rho`
   - Group consecutive lines where the gap between adjacent rho values < `dist_thresh` (1D clustering on rho)

4. **Merge Each Cluster:**
   - Collect all endpoints from lines in the cluster
   - Fit dominant direction using the cluster's median angle
   - Project all endpoints onto the direction vector
   - Take the two endpoints with min and max projection values
   - Return as the merged line `(p_min, p_max)`

**Output:** List of merged line tuples in same format as input

### Modified Function: `pipeline_detection()`

**Changes:**

1. After computing `extended_lines`, call `merge_adjacent_lines()` to get `merged_lines`
2. Draw blue lines from `merged_lines` instead of `extended_lines`
3. Use `merged_lines` for intersection point calculation (replace all references to `extended_lines` in the intersection loop)
4. Print count of merged lines for debugging

**Visual output changes:**
- Fewer, cleaner blue lines
- Red intersection points and segments recalculated based on merged lines

### Unchanged

- `lab_threshold()` — no changes
- `morphological_processing()` — no changes
- `extend_line_2x()` — no changes (still used to create initial extended lines)
- `segment_intersection()`, `merge_intersection_points()`, `point_on_segment()` — no changes
- `find_contours()` — no changes
- `create_composite()` — no changes

## Data Flow

```
HoughLinesP -> raw lines
    |
    v
extend_line_2x -> extended_lines (all 2x segments)
    |
    v
merge_adjacent_lines -> merged_lines (clustered & merged)
    |
    v
Draw blue lines (from merged_lines)
    |
    v
segment_intersection (on merged_lines) -> raw_points
    |
    v
merge_intersection_points -> merged_points
    |
    v
Draw red segments and dots (from merged_lines x merged_points)
```

## Error Handling

- Empty `extended_lines`: return empty list
- Single line: return as-is (no merging needed)
- All lines parallel but too far apart: no merging, return original
- All lines merge into one: return single line

## Testing

- Run existing script against photos in `/home/kangy/MyProjects/cube-experiment/photos/`
- Verify: fewer blue lines drawn, lines are cleaner
- Verify: red intersection points align with merged blue line intersections
- Verify: output images saved correctly to `output/01/` and `output/lines/`
