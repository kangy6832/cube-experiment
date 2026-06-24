# Red Line Extension Logic Change

**Date:** 2026-06-24
**Status:** Approved

## Context

In `cube-detection-01.py`, the `pipeline_detection()` function draws red extension lines from intersection points. Currently, it processes **every** merged intersection point that has exactly 2 blue lines and 1 red segment, drawing one extension per qualifying point.

The new logic categorizes intersection points into three groups and applies different extension behavior to each.

## Point Categories

| Category | Condition | Extension Behavior |
|---|---|---|
| Farthest-pair endpoint | Is an endpoint of a red segment | 1 extension along the "other" blue line (not the red segment's parent) |
| Independent red point | Not on any red segment at all | 2 extensions, one along each blue line through the point |
| Middle point | On a red segment but not an endpoint | No extension |

## Extension Rule (applies uniformly)

For each extension:
1. Direction: along the blue line, pointing toward the yellow bounding box centroid (positive dot product)
2. Clip to box boundary via `clip_ray_to_box()`
3. Cap at `EXTEND_LENGTH` (50px)

## Implementation

### Change 1: Track red segment endpoints

In the red segment drawing loop (lines ~498-503), add endpoints to a new set:

```python
red_endpoints = set()  # new

# Inside the loop, after drawing a red segment:
red_endpoints.add(p_a)
red_endpoints.add(p_b)
```

### Change 2: Rewrite extension loop (lines 514-590)

Replace the current `for mp in merged_points` loop with logic that branches on category:

```python
if box is not None and len(red_segments) > 0:
    centroid = np.mean(box, axis=0)

    for mp in merged_points:
        if cv2.pointPolygonTest(box, mp, False) < 0:
            continue

        # Classify point
        is_endpoint = mp in red_endpoints
        red_on = [seg for seg in red_segments
                  if point_on_segment(mp[0], mp[1], seg[0][0], seg[0][1],
                                     seg[1][0], seg[1][1], tol=3)]
        is_independent = len(red_on) == 0

        if not is_endpoint and not is_independent:
            continue  # middle point, skip

        # Find blue lines through this point
        blue_on = [ml for ml in merged_lines
                   if point_on_segment(mp[0], mp[1], ml[0][0], ml[0][1],
                                      ml[1][0], ml[1][1], tol=3)]

        if is_endpoint:
            # 1 extension: along the blue line that is NOT the red segment's parent
            red_seg = red_on[0]
            _, _, red_parent_line = red_seg
            other_blue = None
            for bl in blue_on:
                if bl is not red_parent_line:
                    other_blue = bl
                    break
            if other_blue is None:
                continue
            blue_lines_to_extend = [other_blue]
        else:
            # Independent: extend along ALL blue lines through the point
            blue_lines_to_extend = blue_on

        # Draw extension(s)
        for bl in blue_lines_to_extend:
            bx = bl[1][0] - bl[0][0]
            by = bl[1][1] - bl[0][1]
            blen = np.sqrt(bx * bx + by * by)
            if blen < 1e-10:
                continue
            dir_x = bx / blen
            dir_y = by / blen

            to_cx = centroid[0] - mp[0]
            to_cy = centroid[1] - mp[1]
            if dir_x * to_cx + dir_y * to_cy < 0:
                dir_x = -dir_x
                dir_y = -dir_y

            endpoint = clip_ray_to_box(mp, (dir_x, dir_y), box)

            ex = endpoint[0] - mp[0]
            ey = endpoint[1] - mp[1]
            actual_len = np.sqrt(ex * ex + ey * ey)
            if actual_len > EXTEND_LENGTH:
                endpoint = (int(round(mp[0] + dir_x * EXTEND_LENGTH)),
                            int(round(mp[1] + dir_y * EXTEND_LENGTH)))

            cv2.line(result, mp, endpoint, (0, 0, 255), 2)
```

## Files Modified

- `cube-detection-01.py` — `pipeline_detection()` function only

## Testing

- Run `_self_check_clip_ray()` (already exists, should still pass)
- Run the pipeline on sample photos and visually verify:
  - Endpoints of red segments get 1 extension
  - Independent intersection points get 2 extensions
  - Middle collinear points get no extensions
