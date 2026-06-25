# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cube detection computer vision pipeline using OpenCV. Detects red/yellow cube edges in photos via LAB color thresholding, morphological processing, contour detection, and Hough line transform with custom line merging/intersection logic.

## Commands

```bash
# Activate virtual environment
source myvenv/bin/activate

# Run main detection pipeline (processes all images in photos/)
python3 cube-detection-01.py

# Run LAB threshold tuner (interactive slider GUI)
python3 adjust-lab.py

# Run 3D cube viewer (OpenCV window with trackbars)
python3 draw-cube.py

# Capture photos from camera to photos/
python3 take-photos.py

# Syntax check
python3 -c "import py_compile; py_compile.compile('cube-detection-01.py', doraise=True)"
```

## Architecture

### Pipeline Flow (cube-detection-01.py)

1. **LAB Thresholding** (`lab_threshold`) — isolates red/yellow regions in LAB color space
2. **Morphological Processing** (`morphological_processing`) — open/close/dilate to clean mask
3. **Contour Detection** (`find_contours`) — filters by area (≥500px) and vertex count (4–8)
4. **Min-Area Bounding Box** (`_compute_min_area_box`) — rotated rect around largest contour, extended outward
5. **Hough Line Detection** — probabilistic Hough on Canny edges
6. **Line Extension** (`extend_lines`) — extends raw Hough lines 2× from midpoint
7. **Line Merging** (`merge_lines`) — polar-coordinate clustering (angle then rho distance)
8. **Intersection Finding** (`find_all_intersections` + `merge_points`) — pairwise segment intersections with 10px clustering
9. **Red Segment Assignment** (`assign_red_segments`) — connects intersection points along merged lines
10. **Independent Point Extension** (`extend_independent_points`) — extends rays from orphan intersections toward box center

### Global Variable Pattern

Intermediate results are stored in module-level globals (prefixed with `_`) and reset via `_reset_drawing_globals()` at the start of each `pipeline_detection` call. Drawing functions read from these globals rather than receiving parameters. This is the established pattern — follow it when adding new detection stages.

Key globals: `_raw_lines`, `_extended_lines`, `_merged_lines`, `_intersection_points`, `_merged_points`, `_red_segments`, `_extension_lines`, `_box`.

### Configuration

All tunable parameters are defined as module-level constants at the top of `cube-detection-01.py` (lines 14–40). Modify these rather than hardcoding values in functions.

## Dependencies

- Python 3.12, OpenCV 4.13, NumPy 2.4, Matplotlib 3.11
- Installed in `myvenv/` (no requirements.txt)
- ROS2 Jazzy Python packages are in the VSCode autocomplete path but not required for this project

## Output Structure

- `output/01/` — per-image results (mask, morphed, edges, contours, composite)
- `output/lines/` — images with detected lines drawn
- `output/cube/` — saved frames from draw-cube.py
- `photos/` — input images (jpg/png)
