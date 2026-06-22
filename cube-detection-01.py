#!/usr/bin/env python3
"""
Cube Detection Script
- LAB color space thresholding: L(0,255), A(146,255), B(115,255)
- Morphological processing on binarized image
- Pipeline (Hough Line) detection for cube edges
"""

import cv2
import numpy as np
import os
import glob

# ============ Configuration ============
PHOTOS_DIR = "/home/kangy/MyProjects/cube-experiment/photos"
OUTPUT_DIR = "/home/kangy/MyProjects/cube-experiment/output/01"
LINES_DIR = "/home/kangy/MyProjects/cube-experiment/output/lines"
LINES_RAW_DIR = "/home/kangy/MyProjects/cube-experiment/output/lines_raw"

# LAB thresholds: L(0,255), A(146,255), B(115,255)
LAB_LOWER = np.array([0, 146, 115])
LAB_UPPER = np.array([255, 255, 255])

# Morphological kernel sizes
MORPH_KERNEL_SIZE = 5
MORPH_ITERATIONS = 2

# Hough Line Transform parameters
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10


def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LINES_DIR, exist_ok=True)
    os.makedirs(LINES_RAW_DIR, exist_ok=True)


def lab_threshold(image):
    """
    Apply LAB color space thresholding.
    Keep pixels where L in [0,255], A in [146,255], B in [115,255].
    This isolates reddish/yellowish regions typical of cubes.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, LAB_LOWER, LAB_UPPER)
    return mask


def morphological_processing(mask):
    """
    Apply morphological operations to clean up the binary image.
    - Opening (erosion then dilation) to remove noise
    - Closing (dilation then erosion) to fill gaps
    - Dilation to enhance edges for line detection
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    # Opening: remove small noise
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)

    # Closing: fill small gaps
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)

    # Slight dilation to enhance edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(closing, kernel_dilate, iterations=1)

    return dilated


def line_intersection(p1, p2, p3, p4):
    """
    Find intersection point of line through (p1,p2) and line through (p3,p4).
    Returns (x, y) or None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (int(round(x)), int(round(y)))


def merge_intersection_points(points, merge_radius=10):
    """
    Merge intersection points that are within merge_radius of each other.
    Returns list of merged points (centroids of each cluster).
    """
    if not points:
        return []

    points = list(points)
    merged = []
    used = [False] * len(points)

    for i in range(len(points)):
        if used[i]:
            continue
        # Start a new cluster with point i
        cluster = [points[i]]
        used[i] = True
        # Find all points close to any point in the cluster
        changed = True
        while changed:
            changed = False
            for j in range(len(points)):
                if used[j]:
                    continue
                # Check if point j is close to any point in current cluster
                for cp in cluster:
                    if abs(points[j][0] - cp[0]) <= merge_radius and \
                       abs(points[j][1] - cp[1]) <= merge_radius:
                        cluster.append(points[j])
                        used[j] = True
                        changed = True
                        break
        # Compute centroid of cluster
        avg_x = int(round(sum(p[0] for p in cluster) / len(cluster)))
        avg_y = int(round(sum(p[1] for p in cluster) / len(cluster)))
        merged.append((avg_x, avg_y))

    return merged


def point_on_segment(px, py, x1, y1, x2, y2, tol=2):
    """
    Check if point (px, py) lies on line segment from (x1,y1) to (x2,y2).
    Uses bounding box check with tolerance for integer rounding,
    PLUS a perpendicular-distance collinearity check to reject points
    that are inside the bounding box but far from the actual line.
    """
    # Point must be within bounding box of segment (with tolerance for rounding)
    if not (min(x1, x2) - tol <= px <= max(x1, x2) + tol and
            min(y1, y2) - tol <= py <= max(y1, y2) + tol):
        return False
    # Collinearity check: perpendicular distance must be small
    dist = point_to_line_distance(px, py, x1, y1, x2, y2)
    return dist <= 5.0


def segment_intersection(p1, p2, p3, p4):
    """
    Find intersection point of line SEGMENTS (p1-p2) and (p3-p4).
    Returns (x, y) only if intersection lies on BOTH segments, else None.
    """
    pt = line_intersection(p1, p2, p3, p4)
    if pt is None:
        return None
    x, y = pt
    if point_on_segment(x, y, p1[0], p1[1], p2[0], p2[1]) and \
       point_on_segment(x, y, p3[0], p3[1], p4[0], p4[1]):
        return pt
    return None


def extend_line(p1, p2, img_w, img_h):
    """
    Extend a line segment to 2x its original length (centered on original).
    Returns two extended points, or None if out of bounds.
    """
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        return None

    # Direction vector
    dx = x2 - x1
    dy = y2 - y1

    # Original segment: t in [0, 1], extend to t in [-0.5, 1.5] for 2x length
    t_min = -0.5
    t_max = 1.5

    ext1 = (int(round(x1 + t_min * dx)), int(round(y1 + t_min * dy)))
    ext2 = (int(round(x1 + t_max * dx)), int(round(y1 + t_max * dy)))

    # Check that at least part of the extended line is within the image
    margin = 100
    if (min(ext1[0], ext2[0]) > img_w + margin or
        max(ext1[0], ext2[0]) < -margin or
        min(ext1[1], ext2[1]) > img_h + margin or
        max(ext1[1], ext2[1]) < -margin):
        return None

    return (ext1, ext2)


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to line through (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-10:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length


def extend_line_2x(p1, p2):
    """
    Extend a line segment to 2x its original length, centered at its midpoint.
    Returns the two new endpoints.
    """
    x1, y1 = p1
    x2, y2 = p2
    # Extend each endpoint outward by 1x its distance from center
    # New length = 2x original, centered at midpoint
    ext1 = (int(round(2 * x1 - x2)), int(round(2 * y1 - y2)))
    ext2 = (int(round(2 * x2 - x1)), int(round(2 * y2 - y1)))
    return ext1, ext2


def pipeline_detection(image, morphed_mask):
    """
    Detect lines using Probabilistic Hough Line Transform.
    Extend ALL detected Hough lines to image boundaries and mark
    their intersections with red dots.

    Returns: result image, raw_lines image, edges, line_count, raw_lines
    """
    h, w = image.shape[:2]

    # Detect edges using Canny on the morphed mask
    edges = cv2.Canny(morphed_mask, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    # Draw all original green lines on a copy of the original image
    result = image.copy()
    line_count = 0

    raw_lines_img = None
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        raw_lines_img = result.copy()

    extended_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ext = extend_line_2x((x1, y1), (x2, y2))
            extended_lines.append(ext)
            cv2.line(result, ext[0], ext[1], (255, 0, 0), 1)  # Blue for extended lines

    # Collect all intersection points
    raw_points = []
    for i in range(len(extended_lines)):
        for j in range(i + 1, len(extended_lines)):
            pt = segment_intersection(
                extended_lines[i][0], extended_lines[i][1],
                extended_lines[j][0], extended_lines[j][1]
            )
            if pt is not None:
                raw_points.append(pt)

    # Merge nearby points (within merge_radius pixels)
    merged_points = merge_intersection_points(raw_points, merge_radius=10)

    # For each extended line, find merged intersection points that lie on it
    # and draw the segment between the first two such points in red.
    for ext_line in extended_lines:
        pt1, pt2 = ext_line
        points_on_line = []
        for mp in merged_points:
            if point_on_segment(mp[0], mp[1], pt1[0], pt1[1], pt2[0], pt2[1], tol=3):
                points_on_line.append(mp)
        if len(points_on_line) >= 2:
            # Draw the segment between the first two intersection points in red
            cv2.line(result, points_on_line[0], points_on_line[1], (0, 0, 255), 2)

    # Draw merged intersection points
    for x, y in merged_points:
        cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(result, (x, y), 5, (0, 0, 255), 2)

    return result, raw_lines_img, edges, line_count, lines, len(merged_points)


def find_contours(image, morphed_mask):
    """
    Find and draw contours that could be cube candidates.
    Filters by area and aspect ratio.
    """
    result = image.copy()
    contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cube_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Filter small contours
            continue

        # Approximate polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Cubes typically have 4-6 vertices in 2D projection
        if 4 <= len(approx) <= 8:
            cube_candidates.append(contour)

    return result, len(cube_candidates)


def process_image(image_path, idx):
    """Process a single image through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not read image {image_path}")
        return

    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}")

    # Step 1: LAB thresholding
    mask = lab_threshold(image)
    white_pixels = cv2.countNonZero(mask)
    ratio = white_pixels / (w * h) * 100
    print(f"  LAB Threshold - White pixels: {white_pixels} ({ratio:.1f}%)")

    # Step 2: Morphological processing
    morphed = morphological_processing(mask)
    morphed_white = cv2.countNonZero(morphed)
    morphed_ratio = morphed_white / (w * h) * 100
    print(f"  Morphological - White pixels: {morphed_white} ({morphed_ratio:.1f}%)")

    # Step 3: Pipeline (Hough Line) detection with border extension
    # This produces the lines image with:
    #   - All detected green lines (thin)
    #   - Outer border lines extended to image borders (thick green)
    #   - Intersection points marked with red dots
    result_lines, raw_lines_img, edges, line_count, raw_lines, intersection_count = pipeline_detection(image, morphed)
    print(f"  Hough Lines detected: {line_count}")
    print(f"  Intersection points (merged): {intersection_count}")

    # Step 4: Contour detection for cube candidates
    result_contours, candidate_count = find_contours(image, morphed)
    print(f"  Cube candidates: {candidate_count}")

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_01_mask.png"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_02_morphed.png"), morphed)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_03_edges.png"), edges)
    cv2.imwrite(os.path.join(LINES_DIR, f"{base_name}_04_lines.png"), result_lines)
    if raw_lines_img is not None:
        cv2.imwrite(os.path.join(LINES_RAW_DIR, f"{base_name}_04_lines_raw.png"), raw_lines_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_05_contours.png"), result_contours)

    # Create a composite visualization
    composite = create_composite(image, mask, morphed, edges, result_lines, result_contours)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_06_composite.png"), composite)

    print(f"  Output saved to: {OUTPUT_DIR}")


def create_composite(image, mask, morphed, edges, lines_img, contours_img):
    """Create a composite image showing all pipeline stages."""
    h, w = image.shape[:2]

    # Resize for display if too large
    max_display = 800
    scale = min(max_display / w, max_display / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(image, (new_w, new_h))
        m = cv2.resize(mask, (new_w, new_h))
        mor = cv2.resize(morphed, (new_w, new_h))
        e = cv2.resize(edges, (new_w, new_h))
        ln = cv2.resize(lines_img, (new_w, new_h))
        ct = cv2.resize(contours_img, (new_w, new_h))
    else:
        img = image
        m = mask
        mor = morphed
        e = edges
        ln = lines_img
        ct = contours_img

    # Convert grayscale to 3-channel for stacking
    m_color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    mor_color = cv2.cvtColor(mor, cv2.COLOR_GRAY2BGR)
    e_color = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 255)
    thickness = 2

    cv2.putText(img, "Original", (10, 25), font, font_scale, color, thickness)
    cv2.putText(m_color, "LAB Mask", (10, 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(mor_color, "Morphed", (10, 25), font, font_scale, color, thickness)
    cv2.putText(e_color, "Edges", (10, 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(ln, "Hough Lines", (10, 25), font, font_scale, color, thickness)
    cv2.putText(ct, "Contours", (10, 25), font, font_scale, color, thickness)

    # Stack: top row (original, mask, morphed), bottom row (edges, lines, contours)
    top_row = np.hstack([img, m_color, mor_color])
    bottom_row = np.hstack([e_color, ln, ct])
    composite = np.vstack([top_row, bottom_row])

    return composite


def main():
    """Main entry point."""
    print("=" * 60)
    print("Cube Detection Pipeline")
    print(f"LAB Threshold: L({LAB_LOWER[0]}-{LAB_UPPER[0]}), "
          f"A({LAB_LOWER[1]}-{LAB_UPPER[1]}), "
          f"B({LAB_LOWER[2]}-{LAB_UPPER[2]})")
    print(f"Photos directory: {PHOTOS_DIR}")
    print("=" * 60)

    create_output_dir()

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(PHOTOS_DIR, ext)))
    image_paths.sort()

    if not image_paths:
        print("No images found in photos directory!")
        return

    print(f"\nFound {len(image_paths)} images to process.")

    # Process each image
    for idx, image_path in enumerate(image_paths):
        process_image(image_path, idx)

    print(f"\n{'='*60}")
    print(f"Processing complete. {len(image_paths)} images processed.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
