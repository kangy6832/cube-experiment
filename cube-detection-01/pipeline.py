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

from config import (PHOTOS_DIR, OUTPUT_DIR, LINES_DIR,
                     HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                     HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP,
                     LAB_LOWER, LAB_UPPER, RECT_EXTEND_PX, ANGLE_THRESHOLD)
import state
from color import lab_threshold, morphological_processing
from contours import find_contours, _compute_min_area_box, draw_min_area_rects
from lines import (extend_lines, merge_lines, find_all_intersections,
                    merge_points, assign_red_segments,
                    filter_intersection_points, extend_independent_points,
                    _is_independent_blue)


def create_output_dir():
    """创建输出目录（如果不存在）。确保结果和线条图片有地方写入。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LINES_DIR, exist_ok=True)


def pipeline_detection(image, morphed_mask, box=None):
    """
    完整直线检测流水线：Canny 边缘检测 → 概率 Hough 变换 → 线段延长 →
    线段合并 → 交点计算与合并 → 红色线段分配 → 独立点延伸。
    所有中间结果存入全局变量供 draw_all_elements 使用。

    参数:
        image: BGR 原始图像
        morphed_mask: 形态学处理后的二值掩码
        box: np.intp 4x2 多边形顶点数组，或 None 表示不限制绘制范围

    返回: edges, line_count, merged_count, intersection_count
    """
    state._reset_drawing_globals()

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

    # Collect raw Hough lines into global
    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            state._raw_lines.append(((x1, y1), (x2, y2)))

    # Extend raw lines (global I/O)
    extend_lines()

    # Merge extended lines (global I/O)
    merge_lines()
    print(f"    Lines after merging: {len(state._merged_lines)} (from {line_count})")

    # Collect all intersection points from merged lines (global I/O)
    find_all_intersections()
    merge_points()

    # Stage 6: Assign red segments (also computes _excluded_points)
    assign_red_segments()

    # Stage 6b: Filter intersection points (remove excluded + out-of-box)
    filter_intersection_points()

    raw_count = len(state._merged_lines)
    state._merged_lines[:] = [bl for bl in state._merged_lines if not _is_independent_blue(bl)]
    if len(state._merged_lines) < raw_count:
        print(f"    Removed {raw_count - len(state._merged_lines)} independent blue lines")

    # Stage 7: Extend independent points
    extend_independent_points()

    return edges, line_count, len(state._merged_lines), len(state._merged_points)


def draw_all_elements(image, box=None):
    """
    将检测到的所有元素绘制到图像上（从全局变量读取）。
    绘制顺序: 绿色原始线 → 蓝色合并线 → 红色线段 → 红色交点 → 红色延长线。
    """
    # Green: raw Hough lines
    for pt1, pt2 in state._raw_lines:
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # Blue: merged extended lines
    for pt1, pt2 in state._merged_lines:
        cv2.line(image, pt1, pt2, (255, 0, 0), 1)

    # Red segments: between intersection points on merged lines
    for pt1, pt2 in state._red_segments:
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)

    # Red dots: intersection points
    for x, y in state._intersection_points:
        if box is not None and cv2.pointPolygonTest(box, (x, y), False) < 0:
            continue
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(image, (x, y), 5, (0, 0, 255), 2)

    # Red extension lines
    for pt1, pt2 in state._extension_lines:
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)


def process_image(image_path, idx):
    """对单张图像执行完整流水线：阈值分割 → 形态学处理 → 轮廓检测 → 直线检测 → 结果保存。"""
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

    # Step 3: Contour detection for cube candidates
    result_contours, candidate_count, contours_list = find_contours(image, morphed)
    print(f"  Cube candidates: {candidate_count}")

    # Step 4: Compute bounding box before drawing red points
    box = _compute_min_area_box(contours_list, extend_px=RECT_EXTEND_PX)

    # Step 5: Pipeline (Hough Line) detection — populate globals
    edges, line_count, merged_count, intersection_count = pipeline_detection(
        image, morphed, box=box)
    print(f"  Hough Lines detected: {line_count}")
    print(f"  Lines after merging: {merged_count}")
    print(f"  Intersection points (merged): {intersection_count}")

    # Step 5b: Draw all elements from globals onto a clean copy
    result_lines = image.copy()
    draw_all_elements(result_lines, box)

    # Step 6: Draw yellow minimum-area bounding boxes on result_lines
    draw_min_area_rects(result_lines, contours_list, extend_px=RECT_EXTEND_PX)

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_01_mask.png"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_02_morphed.png"), morphed)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_03_edges.png"), edges)
    cv2.imwrite(os.path.join(LINES_DIR, f"{base_name}_04_lines.png"), result_lines)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_05_contours.png"), result_contours)

    # Create a composite visualization
    composite = create_composite(image, mask, morphed, edges, result_lines, result_contours)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_06_composite.png"), composite)

    print(f"  Output saved to: {OUTPUT_DIR}")


def create_composite(image, mask, morphed, edges, lines_img, contours_img):
    """将流水线各阶段的输出拼成一张 2×3 的合成图，供可视化检查。"""
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
    """主入口：扫描照片目录，逐张处理并输出结果。"""
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


def _self_check_global_flow():
    """快速自测：验证 _raw_lines 经 extend_lines 后正确产出 _extended_lines。"""
    state._reset_drawing_globals()
    # Two raw lines
    state._raw_lines.append(((10, 10), (50, 10)))
    state._raw_lines.append(((10, 20), (50, 20)))
    state._image_size = (100, 100)
    extend_lines()
    assert len(state._extended_lines) == 2, f"Expected 2, got {len(state._extended_lines)}"
    print("  [self_check] global_flow (extend_lines): PASS")


def _self_check_clip_ray():
    """快速自测：验证 clip_ray_to_box 在正方形包围盒上的交点计算正确。"""
    from geometry import clip_ray_to_box
    # Square box: (10,10), (100,10), (100,100), (10,100)
    box = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.intp)

    # Ray from center going right — should hit right edge
    pt = clip_ray_to_box((50, 50), (1.0, 0.0), box)
    assert pt[0] == 100 and pt[1] == 50, f"Expected (100, 50), got {pt}"

    # Ray from center going up — should hit top edge
    pt = clip_ray_to_box((50, 50), (0.0, -1.0), box)
    assert pt[0] == 50 and pt[1] == 10, f"Expected (50, 10), got {pt}"

    # Ray from center going down-right diagonal
    diag = 1.0 / np.sqrt(2)
    pt = clip_ray_to_box((50, 50), (diag, diag), box)
    assert 99 <= pt[0] <= 101 and 99 <= pt[1] <= 101, f"Expected ~(100,100), got {pt}"

    print("  [self_check] clip_ray_to_box: PASS")


def _self_check_intersections():
    """快速自测：验证两条垂直线段的交点计算结果接近 (50, 50)。"""
    state._reset_drawing_globals()
    # Two perpendicular lines crossing at (50, 50)
    state._merged_lines.append(((10, 50), (90, 50)))  # 水平线
    state._merged_lines.append(((50, 10), (50, 90)))  # 垂直线
    find_all_intersections()
    assert len(state._intersection_points) >= 1, f"Expected >= 1, got {len(state._intersection_points)}"
    pt = state._intersection_points[0]
    assert 48 <= pt[0] <= 52 and 48 <= pt[1] <= 52, f"Expected ~(50,50), got {pt}"
    print("  [self_check] find_all_intersections: PASS")


if __name__ == "__main__":
    _self_check_intersections()
    _self_check_clip_ray()
    main()
