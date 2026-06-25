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
LINE_EXTEND_FACTOR = 2.0     # Blue line extension multiplier
RECT_EXTEND_PX = 15          # Yellow bounding rectangle outward extension (pixels)
EXTEND_LENGTH = 50            # Red extension line length along blue line (pixels)

# Line merging thresholds
ANGLE_THRESHOLD = 4        # degrees
DIST_THRESHOLD = 11         # pixels

# ============ Drawing Element Storage ============
# Populated by stage functions, consumed by draw_all_elements.

def _reset_drawing_globals():
    """清除所有绘图元素的全局变量。每次调用 pipeline_detection 前执行，确保各帧数据互不干扰。"""
    global _raw_lines, _extended_lines, _merged_lines, \
        _intersection_points, _merged_points, _red_segments, \
        _extension_lines, _edges, _box, _image_size, \
        _excluded_points, _red_endpoints
    _raw_lines = []           # 原始 Hough 检测线段 [(p1, p2), ...]
    _extended_lines = []      # 延长后的线段
    _merged_lines = []        # 合并后的长线段（蓝色）
    _intersection_points = []  # 原始交点 / 过滤后交点（红点）
    _merged_points = []        # 聚类合并后的交点
    _red_segments = []         # 红色线段（角点之间的粗红边）
    _extension_lines = []      # 独立点延伸出的红色射线
    _edges = None              # Canny 边缘图
    _box = None                # 最小面积旋转包围盒（黄色矩形）
    _image_size = (0, 0)       # 图像宽高
    _excluded_points = set()   # 被排除的冗余共线交点
    _red_endpoints = set()     # 红色线段的端点集合

_reset_drawing_globals()


def create_output_dir():
    """创建输出目录（如果不存在）。确保结果和线条图片有地方写入。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LINES_DIR, exist_ok=True)


def lab_threshold(image):
    """
    对输入图像进行 LAB 色彩空间阈值分割。
    保留 L 通道在 [0,255]、A 通道在 [146,255]、B 通道在 [115,255] 范围内的像素，
    从而分离出立方体表面常见的红/黄色区域，生成二值掩码。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, LAB_LOWER, LAB_UPPER)
    return mask


def morphological_processing(mask):
    """
    对二值掩码进行形态学处理，去除噪声并填补空洞。
    - 开运算（先腐蚀后膨胀）：消除小的噪声点
    - 闭运算（先膨胀后腐蚀）：填补内部小孔洞
    - 轻微膨胀：增强边缘，便于后续直线检测
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


def _line_intersection_pair(p1, p2, p3, p4):
    """计算过 (p1,p2) 和 (p3,p4) 两条无限延长线的交点。若平行则返回 None。"""
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


def _segment_intersection_pair(p1, p2, p3, p4):
    """
    计算线段 (p1-p2) 与 (p3-p4) 的交点。
    仅当交点同时落在两条线段上时才返回 (x, y)，否则返回 None。
    """
    pt = _line_intersection_pair(p1, p2, p3, p4)
    if pt is None:
        return None
    x, y = pt
    if point_on_segment(x, y, p1[0], p1[1], p2[0], p2[1]) and \
       point_on_segment(x, y, p3[0], p3[1], p4[0], p4[1]):
        return pt
    return None


def find_all_intersections():
    """
    遍历所有合并后的线段，计算两两之间的交点。
    输入: _merged_lines（合并后的线段列表）
    输出: _intersection_points（所有交点坐标）
    """
    global _intersection_points
    _intersection_points = []
    if len(_merged_lines) < 2:
        return
    for i in range(len(_merged_lines)):
        for j in range(i + 1, len(_merged_lines)):
            pt = _segment_intersection_pair(
                _merged_lines[i][0], _merged_lines[i][1],
                _merged_lines[j][0], _merged_lines[j][1]
            )
            if pt is not None:
                _intersection_points.append(pt)


def merge_points():
    """
    将相互距离在 10 像素以内的交点聚类并合并为其质心。
    输入: _intersection_points（原始交点列表）
    输出: _merged_points（合并后的交点列表）
    """
    global _merged_points
    if not _intersection_points:
        return

    points = list(_intersection_points)
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
                    if abs(points[j][0] - cp[0]) <= 10 and \
                       abs(points[j][1] - cp[1]) <= 10:
                        cluster.append(points[j])
                        used[j] = True
                        changed = True
                        break
        # Compute centroid of cluster
        avg_x = int(round(sum(p[0] for p in cluster) / len(cluster)))
        avg_y = int(round(sum(p[1] for p in cluster) / len(cluster)))
        merged.append((avg_x, avg_y))

    _merged_points = merged


def assign_red_segments():
    """
    在每条合并后的线段上找到位于其上的合并交点，
    取距离最远的两个交点作为红色线段端点并存储。
    输入: _merged_lines, _merged_points, _box
    输出: _red_segments；同时产出 _excluded_points 和 _red_endpoints 供后续使用
    """
    global _red_segments
    if not _merged_lines or not _merged_points:
        return

    excluded_points = set()
    red_endpoints = set()

    for ext_line in _merged_lines:
        pt1, pt2 = ext_line
        points_on_line = []
        for mp in _merged_points:
            if point_on_segment(mp[0], mp[1], pt1[0], pt1[1], pt2[0], pt2[1], tol=3):
                points_on_line.append(mp)
        if len(points_on_line) >= 2:
            if len(points_on_line) == 2:
                p_a, p_b = points_on_line[0], points_on_line[1]
            else:
                max_dist = -1
                p_a, p_b = points_on_line[0], points_on_line[1]
                for pi in range(len(points_on_line)):
                    for pj in range(pi + 1, len(points_on_line)):
                        dx = points_on_line[pi][0] - points_on_line[pj][0]
                        dy = points_on_line[pi][1] - points_on_line[pj][1]
                        d = dx * dx + dy * dy
                        if d > max_dist:
                            max_dist = d
                            p_a, p_b = points_on_line[pi], points_on_line[pj]
            if len(points_on_line) >= 3:
                for p in points_on_line:
                    if p != p_a and p != p_b:
                        excluded_points.add(p)
            if _box is None or (
                cv2.pointPolygonTest(_box, p_a, False) >= 0 and
                cv2.pointPolygonTest(_box, p_b, False) >= 0
            ):
                _red_segments.append((p_a, p_b))
                red_endpoints.add(p_a)
                red_endpoints.add(p_b)

    # Store excluded points and endpoints for extend_independent_points
    globals()['_excluded_points'] = excluded_points
    globals()['_red_endpoints'] = red_endpoints


def filter_intersection_points():
    """
    过滤合并后的交点：移除被标记为排除的共线点以及位于包围盒外的点。
    输入: _merged_points, _excluded_points, _box
    输出: _intersection_points（过滤后重建）
    """
    global _intersection_points  # 使用全局的数组
    if not _merged_points:
        _intersection_points = []
        return

    excluded = globals().get('_excluded_points', set())
    filtered = []
    for mp in _merged_points:
        if mp in excluded:
            continue
        if _box is not None and cv2.pointPolygonTest(_box, mp, False) < 0:
            continue
        filtered.append(mp)
    _intersection_points = filtered


def extend_independent_points():
    """
    对独立交点（不在任何红色线段上的点），沿经过它的所有蓝色线段
    向包围盒中心方向延伸，生成红色延长线。
    输入: _merged_points, _merged_lines, _box
    输出: _extension_lines
    """
    global _extension_lines
    if _box is None or not _merged_points:
        return

    # Ensure excluded_points and red_endpoints exist (set by assign_red_segments)
    excluded_points = globals().get('_excluded_points', set())
    red_endpoints = globals().get('_red_endpoints', set())

    centroid = np.mean(_box, axis=0)

    for mp in _merged_points:
        if mp in excluded_points:
            continue
        if cv2.pointPolygonTest(_box, mp, False) < 0:
            continue

        is_endpoint = mp in red_endpoints
        red_on = []
        for seg in _red_segments:
            p_a, p_b = seg
            if point_on_segment(mp[0], mp[1], p_a[0], p_a[1],
                                 p_b[0], p_b[1], tol=3):
                red_on.append(seg)
        is_independent = len(red_on) == 0

        if not is_endpoint and not is_independent:
            continue
        if is_endpoint:
            continue

        blue_on = []
        for ml in _merged_lines:
            if point_on_segment(mp[0], mp[1], ml[0][0], ml[0][1],
                                 ml[1][0], ml[1][1], tol=3):
                blue_on.append(ml)

        for bl in blue_on:
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

            endpoint = clip_ray_to_box(mp, (dir_x, dir_y), _box)

            ex = endpoint[0] - mp[0]
            ey = endpoint[1] - mp[1]
            actual_len = np.sqrt(ex * ex + ey * ey)
            if actual_len > EXTEND_LENGTH:
                endpoint = (int(round(mp[0] + dir_x * EXTEND_LENGTH)),
                            int(round(mp[1] + dir_y * EXTEND_LENGTH)))

            _extension_lines.append((mp, endpoint))


def point_on_segment(px, py, x1, y1, x2, y2, tol=2):
    """
    判断点 (px, py) 是否位于线段 (x1,y1)-(x2,y2) 上。
    使用带容差的边界框检查（应对整数舍入），
    并叠加垂直距离共线检查，排除在边界框内但偏离实际直线的点。
    """
    # Point must be within bounding box of segment (with tolerance for rounding)
    if not (min(x1, x2) - tol <= px <= max(x1, x2) + tol and
            min(y1, y2) - tol <= py <= max(y1, y2) + tol):
        return False
    # Collinearity check: perpendicular distance must be small
    dist = point_to_line_distance(px, py, x1, y1, x2, y2)
    return dist <= 5.0




def point_to_line_distance(px, py, x1, y1, x2, y2):
    """计算点 (px, py) 到直线 (x1,y1)-(x2,y2) 的垂直距离。"""
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-10:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length


def extend_lines():
    """
    将每条原始 Hough 检测线段以中点为中心、按 LINE_EXTEND_FACTOR 倍率向两端延长。
    输入: _raw_lines（原始 Hough 线段）
    输出: _extended_lines（延长后的线段）
    """
    global _extended_lines
    if not _raw_lines:
        return
    for p1, p2 in _raw_lines:
        x1, y1 = p1
        x2, y2 = p2
        ext1 = (int(round(LINE_EXTEND_FACTOR * x1 - (LINE_EXTEND_FACTOR - 1) * x2)),
                int(round(LINE_EXTEND_FACTOR * y1 - (LINE_EXTEND_FACTOR - 1) * y2)))
        ext2 = (int(round(LINE_EXTEND_FACTOR * x2 - (LINE_EXTEND_FACTOR - 1) * x1)),
                int(round(LINE_EXTEND_FACTOR * y2 - (LINE_EXTEND_FACTOR - 1) * y1)))
        _extended_lines.append((ext1, ext2))


def clip_ray_to_box(origin, direction, box):
    """
    将一条射线限制在凸多边形（黄色包围盒）内部。
    从 origin 出发、沿 direction 方向的射线与多边形边界的交点即为终点。

    参数:
        origin: (x, y) 射线起点（已在多边形内）
        direction: (dx, dy) 单位方向向量
        box: np.intp 4x2 顶点数组

    返回:
        (x, y) 射线与多边形边界的交点；若未找到交点则回退为 origin + direction * EXTEND_LENGTH。
    """
    ox, oy = origin
    dx, dy = direction

    best_t = float('inf')
    best_pt = None

    for i in range(4):
        # Edge from box[i] to box[(i+1)%4]
        ex1, ey1 = int(box[i][0]), int(box[i][1])
        ex2, ey2 = int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1])

        # Ray: P = origin + t * direction, t >= 0
        # Edge: Q = edge_start + s * edge_dir, s in [0, 1]
        edge_dx = ex2 - ex1
        edge_dy = ey2 - ey1

        denom = dx * edge_dy - dy * edge_dx
        if abs(denom) < 1e-10:
            continue  # parallel

        t = ((ex1 - ox) * edge_dy - (ey1 - oy) * edge_dx) / denom
        s = ((ex1 - ox) * dy - (ey1 - oy) * dx) / denom

        if t > 1e-6 and 0 <= s <= 1 and t < best_t:
            best_t = t
            best_pt = (int(round(ox + t * dx)), int(round(oy + t * dy)))

    if best_pt is not None:
        return best_pt

    # Fallback: should not be reached for a convex box with interior origin.
    # If triggered (numerical edge case), the endpoint may lie outside the box.
    return (int(round(ox + dx * EXTEND_LENGTH)),
            int(round(oy + dy * EXTEND_LENGTH)))


def merge_lines():
    """
    将接近平行的延长线合并为单一长线段。
    先按极坐标角度聚类，再在同类中按 rho 距离二次聚类，
    最终将每簇内所有端点投影到中位方向并取极值点作为合并结果。
    输入: _extended_lines
    输出: _merged_lines
    """
    global _merged_lines
    if not _extended_lines:
        return

    # --- Convert to polar representation (theta, rho) ---
    lines_polar = []
    for idx, (p1, p2) in enumerate(_extended_lines):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            continue
        theta = np.arctan2(dy, dx) % np.pi
        rho = abs(x1 * y2 - x2 * y1) / length
        lines_polar.append((theta, rho, idx, p1, p2))

    if not lines_polar:
        return

    # --- Sort by angle and cluster ---
    angle_thresh_rad = np.radians(ANGLE_THRESHOLD)
    lines_polar.sort(key=lambda x: x[0])

    # Handle angle wrap-around: duplicate first entries shifted by +pi
    extended = [(theta + np.pi, rho, idx, p1, p2)
                for theta, rho, idx, p1, p2 in lines_polar]
    all_angles = lines_polar + extended

    # Cluster by angle, ensuring each original index appears in at most one cluster
    angle_clusters = []
    current_cluster = [all_angles[0]]
    used_indices = {all_angles[0][2]}
    for i in range(1, len(all_angles)):
        idx = all_angles[i][2]
        if idx in used_indices:
            continue
        if all_angles[i][0] - current_cluster[-1][0] < angle_thresh_rad:
            current_cluster.append(all_angles[i])
            used_indices.add(idx)
        else:
            angle_clusters.append(current_cluster)
            current_cluster = [all_angles[i]]
            used_indices.add(idx)
    angle_clusters.append(current_cluster)

    # --- Within each angle cluster, sub-cluster by rho ---
    for cluster in angle_clusters:
        # Deduplicate by original index (from the wrap-around duplication)
        seen = set()
        unique_lines = []
        for theta, rho, idx, p1, p2 in cluster:
            if idx not in seen:
                seen.add(idx)
                unique_lines.append((theta, rho, idx, p1, p2))

        if not unique_lines:
            continue

        # Sort by rho for secondary clustering
        unique_lines.sort(key=lambda x: x[1])

        rho_clusters = []
        current_rho_cluster = [unique_lines[0]]
        for i in range(1, len(unique_lines)):
            if unique_lines[i][1] - current_rho_cluster[-1][1] < DIST_THRESHOLD:
                current_rho_cluster.append(unique_lines[i])
            else:
                rho_clusters.append(current_rho_cluster)
                current_rho_cluster = [unique_lines[i]]
        rho_clusters.append(current_rho_cluster)

        # --- Merge each rho cluster into one line ---
        for rho_cluster in rho_clusters:
            # Collect all endpoints
            all_points = []
            for theta, rho, idx, p1, p2 in rho_cluster:
                all_points.append(p1)
                all_points.append(p2)

            # Use the cluster's median angle as the merge direction
            median_theta = np.median([item[0] for item in rho_cluster])
            direction = np.array([np.cos(median_theta), np.sin(median_theta)])

            # Project all points onto the direction vector
            projections = [p[0] * direction[0] + p[1] * direction[1] for p in all_points]
            min_idx = int(np.argmin(projections))
            max_idx = int(np.argmax(projections))

            _merged_lines.append((all_points[min_idx], all_points[max_idx]))


def _is_independent_blue(blue_line):
    pt1, pt2 = blue_line
    for rp1, rp2 in _red_segments:
        if (point_on_segment(rp1[0], rp1[1], pt1[0], pt1[1], pt2[0], pt2[1], tol=3) and
            point_on_segment(rp2[0], rp2[1], pt1[0], pt1[1], pt2[0], pt2[1], tol=3)):
            return False
    for ix, iy in _merged_points:
        if point_on_segment(ix, iy, pt1[0], pt1[1], pt2[0], pt2[1], tol=3):
            return False
    return True


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
    _reset_drawing_globals()

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
            _raw_lines.append(((x1, y1), (x2, y2)))

    # Extend raw lines (global I/O)
    extend_lines()

    # Merge extended lines (global I/O)
    merge_lines()
    print(f"    Lines after merging: {len(_merged_lines)} (from {line_count})")

    # Collect all intersection points from merged lines (global I/O)
    find_all_intersections()
    merge_points()

    # Stage 6: Assign red segments (also computes _excluded_points)
    assign_red_segments()

    # Stage 6b: Filter intersection points (remove excluded + out-of-box)
    filter_intersection_points()

    raw_count = len(_merged_lines)
    _merged_lines[:] = [bl for bl in _merged_lines if not _is_independent_blue(bl)]
    if len(_merged_lines) < raw_count:
        print(f"    Removed {raw_count - len(_merged_lines)} independent blue lines")

    # Stage 7: Extend independent points
    extend_independent_points()

    return edges, line_count, len(_merged_lines), len(_merged_points)


def draw_all_elements(image, box=None):
    """
    将检测到的所有元素绘制到图像上（从全局变量读取）。
    绘制顺序: 绿色原始线 → 蓝色合并线 → 红色线段 → 红色交点 → 红色延长线。
    """
    # Green: raw Hough lines
    for pt1, pt2 in _raw_lines:
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # Blue: merged extended lines
    for pt1, pt2 in _merged_lines:
        cv2.line(image, pt1, pt2, (255, 0, 0), 1)

    # Red segments: between intersection points on merged lines
    for pt1, pt2 in _red_segments:
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)

    # Red dots: intersection points
    for x, y in _intersection_points:
        if box is not None and cv2.pointPolygonTest(box, (x, y), False) < 0:
            continue
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(image, (x, y), 5, (0, 0, 255), 2)

    # Red extension lines
    for pt1, pt2 in _extension_lines:
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)


def find_contours(image, morphed_mask):
    """
    在二值掩码上查找轮廓，筛选出可能是立方体的候选区域。
    过滤条件：面积 ≥ 500，多边形逼近后顶点数在 4–8 之间（立方体 2D 投影的典型范围）。
    返回: 绘制了轮廓的图像、候选数量、轮廓列表
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

    return result, len(cube_candidates), cube_candidates


def _compute_min_area_box(contours, extend_px=0):
    """
    计算最大轮廓的最小面积旋转包围矩形。

    参数:
        contours: cv2.findContours 返回的轮廓列表
        extend_px: 每条边向外扩展的像素数

    返回:
        np.intp 4x2 顶点数组；若无轮廓则返回 None
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


def draw_min_area_rects(image, contours, color=(0, 255, 255), thickness=2, extend_px=0):
    """
    在图像上绘制最大轮廓的最小面积旋转包围矩形。

    参数:
        image: BGR 图像（原地修改）
        contours: cv2.findContours 返回的轮廓列表
        color: BGR 颜色元组，默认黄色 (0, 255, 255)
        thickness: 线宽（像素）
        extend_px: 每条边向外扩展的像素数

    返回:
        绘制了矩形的图像（与输入同一引用）
    """
    box = _compute_min_area_box(contours, extend_px)
    if box is None:
        return image

    cv2.drawContours(image, [box], 0, color, thickness)
    return image


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
    _reset_drawing_globals()
    # Two raw lines
    _raw_lines.append(((10, 10), (50, 10)))
    _raw_lines.append(((10, 20), (50, 20)))
    _image_size = (100, 100)
    extend_lines()
    assert len(_extended_lines) == 2, f"Expected 2, got {len(_extended_lines)}"
    print("  [self_check] global_flow (extend_lines): PASS")


def _self_check_clip_ray():
    """快速自测：验证 clip_ray_to_box 在正方形包围盒上的交点计算正确。"""
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
    _reset_drawing_globals()
    # Two perpendicular lines crossing at (50, 50)
    _merged_lines.append(((10, 50), (90, 50)))  # 水平线
    _merged_lines.append(((50, 10), (50, 90)))  # 垂直线
    find_all_intersections()
    assert len(_intersection_points) >= 1, f"Expected >= 1, got {len(_intersection_points)}"
    pt = _intersection_points[0]
    assert 48 <= pt[0] <= 52 and 48 <= pt[1] <= 52, f"Expected ~(50,50), got {pt}"
    print("  [self_check] find_all_intersections: PASS")


if __name__ == "__main__":
    _self_check_intersections()
    _self_check_clip_ray()
    main()
