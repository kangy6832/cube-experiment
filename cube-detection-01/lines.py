import cv2
import numpy as np

from config import (LINE_EXTEND_FACTOR, ANGLE_THRESHOLD, DIST_THRESHOLD,
                     EXTEND_LENGTH)
import state
from geometry import (_line_intersection_pair, _segment_intersection_pair,
                       point_on_segment, clip_ray_to_box)


def find_all_intersections():
    """
    遍历所有合并后的线段，计算两两之间的交点。
    输入: _merged_lines（合并后的线段列表）
    输出: _intersection_points（所有交点坐标）
    """
    state._intersection_points = []
    if len(state._merged_lines) < 2:
        return
    for i in range(len(state._merged_lines)):
        for j in range(i + 1, len(state._merged_lines)):
            pt = _segment_intersection_pair(
                state._merged_lines[i][0], state._merged_lines[i][1],
                state._merged_lines[j][0], state._merged_lines[j][1]
            )
            if pt is not None:
                state._intersection_points.append(pt)


def merge_points():
    """
    将相互距离在 10 像素以内的交点聚类并合并为其质心。
    输入: _intersection_points（原始交点列表）
    输出: _merged_points（合并后的交点列表）
    """
    if not state._intersection_points:
        return

    points = list(state._intersection_points)
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

    state._merged_points = merged


def assign_red_segments():
    """
    在每条合并后的线段上找到位于其上的合并交点，
    取距离最远的两个交点作为红色线段端点并存储。
    输入: _merged_lines, _merged_points, _box
    输出: _red_segments；同时产出 _excluded_points 和 _red_endpoints 供后续使用
    """
    state._red_segments = []
    if not state._merged_lines or not state._merged_points:
        return

    excluded_points = set()
    red_endpoints = set()

    for ext_line in state._merged_lines:
        pt1, pt2 = ext_line
        points_on_line = []
        for mp in state._merged_points:
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
            if state._box is None or (
                cv2.pointPolygonTest(state._box, p_a, False) >= 0 and
                cv2.pointPolygonTest(state._box, p_b, False) >= 0
            ):
                state._red_segments.append((p_a, p_b))
                red_endpoints.add(p_a)
                red_endpoints.add(p_b)

    # Store excluded points and endpoints for extend_independent_points
    state._excluded_points = excluded_points
    state._red_endpoints = red_endpoints


def filter_intersection_points():
    """
    过滤合并后的交点：移除被标记为排除的共线点以及位于包围盒外的点。
    输入: _merged_points, _excluded_points, _box
    输出: _intersection_points（过滤后重建）
    """
    if not state._merged_points:
        state._intersection_points = []
        return

    excluded = state._excluded_points
    filtered = []
    for mp in state._merged_points:
        if mp in excluded:
            continue
        if state._box is not None and cv2.pointPolygonTest(state._box, mp, False) < 0:
            continue
        filtered.append(mp)
    state._intersection_points = filtered


def extend_independent_points():
    """
    对独立交点（不在任何红色线段上的点），沿经过它的所有蓝色线段
    向包围盒中心方向延伸，生成红色延长线。
    输入: _merged_points, _merged_lines, _box
    输出: _extension_lines
    """
    state._extension_lines = []
    if state._box is None or not state._merged_points:
        return

    # Ensure excluded_points and red_endpoints exist (set by assign_red_segments)
    excluded_points = state._excluded_points
    red_endpoints = state._red_endpoints

    centroid = np.mean(state._box, axis=0)

    for mp in state._merged_points:
        if mp in excluded_points:
            continue
        if cv2.pointPolygonTest(state._box, mp, False) < 0:
            continue

        is_endpoint = mp in red_endpoints
        red_on = []
        for seg in state._red_segments:
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
        for ml in state._merged_lines:
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

            endpoint = clip_ray_to_box(mp, (dir_x, dir_y), state._box)

            ex = endpoint[0] - mp[0]
            ey = endpoint[1] - mp[1]
            actual_len = np.sqrt(ex * ex + ey * ey)
            if actual_len > EXTEND_LENGTH:
                endpoint = (int(round(mp[0] + dir_x * EXTEND_LENGTH)),
                            int(round(mp[1] + dir_y * EXTEND_LENGTH)))

            state._extension_lines.append((mp, endpoint))


def extend_lines():
    """
    将每条原始 Hough 检测线段以中点为中心、按 LINE_EXTEND_FACTOR 倍率向两端延长。
    输入: _raw_lines（原始 Hough 线段）
    输出: _extended_lines（延长后的线段）
    """
    state._extended_lines = []
    if not state._raw_lines:
        return
    for p1, p2 in state._raw_lines:
        x1, y1 = p1
        x2, y2 = p2
        ext1 = (int(round(LINE_EXTEND_FACTOR * x1 - (LINE_EXTEND_FACTOR - 1) * x2)),
                int(round(LINE_EXTEND_FACTOR * y1 - (LINE_EXTEND_FACTOR - 1) * y2)))
        ext2 = (int(round(LINE_EXTEND_FACTOR * x2 - (LINE_EXTEND_FACTOR - 1) * x1)),
                int(round(LINE_EXTEND_FACTOR * y2 - (LINE_EXTEND_FACTOR - 1) * y1)))
        state._extended_lines.append((ext1, ext2))


def merge_lines():
    """
    将接近平行的延长线合并为单一长线段。
    先按极坐标角度聚类，再在同类中按 rho 距离二次聚类，
    最终将每簇内所有端点投影到中位方向并取极值点作为合并结果。
    输入: _extended_lines
    输出: _merged_lines
    """
    state._merged_lines = []
    if not state._extended_lines:
        return

    # --- Convert to polar representation (theta, rho) ---
    lines_polar = []
    for idx, (p1, p2) in enumerate(state._extended_lines):
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

            state._merged_lines.append((all_points[min_idx], all_points[max_idx]))


def _is_independent_blue(blue_line):
    pt1, pt2 = blue_line
    for rp1, rp2 in state._red_segments:
        if (point_on_segment(rp1[0], rp1[1], pt1[0], pt1[1], pt2[0], pt2[1], tol=3) and
            point_on_segment(rp2[0], rp2[1], pt1[0], pt1[1], pt2[0], pt2[1], tol=3)):
            return False
    for ix, iy in state._merged_points:
        if point_on_segment(ix, iy, pt1[0], pt1[1], pt2[0], pt2[1], tol=3):
            return False
    return True
