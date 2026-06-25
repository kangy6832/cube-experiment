"""
几何计算工具函数。

提供:
- _line_intersection_pair: 两条无限延长线的交点
- _segment_intersection_pair: 两条线段的交点 (需同时落在两线段上)
- point_on_segment: 判断点是否在线段上 (带容差)
- point_to_line_distance: 点到直线的垂直距离
- clip_ray_to_box: 将射线限制在凸多边形内部，返回与边界的交点
"""

import numpy as np

from config import EXTEND_LENGTH


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
