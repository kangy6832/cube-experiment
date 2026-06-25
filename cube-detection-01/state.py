"""
全局状态存储 — 流水线各阶段的中间结果。

所有检测阶段通过读写此模块的模块级变量传递数据，
避免在函数签名中传递大量参数。每次 pipeline_detection
调用前由 _reset_drawing_globals() 清空，确保帧间隔离。

变量命名前缀 _ 表示私有，仅供 pipeline 内部使用。
"""

# ============ Drawing Element Storage ============
# Populated by stage functions, consumed by draw_all_elements.

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
