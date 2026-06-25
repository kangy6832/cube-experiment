import cv2
import numpy as np

import state


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
