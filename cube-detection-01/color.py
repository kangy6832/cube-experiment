import cv2

from config import LAB_LOWER, LAB_UPPER, MORPH_KERNEL_SIZE, MORPH_ITERATIONS


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
