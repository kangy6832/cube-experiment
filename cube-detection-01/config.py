"""
集中管理流水线的所有可调参数。
包括: 目录路径、LAB 阈值、形态学参数、Hough 变换参数、线段合并阈值。
修改此处数值即可调整检测行为，无需改动逻辑代码。
"""

import numpy as np

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
