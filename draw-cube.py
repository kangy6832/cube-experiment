import cv2
import numpy as np
import os

# --- 图像参数 ---
width, height = 800, 600
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "cube")
os.makedirs(output_dir, exist_ok=True)

# --- 立方体 3D 顶点 ---
s = 150
vertices_3d = np.array([
    [-s, -s, -s],  # 0
    [ s, -s, -s],  # 1
    [ s,  s, -s],  # 2
    [-s,  s, -s],  # 3
    [-s, -s,  s],  # 4
    [ s, -s,  s],  # 5
    [ s,  s,  s],  # 6
    [-s,  s,  s],  # 7
], dtype=np.float64)

# --- 12 条棱 ---
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 后面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 前面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 连接前后
]

# --- 6 个面 ---
faces = [
    (5, 4, 7, 6),  # 后面
    (0, 1, 2, 3),  # 前面
    (4, 0, 3, 7),  # 左面
    (1, 5, 6, 2),  # 右面
    (7, 6, 2, 3),  # 上面
    (4, 5, 1, 0),  # 下面
]

# --- 透视投影参数 ---
fov = 600
viewer_z = 600
viewer_dir = np.array([0, 0, -1])  # 相机看向 -z


def draw_cube(yaw, pitch, roll):
    """根据 yaw/pitch/roll（度）绘制立方体 2D 投影，返回 BGR 图像"""
    # radians : 角度转弧度
    angle_y = np.radians(yaw)  
    angle_x = np.radians(pitch)
    angle_z = np.radians(roll)

    # 旋转矩阵
    # array : 创建数组(3x3)的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x),  np.cos(angle_x)],
    ])
    Ry = np.array([
        [ np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0, 0, 1],
    ])
    # @ : 矩阵乘法，组合三个轴的旋转。
    R = Rz @ Ry @ Rx 

    # 旋转顶点
    rotated = (R @ vertices_3d.T).T

    # 投影到 2D
    def project(p):
        z = p[2] + viewer_z
        factor = fov / z
        x = p[0] * factor + width / 2
        y = -p[1] * factor + height / 2
        return np.array([x, y], dtype=np.int32)

    vertices_2d = np.array([project(p) for p in rotated])

    # 面法线判断可见性（使用外法线朝向）
    visible_faces = set()
    for i, face in enumerate(faces):
        v0, v1, v2 = [rotated[j] for j in face[:3]]
        n = np.cross(v1 - v0, v2 - v0)
        n = n / np.linalg.norm(n)
        # 外法线朝向相机（+z 方向）→ 可见
        if n[2] > 0:
            visible_faces.add(i)

    # 可见棱：只要属于任意一个可见面就算可见（包括可见面与隐藏面之间的棱）
    visible_edges = set()
    for i in range(len(faces)):
        if i in visible_faces:
            f = faces[i]
            for j in range(len(f)):
                edge = tuple(sorted((f[j], f[(j + 1) % len(f)])))
                visible_edges.add(edge)

    # 绘制
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    # 所有棱（实线）
    for e in edges:
        cv2.line(img, tuple(vertices_2d[e[0]]), tuple(vertices_2d[e[1]]), (0, 255, 0), thickness=2)

    # 角点（红色）
    for v in vertices_2d:
        cv2.circle(img, tuple(v), 6, (0, 0, 255), -1)
        cv2.circle(img, tuple(v), 6, (0, 0, 0), 1)

    # 顶点编号
    for i, v in enumerate(vertices_2d):
        cv2.putText(img, str(i), (v[0] + 8, v[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 角度标注
    info = f"Yaw={yaw:+.0f}  Pitch={pitch:+.0f}  Roll={roll:+.0f}"
    cv2.putText(img, info, (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    return img


def main():
    # print("新的代码")
    # 创建窗口
    cv2.namedWindow("Cube", cv2.WINDOW_AUTOSIZE)

    # 滑动条放在图像窗口下方（0~360 → -180~180），回调为空，刷新由主循环统一处理
    cv2.createTrackbar("Yaw", "Cube", 215, 360, lambda v: None)
    cv2.createTrackbar("Pitch", "Cube", 115, 360, lambda v: None)
    cv2.createTrackbar("Roll", "Cube", 180, 360, lambda v: None)

    def get_angles():
        yaw = cv2.getTrackbarPos("Yaw", "Cube") - 180
        pitch = cv2.getTrackbarPos("Pitch", "Cube") - 180
        roll = cv2.getTrackbarPos("Roll", "Cube") - 180
        return yaw, pitch, roll

    def redraw():
        yaw, pitch, roll = get_angles()
        img = draw_cube(yaw, pitch, roll)
        cv2.imshow("Cube", img)

    # 初始绘制
    redraw()

    print("交互式立方体查看器")
    print("  拖动滑动条旋转立方体")
    print("  按 S 键保存图像")
    print("  按 Q 或 ESC 退出")

    while True:
        redraw()
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            yaw, pitch, roll = get_angles()
            filename = f"cube_yaw{yaw:+03d}_pitch{pitch:+03d}_roll{roll:+03d}.png"
            filepath = os.path.join(output_dir, filename)
            img = draw_cube(yaw, pitch, roll)
            cv2.imwrite(filepath, img)
            print(f"已保存: {filepath}")
        elif key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
