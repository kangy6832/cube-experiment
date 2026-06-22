import cv2
import os
from datetime import datetime


def take_photos(save_dir="photos", camera_index=0):
    """从摄像头拍照并保存到指定目录"""

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 打开摄像头（使用 V4L2 后端避免 Qt 问题）
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    print(f"摄像头已打开 (索引: {camera_index})")
    print("按 'c' 键拍照，按 'q' 键退出")

    # 创建窗口（可调整大小）
    window_name = "拍照预览"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    photo_count = 0

    while True:
        # 读取帧
        ret, frame = cap.read()

        if not ret:
            print("错误：无法读取帧")
            break

        # 显示实时预览
        cv2.imshow(window_name, frame)

        # 等待按键（需要足够时间让窗口刷新）
        key = cv2.waitKey(50) & 0xFF

        if key == ord("c"):
            # 生成文件名（使用时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            # 保存照片
            cv2.imwrite(filepath, frame)
            photo_count += 1
            print(f"照片已保存: {filepath} (共 {photo_count} 张)")

        elif key == ord("q") or key == 27:  # 'q' 或 ESC
            print("退出拍照")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n共拍摄 {photo_count} 张照片，保存在 '{save_dir}/' 目录")


if __name__ == "__main__":
    take_photos(camera_index=4)
