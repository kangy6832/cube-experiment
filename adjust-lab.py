import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def lab_threshold(image, l_min, l_max, a_min, a_max, b_min, b_max):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lower = (l_min, a_min, b_min)
    upper = (l_max, a_max, b_max)
    return cv2.inRange(lab, lower, upper)


def main():
    photos_dir = "photos"

    image_files = sorted(glob.glob(os.path.join(photos_dir, "*.jpg")) +
                         glob.glob(os.path.join(photos_dir, "*.png")) +
                         glob.glob(os.path.join(photos_dir, "*.jpeg")))

    if not image_files:
        print(f"错误：'{photos_dir}' 目录中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")
    print("A/D 或 ← → 切换图片，关闭窗口退出")

    current_index = [0]  # 使用列表以便在回调中修改

    # 读取第一张图片
    image = cv2.imread(image_files[0])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = lab_threshold(image, 0, 255, 0, 255, 0, 255)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(bottom=0.35)

    # 显示图片
    im1 = ax1.imshow(image_rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    im2 = ax2.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("LAB Binary")
    ax2.axis("off")

    # 滑动条位置
    axcolor = "lightgoldenrodyellow"
    ax_l_min = fig.add_axes([0.15, 0.25, 0.7, 0.03], facecolor=axcolor)
    ax_l_max = fig.add_axes([0.15, 0.20, 0.7, 0.03], facecolor=axcolor)
    ax_a_min = fig.add_axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
    ax_a_max = fig.add_axes([0.15, 0.10, 0.7, 0.03], facecolor=axcolor)
    ax_b_min = fig.add_axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)
    ax_b_max = fig.add_axes([0.15, 0.00, 0.7, 0.03], facecolor=axcolor)

    # 创建滑动条
    s_l_min = Slider(ax_l_min, "L min", 0, 255, valinit=0, valstep=1)
    s_l_max = Slider(ax_l_max, "L max", 0, 255, valinit=255, valstep=1)
    s_a_min = Slider(ax_a_min, "A min", 0, 255, valinit=0, valstep=1)
    s_a_max = Slider(ax_a_max, "A max", 0, 255, valinit=255, valstep=1)
    s_b_min = Slider(ax_b_min, "B min", 0, 255, valinit=0, valstep=1)
    s_b_max = Slider(ax_b_max, "B max", 0, 255, valinit=255, valstep=1)

    # 更新函数
    def update(val):
        img = cv2.imread(image_files[current_index[0]])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = lab_threshold(
            img,
            int(s_l_min.val), int(s_l_max.val),
            int(s_a_min.val), int(s_a_max.val),
            int(s_b_min.val), int(s_b_max.val)
        )

        im1.set_data(img_rgb)
        # 转换为 RGB 显示
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        im2.set_data(result_rgb)

        info = f"[{current_index[0] + 1}/{len(image_files)}] {os.path.basename(image_files[current_index[0]])}"
        ax1.set_title(info)

        fig.canvas.draw()
        fig.canvas.flush_events()

    # 绑定滑动条事件
    s_l_min.on_changed(update)
    s_l_max.on_changed(update)
    s_a_min.on_changed(update)
    s_a_max.on_changed(update)
    s_b_min.on_changed(update)
    s_b_max.on_changed(update)

    # 键盘事件
    def on_key(event):
        if event.key in ["left", "a", "A"]:
            current_index[0] = (current_index[0] - 1) % len(image_files)
            update(None)
        elif event.key in ["right", "d", "D"]:
            current_index[0] = (current_index[0] + 1) % len(image_files)
            update(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    print("程序已退出")


if __name__ == "__main__":
    main()
