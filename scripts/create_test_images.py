"""
创建分数和上下标测试图像
使用训练数据中的符号拼接
"""

import cv2
import numpy as np
import os
import sys
import random

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TRAINING_DIR = "data/training"
OUTPUT_DIR = "data/test/generated"


def load_symbol(label, scale=1.0):
    """
    从训练数据加载符号图像
    
    Args:
        label: 符号标签（对应训练数据文件夹名）
        scale: 缩放比例
        
    Returns:
        符号图像（灰度，白底黑字）
    """
    label_dir = os.path.join(TRAINING_DIR, label)
    if not os.path.exists(label_dir):
        print(f"警告: 找不到训练数据 {label_dir}")
        return None
    
    # 获取随机一张图像
    images = [f for f in os.listdir(label_dir) if f.endswith('.png')]
    if not images:
        print(f"警告: {label_dir} 中没有图像")
        return None
    
    img_path = os.path.join(label_dir, random.choice(images))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # 缩放
    if scale != 1.0:
        new_w = max(1, int(img.shape[1] * scale))
        new_h = max(1, int(img.shape[0] * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def create_fraction(numerator_labels, denominator_labels, output_name):
    """
    创建分数图像
    
    Args:
        numerator_labels: 分子符号列表
        denominator_labels: 分母符号列表
        output_name: 输出文件名
    """
    # 加载分子符号
    num_images = []
    for label in numerator_labels:
        img = load_symbol(label)
        if img is not None:
            num_images.append(img)
    
    # 加载分母符号
    den_images = []
    for label in denominator_labels:
        img = load_symbol(label)
        if img is not None:
            den_images.append(img)
    
    if not num_images or not den_images:
        print(f"无法创建分数: 缺少分子或分母图像")
        return None
    
    # 计算分子行的尺寸
    num_height = max(img.shape[0] for img in num_images)
    num_width = sum(img.shape[1] for img in num_images) + 2 * (len(num_images) - 1)  # 间距
    
    # 计算分母行的尺寸
    den_height = max(img.shape[0] for img in den_images)
    den_width = sum(img.shape[1] for img in den_images) + 2 * (len(den_images) - 1)
    
    # 分数线宽度和高度
    line_width = max(num_width, den_width) + 10
    line_height = 2
    
    # 间距
    vertical_gap = 5
    
    # 计算画布尺寸
    canvas_width = line_width + 20
    canvas_height = num_height + line_height + den_height + 2 * vertical_gap + 20
    
    # 创建白色画布
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
    
    # 计算起始位置（居中）
    start_x = (canvas_width - line_width) // 2
    
    # 绘制分子
    num_start_x = start_x + (line_width - num_width) // 2
    num_y = 10
    current_x = num_start_x
    for img in num_images:
        h, w = img.shape
        y_offset = num_y + (num_height - h) // 2
        canvas[y_offset:y_offset+h, current_x:current_x+w] = np.minimum(
            canvas[y_offset:y_offset+h, current_x:current_x+w], img
        )
        current_x += w + 2
    
    # 绘制分数线
    line_y = num_y + num_height + vertical_gap
    cv2.line(canvas, (start_x, line_y), (start_x + line_width, line_y), 0, line_height)
    
    # 绘制分母
    den_start_x = start_x + (line_width - den_width) // 2
    den_y = line_y + line_height + vertical_gap
    current_x = den_start_x
    for img in den_images:
        h, w = img.shape
        y_offset = den_y + (den_height - h) // 2
        canvas[y_offset:y_offset+h, current_x:current_x+w] = np.minimum(
            canvas[y_offset:y_offset+h, current_x:current_x+w], img
        )
        current_x += w + 2
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, canvas)
    print(f"创建分数图像: {output_path}")
    return output_path


def create_superscript(base_label, superscript_labels, output_name):
    """
    创建上标图像
    
    Args:
        base_label: 基础符号标签
        superscript_labels: 上标符号列表
        output_name: 输出文件名
    """
    base_img = load_symbol(base_label)
    if base_img is None:
        return None
    
    sup_images = []
    for label in superscript_labels:
        img = load_symbol(label, scale=0.7)  # 上标缩小
        if img is not None:
            sup_images.append(img)
    
    if not sup_images:
        return None
    
    # 计算上标行尺寸
    sup_height = max(img.shape[0] for img in sup_images)
    sup_width = sum(img.shape[1] for img in sup_images) + 2 * (len(sup_images) - 1)
    
    base_h, base_w = base_img.shape
    
    # 画布尺寸
    canvas_width = base_w + sup_width + 15
    canvas_height = base_h + sup_height // 2 + 10
    
    # 创建画布
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
    
    # 绘制基础符号（靠下）
    base_y = canvas_height - base_h - 5
    base_x = 5
    canvas[base_y:base_y+base_h, base_x:base_x+base_w] = np.minimum(
        canvas[base_y:base_y+base_h, base_x:base_x+base_w], base_img
    )
    
    # 绘制上标（靠上，在基础符号右侧）
    sup_y = 5
    sup_x = base_x + base_w + 2
    current_x = sup_x
    for img in sup_images:
        h, w = img.shape
        canvas[sup_y:sup_y+h, current_x:current_x+w] = np.minimum(
            canvas[sup_y:sup_y+h, current_x:current_x+w], img
        )
        current_x += w + 1
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, canvas)
    print(f"创建上标图像: {output_path}")
    return output_path


def create_subscript(base_label, subscript_labels, output_name):
    """
    创建下标图像
    
    Args:
        base_label: 基础符号标签
        subscript_labels: 下标符号列表
        output_name: 输出文件名
    """
    base_img = load_symbol(base_label)
    if base_img is None:
        return None
    
    sub_images = []
    for label in subscript_labels:
        img = load_symbol(label, scale=0.7)  # 下标缩小
        if img is not None:
            sub_images.append(img)
    
    if not sub_images:
        return None
    
    # 计算下标行尺寸
    sub_height = max(img.shape[0] for img in sub_images)
    sub_width = sum(img.shape[1] for img in sub_images) + 2 * (len(sub_images) - 1)
    
    base_h, base_w = base_img.shape
    
    # 画布尺寸
    canvas_width = base_w + sub_width + 15
    canvas_height = base_h + sub_height // 2 + 10
    
    # 创建画布
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
    
    # 绘制基础符号（靠上）
    base_y = 5
    base_x = 5
    canvas[base_y:base_y+base_h, base_x:base_x+base_w] = np.minimum(
        canvas[base_y:base_y+base_h, base_x:base_x+base_w], base_img
    )
    
    # 绘制下标（靠下，在基础符号右侧）
    sub_y = canvas_height - sub_height - 5
    sub_x = base_x + base_w + 2
    current_x = sub_x
    for img in sub_images:
        h, w = img.shape
        canvas[sub_y:sub_y+h, current_x:current_x+w] = np.minimum(
            canvas[sub_y:sub_y+h, current_x:current_x+w], img
        )
        current_x += w + 1
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, canvas)
    print(f"创建下标图像: {output_path}")
    return output_path


def main():
    print("创建测试图像...")
    print(f"训练数据目录: {TRAINING_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建分数测试图像
    print("\n=== 创建分数测试图像 ===")
    create_fraction(["1"], ["2"], "frac_1_2.png")
    create_fraction(["3"], ["4"], "frac_3_4.png")
    create_fraction(["A"], ["B"], "frac_A_B.png")
    create_fraction(["X"], ["Y"], "frac_X_Y.png")
    create_fraction(["alpha"], ["beta"], "frac_alpha_beta.png")
    create_fraction(["1", "plus", "2"], ["3"], "frac_1plus2_3.png")
    create_fraction(["X"], ["2"], "frac_X_2.png")
    
    # 创建上标测试图像
    print("\n=== 创建上标测试图像 ===")
    create_superscript("X", ["2"], "x_sup_2.png")
    create_superscript("A", ["N"], "a_sup_n.png")
    create_superscript("E", ["X"], "e_sup_x.png")
    create_superscript("2", ["3"], "2_sup_3.png")
    
    # 创建下标测试图像
    print("\n=== 创建下标测试图像 ===")
    create_subscript("X", ["1"], "x_sub_1.png")
    create_subscript("A", ["N"], "a_sub_n.png")
    create_subscript("Y", ["0"], "y_sub_0.png")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
