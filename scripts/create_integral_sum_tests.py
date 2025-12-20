"""
生成积分和求和测试图像
上标/下标位于符号的右上方和右下方（而非正上正下）
"""
import cv2
import numpy as np
import os
import random

def load_random_symbol(symbol_name):
    """从训练集加载随机一个符号图像"""
    folder = f'data/training/{symbol_name}'
    if not os.path.exists(folder):
        print(f'警告: 找不到文件夹 {folder}')
        return None
    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    if not files:
        print(f'警告: {folder} 中没有png文件')
        return None
    img_path = os.path.join(folder, random.choice(files))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

def resize_symbol(img, target_height):
    """按目标高度缩放符号，保持宽高比"""
    if img is None:
        return None
    h, w = img.shape
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))

def create_canvas(width, height):
    """创建白色画布"""
    return np.ones((height, width), dtype=np.uint8) * 255

def paste_symbol(canvas, symbol, x, y):
    """将符号粘贴到画布上"""
    if symbol is None:
        return
    h, w = symbol.shape
    ch, cw = canvas.shape
    
    # 确保不超出边界
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + w), min(ch, y + h)
    
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    
    if x2 > x1 and y2 > y1:
        # 使用最小值合成（黑色部分保留）
        canvas[y1:y2, x1:x2] = np.minimum(canvas[y1:y2, x1:x2], symbol[sy1:sy2, sx1:sx2])

def create_integral_with_limits(lower, upper, output_name):
    """
    创建带上下限的积分符号
    上下限位于积分号的右上方和右下方
    """
    # 加载符号
    int_img = load_random_symbol('int')
    lower_img = load_random_symbol(lower)
    upper_img = load_random_symbol(upper)
    
    if int_img is None:
        print(f'无法创建 {output_name}: 积分符号不可用')
        return
    
    # 设置尺寸
    int_height = 50
    limit_height = 18  # 上下限符号较小
    
    int_img = resize_symbol(int_img, int_height)
    lower_img = resize_symbol(lower_img, limit_height) if lower_img is not None else None
    upper_img = resize_symbol(upper_img, limit_height) if upper_img is not None else None
    
    int_h, int_w = int_img.shape
    
    # 计算画布大小
    canvas_width = int_w + 40  # 给上下限留空间
    canvas_height = int_height + 20
    
    canvas = create_canvas(canvas_width, canvas_height)
    
    # 积分符号居中放置
    int_x = 5
    int_y = 10
    paste_symbol(canvas, int_img, int_x, int_y)
    
    # 下限：在积分符号的右下方
    if lower_img is not None:
        lower_x = int_x + int_w - 5  # 稍微向左偏移，更贴近积分号
        lower_y = int_y + int_height - limit_height + 5  # 底部对齐，稍微偏下
        paste_symbol(canvas, lower_img, lower_x, lower_y)
    
    # 上限：在积分符号的右上方
    if upper_img is not None:
        upper_x = int_x + int_w - 5
        upper_y = int_y - 5  # 顶部对齐，稍微偏上
        paste_symbol(canvas, upper_img, upper_x, upper_y)
    
    # 保存
    output_path = f'data/test/generated/{output_name}'
    cv2.imwrite(output_path, canvas)
    print(f'已创建: {output_path}')

def create_sum_with_limits(lower, upper, output_name):
    """
    创建带上下限的求和符号
    上下限位于求和符号的右上方和右下方
    """
    # 加载符号
    sum_img = load_random_symbol('sum')
    lower_img = load_random_symbol(lower)
    upper_img = load_random_symbol(upper)
    
    if sum_img is None:
        print(f'无法创建 {output_name}: 求和符号不可用')
        return
    
    # 设置尺寸
    sum_height = 40
    limit_height = 16
    
    sum_img = resize_symbol(sum_img, sum_height)
    lower_img = resize_symbol(lower_img, limit_height) if lower_img is not None else None
    upper_img = resize_symbol(upper_img, limit_height) if upper_img is not None else None
    
    sum_h, sum_w = sum_img.shape
    
    # 计算画布大小
    canvas_width = sum_w + 35
    canvas_height = sum_height + 20
    
    canvas = create_canvas(canvas_width, canvas_height)
    
    # 求和符号放置
    sum_x = 5
    sum_y = 10
    paste_symbol(canvas, sum_img, sum_x, sum_y)
    
    # 下限：在求和符号的右下方
    if lower_img is not None:
        lower_x = sum_x + sum_w - 3
        lower_y = sum_y + sum_height - limit_height + 3
        paste_symbol(canvas, lower_img, lower_x, lower_y)
    
    # 上限：在求和符号的右上方
    if upper_img is not None:
        upper_x = sum_x + sum_w - 3
        upper_y = sum_y - 3
        paste_symbol(canvas, upper_img, upper_x, upper_y)
    
    # 保存
    output_path = f'data/test/generated/{output_name}'
    cv2.imwrite(output_path, canvas)
    print(f'已创建: {output_path}')

def create_simple_integral(output_name):
    """创建简单积分符号（无上下限）"""
    int_img = load_random_symbol('int')
    if int_img is None:
        return
    
    int_img = resize_symbol(int_img, 50)
    h, w = int_img.shape
    
    canvas = create_canvas(w + 20, h + 20)
    paste_symbol(canvas, int_img, 10, 10)
    
    output_path = f'data/test/generated/{output_name}'
    cv2.imwrite(output_path, canvas)
    print(f'已创建: {output_path}')

def create_simple_sum(output_name):
    """创建简单求和符号（无上下限）"""
    sum_img = load_random_symbol('sum')
    if sum_img is None:
        return
    
    sum_img = resize_symbol(sum_img, 40)
    h, w = sum_img.shape
    
    canvas = create_canvas(w + 20, h + 20)
    paste_symbol(canvas, sum_img, 10, 10)
    
    output_path = f'data/test/generated/{output_name}'
    cv2.imwrite(output_path, canvas)
    print(f'已创建: {output_path}')

if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs('data/test/generated', exist_ok=True)
    
    print('=== 生成积分测试图像 ===')
    
    # 简单积分
    create_simple_integral('integral_simple.png')
    
    # 带上下限的积分
    create_integral_with_limits('0', '1', 'integral_0_1.png')
    create_integral_with_limits('a', 'b', 'integral_a_b.png')
    create_integral_with_limits('0', 'infty', 'integral_0_infty.png')
    
    print('\n=== 生成求和测试图像 ===')
    
    # 简单求和
    create_simple_sum('sum_simple.png')
    
    # 带下限的求和
    create_sum_with_limits('k', None, 'sum_k.png')
    
    # 带上下限的求和
    create_sum_with_limits('1', 'n', 'sum_1_n.png')
    create_sum_with_limits('i', 'N', 'sum_i_N.png')
    
    print('\n=== 完成 ===')
