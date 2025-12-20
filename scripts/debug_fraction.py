"""
调试分数和上下标识别问题
"""

import cv2
import numpy as np
import os
import sys

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import ImagePreprocessor
from src.segmentation import SymbolSegmenter
from src.recognition import SymbolRecognizer
from src.structure_analysis import StructureAnalyzer, SpatialRelation
from src.utils import BoundingBox

def debug_image(image_path):
    """调试单张图像的识别过程"""
    print(f"\n{'='*60}")
    print(f"调试图像: {image_path}")
    print('='*60)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    print(f"图像尺寸: {image.shape}")
    
    # 预处理
    preprocessor = ImagePreprocessor()
    binary = preprocessor.process(image)
    print(f"二值图像尺寸: {binary.shape}")
    
    # 分割
    segmenter = SymbolSegmenter()
    symbols = segmenter.segment(binary)
    print(f"\n检测到 {len(symbols)} 个符号:")
    
    for i, sym in enumerate(symbols):
        bbox = sym.bbox
        print(f"  [{i}] bbox=({bbox.x}, {bbox.y}, {bbox.width}, {bbox.height}), "
              f"is_fraction_line={sym.is_fraction_line}")
    
    # 识别
    recognizer = SymbolRecognizer()
    recognized = recognizer.recognize_symbols(symbols)
    print(f"\n识别结果:")
    
    for i, sym in enumerate(recognized):
        bbox = sym.bbox
        print(f"  [{i}] label='{sym.label}', confidence={sym.confidence:.3f}, "
              f"bbox=({bbox.x}, {bbox.y}, w={bbox.width}, h={bbox.height}), "
              f"center=({bbox.center_x:.1f}, {bbox.center_y:.1f}), "
              f"is_fraction_line={sym.is_fraction_line}")
    
    # 结构分析
    analyzer = StructureAnalyzer()
    
    # 手动计算空间关系
    print(f"\n空间关系分析:")
    relations = analyzer._compute_relations(recognized)
    for (i, j), rel in relations.items():
        print(f"  符号[{i}]({recognized[i].label}) -> 符号[{j}]({recognized[j].label}): {rel.value}")
    
    # 检查分数分组
    print(f"\n分数分组检查:")
    _, groups = analyzer._process_special_structures(recognized, relations)
    for g in groups:
        print(f"  分组: {g}")
    
    # 检查分数线相关关系
    print(f"\n分数线关系检查:")
    for i, sym in enumerate(recognized):
        if sym.is_fraction_line:
            print(f"  分数线 [{i}]:")
            fraction_bbox = sym.bbox
            for j, other in enumerate(recognized):
                if i == j:
                    continue
                other_bbox = other.bbox
                
                # 计算水平覆盖
                overlap_left = max(fraction_bbox.x, other_bbox.x)
                overlap_right = min(fraction_bbox.x2, other_bbox.x2)
                h_coverage = 0
                if overlap_right > overlap_left:
                    h_coverage = (overlap_right - overlap_left) / other_bbox.width
                
                # 计算垂直距离
                dy = other_bbox.center_y - fraction_bbox.center_y
                
                position = "上方" if dy < 0 else "下方"
                print(f"    符号[{j}]({other.label}): {position}, dy={dy:.1f}, "
                      f"h_coverage={h_coverage:.2f}")
    
    # 生成 LaTeX
    syntax_tree, latex = analyzer.analyze(recognized)
    print(f"\n生成的 LaTeX: {latex}")
    
    return latex


def create_test_image(output_path, symbols_config):
    """
    使用训练数据创建测试图像
    
    symbols_config: list of dict, 每个dict包含:
        - 'label': 符号标签（对应训练数据文件夹名）
        - 'x': x坐标
        - 'y': y坐标
        - 'scale': 缩放比例（可选，默认1.0）
    """
    training_dir = "data/training"
    
    # 首先确定画布大小
    canvas_width = 200
    canvas_height = 100
    
    # 创建白色画布
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
    
    for config in symbols_config:
        label = config['label']
        x = config['x']
        y = config['y']
        scale = config.get('scale', 1.0)
        
        # 查找训练图像
        label_dir = os.path.join(training_dir, label)
        if not os.path.exists(label_dir):
            print(f"警告: 找不到训练数据 {label_dir}")
            continue
        
        # 获取第一张图像
        images = [f for f in os.listdir(label_dir) if f.endswith('.png')]
        if not images:
            print(f"警告: {label_dir} 中没有图像")
            continue
        
        img_path = os.path.join(label_dir, images[0])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # 缩放
        if scale != 1.0:
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # 粘贴到画布
        h, w = img.shape
        x_end = min(x + w, canvas_width)
        y_end = min(y + h, canvas_height)
        
        if x < canvas_width and y < canvas_height:
            paste_w = x_end - x
            paste_h = y_end - y
            
            # 使用最小值合并（黑色为前景）
            canvas[y:y_end, x:x_end] = np.minimum(canvas[y:y_end, x:x_end], img[:paste_h, :paste_w])
    
    cv2.imwrite(output_path, canvas)
    print(f"创建测试图像: {output_path}")
    return output_path


def create_fraction_line(output_path, width=50, height=3):
    """创建分数线图像"""
    img = np.ones((height + 10, width + 10), dtype=np.uint8) * 255
    cv2.line(img, (5, height//2 + 5), (width + 5, height//2 + 5), 0, height)
    cv2.imwrite(output_path, img)
    return output_path


def main():
    # 测试生成的图像
    test_dir = "data/test/generated"
    
    if os.path.exists(test_dir):
        # 测试分数图像
        print("\n" + "="*60)
        print("测试分数图像")
        print("="*60)
        
        frac_files = ['frac_1_2.png', 'frac_3_4.png', 'frac_A_B.png']
        for f in frac_files:
            path = os.path.join(test_dir, f)
            if os.path.exists(path):
                debug_image(path)
        
        # 测试上标图像
        print("\n" + "="*60)
        print("测试上标图像")
        print("="*60)
        
        sup_files = ['x_sup_2.png', 'a_sup_n.png']
        for f in sup_files:
            path = os.path.join(test_dir, f)
            if os.path.exists(path):
                debug_image(path)
        
        # 测试下标图像
        print("\n" + "="*60)
        print("测试下标图像")
        print("="*60)
        
        sub_files = ['x_sub_1.png', 'a_sub_n.png']
        for f in sub_files:
            path = os.path.join(test_dir, f)
            if os.path.exists(path):
                debug_image(path)
    
    # 查看训练数据样本大小
    print("\n" + "="*60)
    print("训练数据信息")
    print("="*60)
    
    training_dir = "data/training"
    sample_img = cv2.imread(os.path.join(training_dir, "1", os.listdir(os.path.join(training_dir, "1"))[0]), cv2.IMREAD_GRAYSCALE)
    if sample_img is not None:
        print(f"训练数据样本大小: {sample_img.shape}")


if __name__ == "__main__":
    main()
