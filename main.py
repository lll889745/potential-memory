"""
手写数学公式识别与语义理解系统 - 主程序
======================================

该程序整合所有模块，提供完整的公式识别流程。

使用方法：
    python main.py --image path/to/formula.png
    python main.py --demo  # 运行演示
"""

import argparse
import logging
import os
import sys
from typing import Optional, Dict, Any

import cv2
import numpy as np

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import ImagePreprocessor
from src.segmentation import SymbolSegmenter
from src.recognition import SymbolRecognizer, FeatureExtractor
from src.structure_analysis import StructureAnalyzer
from src.semantic import SemanticProcessor
from src.config import DEFAULT_CONFIG
from src.utils import visualize_symbols, visualize_syntax_tree, load_image, save_image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FormulaRecognitionSystem:
    """
    手写数学公式识别系统
    
    整合所有模块，提供端到端的公式识别流程。
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化识别系统
        
        Args:
            model_path: 预训练模型路径（可选）
        """
        logger.info("初始化公式识别系统...")
        
        # 初始化各模块
        self.preprocessor = ImagePreprocessor()
        self.segmenter = SymbolSegmenter()
        self.recognizer = SymbolRecognizer()
        self.structure_analyzer = StructureAnalyzer()
        self.semantic_processor = SemanticProcessor()
        
        # 加载模型（如果有）
        if model_path and os.path.exists(model_path):
            self.recognizer.load_model(model_path)
            logger.info(f"已加载模型: {model_path}")
        else:
            logger.warning("未加载符号识别模型，需要先训练或加载模型")
        
        logger.info("系统初始化完成")
    
    def recognize(self, image: np.ndarray, 
                  return_intermediate: bool = False) -> Dict[str, Any]:
        """
        识别公式
        
        Args:
            image: 输入图像（灰度或彩色）
            return_intermediate: 是否返回中间结果
            
        Returns:
            识别结果字典
        """
        result = {
            'success': False,
            'latex': '',
            'semantic': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 获取图像尺寸
            if len(image.shape) == 3:
                h, w = image.shape[:2]
            else:
                h, w = image.shape
            
            # 对于小图像（如单个符号的32x32或64x64），直接识别整张图片
            is_single_symbol = max(h, w) <= 64
            
            # 步骤1：图像预处理
            logger.info("步骤1: 图像预处理")
            if return_intermediate:
                preprocess_result = self.preprocessor.process(image, return_intermediate=True)
                binary = preprocess_result['cleaned']
                result['preprocessing'] = preprocess_result
            else:
                binary = self.preprocessor.process(image)
            
            # 步骤2：符号分割（小图像跳过分割，直接作为单个符号）
            logger.info("步骤2: 符号分割")
            if is_single_symbol:
                # 小图像：整张图片作为一个符号
                from src.utils import Symbol, BoundingBox
                symbol = Symbol(
                    image=binary,
                    bbox=BoundingBox(0, 0, w, h)
                )
                symbols = [symbol]
                logger.info(f"小图像模式：整张图片作为单个符号处理")
            else:
                symbols = self.segmenter.segment(binary)
            
            result['num_symbols'] = len(symbols)
            
            if not symbols:
                result['warnings'].append("未检测到任何符号")
                return result
            
            # 步骤3：符号识别
            logger.info("步骤3: 符号识别")
            if self.recognizer.is_trained:
                symbols = self.recognizer.recognize_symbols(symbols)
            else:
                result['warnings'].append("符号识别模型未训练，跳过识别步骤")
                # 使用占位标签
                for i, sym in enumerate(symbols):
                    sym.label = f'symbol_{i}'
            
            if return_intermediate:
                result['symbols'] = [
                    {
                        'label': s.label,
                        'confidence': s.confidence,
                        'bbox': s.bbox.to_tuple(),
                        'candidates': s.candidates
                    }
                    for s in symbols
                ]
            
            # 步骤4：结构分析
            logger.info("步骤4: 结构分析")
            syntax_tree, latex = self.structure_analyzer.analyze(symbols)
            result['latex'] = latex
            result['syntax_tree'] = syntax_tree.to_dict() if syntax_tree else None
            
            # 步骤5：语义理解
            logger.info("步骤5: 语义理解")
            semantic_result = self.semantic_processor.process(latex, syntax_tree)
            result['semantic'] = semantic_result.to_dict()
            
            result['success'] = True
            logger.info(f"识别完成: {latex}")
            
        except Exception as e:
            logger.error(f"识别过程出错: {e}")
            result['errors'].append(str(e))
        
        return result
    
    def recognize_file(self, image_path: str, 
                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        识别图像文件中的公式
        
        Args:
            image_path: 图像文件路径
            output_path: 输出可视化结果的路径（可选）
            
        Returns:
            识别结果字典
        """
        # 加载图像
        image = load_image(image_path, grayscale=True)
        
        # 识别
        result = self.recognize(image, return_intermediate=True)
        
        # 保存可视化结果
        if output_path and 'preprocessing' in result:
            binary = result['preprocessing']['cleaned']
            symbols = self.segmenter.segment(binary)
            vis = visualize_symbols(binary, symbols, show_labels=True)
            save_image(vis, output_path)
            logger.info(f"可视化结果已保存: {output_path}")
        
        return result
    
    def train_recognizer(self, images: list, labels: list,
                         classifier_type: str = 'svm',
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        训练符号识别器
        
        Args:
            images: 符号图像列表
            labels: 标签列表
            classifier_type: 分类器类型
            save_path: 模型保存路径
            
        Returns:
            训练报告
        """
        report = self.recognizer.train(images, labels, classifier_type)
        
        if save_path:
            self.recognizer.save_model(save_path)
        
        return report


def create_demo_image() -> np.ndarray:
    """
    创建演示用的手写公式图像
    
    Returns:
        模拟的手写公式图像
    """
    # 创建白色背景
    img = np.ones((200, 600), dtype=np.uint8) * 255
    
    # 添加简单的文字（模拟手写）
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 绘制 "x^2 + y = 5"
    cv2.putText(img, 'x', (50, 120), font, 2, 0, 3)
    cv2.putText(img, '2', (110, 80), font, 1, 0, 2)  # 上标
    cv2.putText(img, '+', (160, 120), font, 2, 0, 3)
    cv2.putText(img, 'y', (230, 120), font, 2, 0, 3)
    cv2.putText(img, '=', (310, 120), font, 2, 0, 3)
    cv2.putText(img, '5', (390, 120), font, 2, 0, 3)
    
    # 添加一些噪声使其更像手写
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def demo():
    """运行演示"""
    print("=" * 60)
    print("手写数学公式识别与语义理解系统 - 演示")
    print("=" * 60)
    
    # 创建演示图像
    print("\n创建演示图像...")
    demo_img = create_demo_image()
    
    # 保存演示图像
    demo_path = os.path.join(os.path.dirname(__file__), 'demo_input.png')
    cv2.imwrite(demo_path, demo_img)
    print(f"演示图像已保存: {demo_path}")
    
    # 初始化系统
    print("\n初始化识别系统...")
    system = FormulaRecognitionSystem()
    
    # 演示预处理
    print("\n--- 演示: 图像预处理 ---")
    preprocessor = ImagePreprocessor()
    binary = preprocessor.process(demo_img)
    
    binary_path = os.path.join(os.path.dirname(__file__), 'demo_binary.png')
    cv2.imwrite(binary_path, binary)
    print(f"二值化结果已保存: {binary_path}")
    
    # 演示符号分割
    print("\n--- 演示: 符号分割 ---")
    segmenter = SymbolSegmenter()
    symbols = segmenter.segment(binary)
    print(f"检测到 {len(symbols)} 个符号")
    
    for i, sym in enumerate(symbols):
        print(f"  符号 {i+1}: 位置 ({sym.bbox.x}, {sym.bbox.y}), "
              f"大小 {sym.bbox.width}x{sym.bbox.height}")
    
    # 可视化分割结果
    vis = visualize_symbols(binary, symbols, show_labels=False)
    vis_path = os.path.join(os.path.dirname(__file__), 'demo_segmented.png')
    cv2.imwrite(vis_path, vis)
    print(f"分割结果已保存: {vis_path}")
    
    # 演示特征提取
    print("\n--- 演示: 特征提取 ---")
    extractor = FeatureExtractor()
    if symbols:
        features = extractor.extract(symbols[0].image)
        print(f"特征维度: {len(features)}")
        print(f"特征前10维: {features[:10]}")
    
    # 演示语义理解
    print("\n--- 演示: 语义理解 ---")
    semantic = SemanticProcessor()
    
    # 使用已知的 LaTeX 测试
    test_cases = [
        r"x^2 + y = 5",
        r"x^2 - 4 = 0",
        r"\frac{x+1}{x-1}",
        r"\sin(x) + \cos(x)",
    ]
    
    for latex in test_cases:
        print(f"\n输入: {latex}")
        result = semantic.process(latex)
        print(f"  类型: {result.formula_type.value}")
        print(f"  变量: {result.variables}")
        print(f"  解释: {result.interpretation}")
        if result.solution:
            print(f"  计算结果: {result.solution}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='手写数学公式识别与语义理解系统'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='输入图像路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出可视化结果路径'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='符号识别模型路径'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行演示'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        demo()
        return
    
    if not args.image:
        parser.print_help()
        print("\n请提供输入图像路径，或使用 --demo 运行演示")
        return
    
    # 初始化系统
    system = FormulaRecognitionSystem(model_path=args.model)
    
    # 识别公式
    result = system.recognize_file(args.image, args.output)
    
    # 输出结果
    print("\n" + "=" * 40)
    print("识别结果")
    print("=" * 40)
    
    if result['success']:
        print(f"LaTeX: {result['latex']}")
        
        if result.get('semantic'):
            semantic = result['semantic']
            print(f"公式类型: {semantic.get('formula_type', 'unknown')}")
            print(f"变量: {semantic.get('variables', [])}")
            print(f"解释: {semantic.get('interpretation', '')}")
            
            if semantic.get('solution'):
                print(f"计算结果: {semantic['solution']}")
    else:
        print("识别失败")
        for error in result.get('errors', []):
            print(f"错误: {error}")
    
    for warning in result.get('warnings', []):
        print(f"警告: {warning}")


if __name__ == '__main__':
    main()
