"""
图像预处理模块
==============

实现以下功能：
- 自适应二值化（Sauvola 方法）
- 图像去噪
- 倾斜校正
- 笔画细化（骨架提取）
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import logging

from .config import PreprocessingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    图像预处理器
    
    将输入的手写公式图像转换为干净的二值图像，
    便于后续的符号分割和识别。
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置，如果为 None 则使用默认配置
        """
        self.config = config or DEFAULT_CONFIG.preprocessing
    
    def process(self, image: np.ndarray, 
                return_intermediate: bool = False) -> np.ndarray:
        """
        完整的预处理流程
        
        Args:
            image: 输入图像（灰度或彩色）
            return_intermediate: 是否返回中间结果
            
        Returns:
            预处理后的二值图像，如果 return_intermediate=True，
            返回包含中间结果的字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        logger.info(f"输入图像尺寸: {gray.shape}")
        
        # 步骤1：二值化（根据图像尺寸选择合适的方法）
        min_dim = min(gray.shape)
        if min_dim <= 64:
            # 小图像使用Otsu或简单阈值
            binary = self.otsu_binarization(gray)
            logger.info("完成Otsu二值化（小图像）")
        else:
            # 大图像使用自适应二值化
            binary = self.sauvola_binarization(gray)
            logger.info("完成自适应二值化")
        
        # 步骤2：去噪
        denoised = self.denoise(binary)
        logger.info("完成去噪处理")
        
        # 步骤3：倾斜校正
        corrected, angle = self.correct_skew(denoised)
        logger.info(f"完成倾斜校正，校正角度: {angle:.2f}°")
        
        # 步骤4：去除边界噪声
        cleaned = self.remove_border_noise(corrected)
        
        if return_intermediate:
            # 步骤5（可选）：骨架提取
            skeleton = self.extract_skeleton(cleaned)
            
            return {
                'gray': gray,
                'binary': binary,
                'denoised': denoised,
                'corrected': corrected,
                'cleaned': cleaned,
                'skeleton': skeleton,
                'skew_angle': angle
            }
        
        return cleaned
    
    def sauvola_binarization(self, gray: np.ndarray) -> np.ndarray:
        """
        Sauvola 自适应二值化
        
        Sauvola 方法的阈值公式：
        T(x,y) = μ(x,y) * [1 + k * (σ(x,y)/R - 1)]
        
        其中：
        - μ(x,y): 局部窗口的均值
        - σ(x,y): 局部窗口的标准差
        - k: 控制阈值的参数（通常 0.2-0.5）
        - R: 标准差的动态范围（通常 128）
        
        Args:
            gray: 灰度图像
            
        Returns:
            二值图像（前景为白色 255，背景为黑色 0）
        """
        window_size = self.config.sauvola_window_size
        k = self.config.sauvola_k
        R = self.config.sauvola_r
        
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
        
        # 计算局部均值
        mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
        
        # 计算局部方差和标准差
        mean_sq = cv2.blur(gray.astype(np.float64) ** 2, (window_size, window_size))
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)  # 避免数值误差导致的负值
        std = np.sqrt(variance)
        
        # 计算 Sauvola 阈值
        threshold = mean * (1 + k * (std / R - 1))
        
        # 应用阈值
        binary = np.zeros_like(gray)
        binary[gray > threshold] = 255
        
        # 自动检测前景/背景：检查边界像素判断背景颜色
        # 如果边界主要是白色(>200)，说明是白底黑字，需要反转
        # 如果边界主要是黑色(<50)，说明是黑底白字，不需要反转
        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :],  # 上下边界
            gray[:, 0], gray[:, -1]   # 左右边界
        ])
        border_mean = np.mean(border_pixels)
        
        # 如果边界平均值高（白色背景），反转图像
        if border_mean > 127:
            binary = 255 - binary
        
        return binary.astype(np.uint8)
    
    def niblack_binarization(self, gray: np.ndarray, 
                             k: float = -0.2) -> np.ndarray:
        """
        Niblack 自适应二值化（备选方法）
        
        阈值公式：T(x,y) = μ(x,y) + k * σ(x,y)
        
        Args:
            gray: 灰度图像
            k: Niblack 参数（通常为负值）
            
        Returns:
            二值图像
        """
        window_size = self.config.sauvola_window_size
        
        if window_size % 2 == 0:
            window_size += 1
        
        # 计算局部均值
        mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
        
        # 计算局部标准差
        mean_sq = cv2.blur(gray.astype(np.float64) ** 2, (window_size, window_size))
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)
        
        # 计算阈值
        threshold = mean + k * std
        
        # 应用阈值
        binary = np.zeros_like(gray)
        binary[gray > threshold] = 255
        binary = 255 - binary
        
        return binary.astype(np.uint8)
    
    def otsu_binarization(self, gray: np.ndarray) -> np.ndarray:
        """
        Otsu 自动阈值二值化（备选方法）
        
        适用于双峰直方图的图像。
        输出格式：黑底(0)白字(255)，白色是前景/笔画，与OpenCV连通域分析兼容。
        
        Args:
            gray: 灰度图像
            
        Returns:
            二值图像（前景为白色255，背景为黑色0）
        """
        # 高斯模糊减少噪声（对小图像使用更小的核）
        kernel_size = 3 if min(gray.shape) <= 64 else 5
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Otsu 阈值
        _, binary = cv2.threshold(blurred, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 自动检测前景/背景：根据原图边界像素判断背景颜色
        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :],  # 上下边界
            gray[:, 0], gray[:, -1]   # 左右边界
        ])
        border_mean = np.mean(border_pixels)
        
        # 输出格式：黑底白字（白色255是前景）
        # 如果输入是白底黑字（border_mean > 127），反转使前景为白色
        # 如果输入是黑底白字（border_mean <= 127），保持不变
        if border_mean > 127:
            # 白底黑字，反转成黑底白字
            binary = 255 - binary
        
        return binary
    
    def denoise(self, binary: np.ndarray) -> np.ndarray:
        """
        去除噪声
        
        使用形态学操作和连通域分析去除噪声点。
        
        对于已经是干净的二值图像（如训练数据拼接），跳过形态学开运算
        以避免损失边缘像素。
        
        Args:
            binary: 二值图像
            
        Returns:
            去噪后的图像
        """
        kernel_size = self.config.denoise_kernel_size
        min_area = self.config.min_component_area
        
        # 检测图像是否已经是干净的二值图像（只有0和255两个值）
        unique_vals = np.unique(binary)
        is_clean_binary = len(unique_vals) == 2 and set(unique_vals) == {0, 255}
        
        if is_clean_binary:
            # 对于干净的二值图像，只去除非常小的噪点，不做形态学开运算
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            denoised = np.zeros_like(binary)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    denoised[labels == i] = 255
            return denoised
        
        # 对于非干净图像，使用完整的去噪流程
        # 形态学开运算（去除小噪点）
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 连通域分析，去除过小的区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            opened, connectivity=8
        )
        
        # 创建输出图像
        denoised = np.zeros_like(opened)
        
        for i in range(1, num_labels):  # 跳过背景（标签 0）
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                denoised[labels == i] = 255
        
        return denoised
    
    def correct_skew(self, binary: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        检测并校正图像倾斜
        
        使用霍夫变换检测主要直线方向，计算倾斜角度后进行仿射变换。
        
        Args:
            binary: 二值图像
            
        Returns:
            校正后的图像和检测到的倾斜角度
        """
        # 使用 Canny 边缘检测
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(
            edges, 
            rho=1, 
            theta=np.pi / 180, 
            threshold=self.config.hough_threshold
        )
        
        if lines is None or len(lines) == 0:
            logger.info("未检测到明显倾斜")
            return binary, 0.0
        
        # 计算所有检测到的角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            # 转换为度数
            angle = np.degrees(theta) - 90
            # 只考虑小角度倾斜
            if abs(angle) < self.config.max_skew_angle:
                angles.append(angle)
        
        if not angles:
            return binary, 0.0
        
        # 使用中位数角度（更鲁棒）
        skew_angle = np.median(angles)
        
        if abs(skew_angle) < 0.5:  # 忽略非常小的倾斜
            return binary, 0.0
        
        # 执行旋转校正
        h, w = binary.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        
        # 计算新的边界框大小
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # 调整旋转矩阵
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        corrected = cv2.warpAffine(
            binary, 
            rotation_matrix, 
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return corrected, skew_angle
    
    def correct_skew_projection(self, binary: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用投影法校正倾斜（备选方法）
        
        通过最小化水平投影直方图的熵来找到最佳旋转角度。
        
        Args:
            binary: 二值图像
            
        Returns:
            校正后的图像和检测到的倾斜角度
        """
        best_angle = 0
        best_score = 0
        
        # 在 [-15, 15] 度范围内搜索
        for angle in np.arange(-15, 15, 0.5):
            h, w = binary.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (w, h))
            
            # 计算水平投影
            projection = np.sum(rotated, axis=1)
            
            # 计算投影的方差（方差越大，文本行越分明）
            score = np.var(projection)
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        if abs(best_angle) < 0.5:
            return binary, 0.0
        
        h, w = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(binary, M, (w, h))
        
        return corrected, best_angle
    
    def remove_border_noise(self, binary: np.ndarray, 
                            border_ratio: float = 0.02) -> np.ndarray:
        """
        去除边界噪声
        
        Args:
            binary: 二值图像
            border_ratio: 边界区域比例
            
        Returns:
            处理后的图像
        """
        h, w = binary.shape
        border_h = int(h * border_ratio)
        border_w = int(w * border_ratio)
        
        result = binary.copy()
        
        # 清除边界区域（仅当边界大小>0时才清除）
        if border_h > 0:
            result[:border_h, :] = 0
            result[-border_h:, :] = 0
        if border_w > 0:
            result[:, :border_w] = 0
            result[:, -border_w:] = 0
        
        return result
    
    def extract_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """
        提取骨架（细化）
        
        使用 Zhang-Suen 细化算法提取单像素宽度的骨架。
        
        Args:
            binary: 二值图像
            
        Returns:
            骨架图像
        """
        # 使用 skimage 的骨架化（如果可用）或自实现
        try:
            from skimage.morphology import skeletonize
            # 转换为布尔数组
            bool_img = binary > 0
            skeleton = skeletonize(bool_img)
            return (skeleton * 255).astype(np.uint8)
        except ImportError:
            # 使用自实现的 Zhang-Suen 算法
            return self._zhang_suen_thinning(binary)
    
    def _zhang_suen_thinning(self, binary: np.ndarray) -> np.ndarray:
        """
        Zhang-Suen 细化算法
        
        迭代剥离边缘像素直到收敛。
        
        Args:
            binary: 二值图像
            
        Returns:
            细化后的图像
        """
        # 将图像转换为 0/1 表示
        img = (binary > 0).astype(np.uint8)
        
        def get_neighbors(img: np.ndarray, r: int, c: int) -> list:
            """获取 8 邻域像素值"""
            return [
                img[r-1, c],   # P2
                img[r-1, c+1], # P3
                img[r, c+1],   # P4
                img[r+1, c+1], # P5
                img[r+1, c],   # P6
                img[r+1, c-1], # P7
                img[r, c-1],   # P8
                img[r-1, c-1]  # P9
            ]
        
        def transitions(neighbors: list) -> int:
            """计算 0-1 跃迁次数"""
            n = neighbors + [neighbors[0]]
            return sum((n[i], n[i+1]) == (0, 1) for i in range(8))
        
        def black_neighbors(neighbors: list) -> int:
            """计算黑色邻居数量"""
            return sum(neighbors)
        
        changed = True
        while changed:
            changed = False
            
            # 第一次迭代
            to_remove = []
            for r in range(1, img.shape[0] - 1):
                for c in range(1, img.shape[1] - 1):
                    if img[r, c] != 1:
                        continue
                    
                    neighbors = get_neighbors(img, r, c)
                    B = black_neighbors(neighbors)
                    A = transitions(neighbors)
                    
                    P2, P3, P4, P5, P6, P7, P8, P9 = neighbors
                    
                    if (2 <= B <= 6 and A == 1 and
                        P2 * P4 * P6 == 0 and
                        P4 * P6 * P8 == 0):
                        to_remove.append((r, c))
            
            for r, c in to_remove:
                img[r, c] = 0
                changed = True
            
            # 第二次迭代
            to_remove = []
            for r in range(1, img.shape[0] - 1):
                for c in range(1, img.shape[1] - 1):
                    if img[r, c] != 1:
                        continue
                    
                    neighbors = get_neighbors(img, r, c)
                    B = black_neighbors(neighbors)
                    A = transitions(neighbors)
                    
                    P2, P3, P4, P5, P6, P7, P8, P9 = neighbors
                    
                    if (2 <= B <= 6 and A == 1 and
                        P2 * P4 * P8 == 0 and
                        P2 * P6 * P8 == 0):
                        to_remove.append((r, c))
            
            for r, c in to_remove:
                img[r, c] = 0
                changed = True
        
        return (img * 255).astype(np.uint8)
    
    def normalize_stroke_width(self, binary: np.ndarray, 
                               target_width: int = 3) -> np.ndarray:
        """
        标准化笔画宽度
        
        通过腐蚀和膨胀操作使笔画宽度一致。
        
        Args:
            binary: 二值图像
            target_width: 目标笔画宽度
            
        Returns:
            标准化后的图像
        """
        # 估计当前笔画宽度（使用距离变换）
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        current_width = np.median(dist[dist > 0]) * 2 if np.any(dist > 0) else target_width
        
        if current_width > target_width:
            # 需要细化
            iterations = int((current_width - target_width) / 2)
            kernel = np.ones((3, 3), np.uint8)
            result = cv2.erode(binary, kernel, iterations=iterations)
        else:
            # 需要加粗
            iterations = int((target_width - current_width) / 2)
            kernel = np.ones((3, 3), np.uint8)
            result = cv2.dilate(binary, kernel, iterations=iterations)
        
        return result
    
    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        增强对比度
        
        使用 CLAHE（限制对比度自适应直方图均衡化）。
        
        Args:
            gray: 灰度图像
            
        Returns:
            增强后的图像
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)


# 便捷函数
def preprocess_image(image: np.ndarray, 
                     config: Optional[PreprocessingConfig] = None) -> np.ndarray:
    """
    预处理图像的便捷函数
    
    Args:
        image: 输入图像
        config: 预处理配置
        
    Returns:
        预处理后的二值图像
    """
    preprocessor = ImagePreprocessor(config)
    return preprocessor.process(image)
