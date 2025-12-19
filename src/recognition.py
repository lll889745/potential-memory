"""
符号识别模块
============

实现以下功能：
- 特征提取（几何特征、结构特征、HOG 特征等）
- 分类器训练和预测
- 易混淆符号消解
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import pickle
import os

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .config import RecognitionConfig, DEFAULT_CONFIG, SYMBOL_CATEGORIES, CONFUSABLE_SYMBOLS
from .utils import Symbol, BoundingBox, pad_image, calculate_hu_moments

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    特征提取器
    
    从符号图像中提取多种特征用于分类。
    """
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        """
        初始化特征提取器
        
        Args:
            config: 识别配置
        """
        self.config = config or DEFAULT_CONFIG.recognition
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取所有特征
        
        Args:
            image: 符号图像（二值）
            
        Returns:
            特征向量
        """
        # 标准化图像大小
        normalized = self._normalize_image(image)
        
        features = []
        
        # 1. 几何特征
        geo_features = self._extract_geometric_features(normalized)
        features.extend(geo_features)
        
        # 2. 结构特征
        struct_features = self._extract_structural_features(normalized)
        features.extend(struct_features)
        
        # 3. Hu 不变矩
        hu_features = self._extract_hu_moments(normalized)
        features.extend(hu_features)
        
        # 4. HOG 特征
        hog_features = self._extract_hog_features(normalized)
        features.extend(hog_features)
        
        # 5. 网格特征
        grid_features = self._extract_grid_features(normalized)
        features.extend(grid_features)
        
        # 6. 投影特征
        proj_features = self._extract_projection_features(normalized)
        features.extend(proj_features)
        
        # 7. 轮廓特征
        contour_features = self._extract_contour_features(normalized)
        features.extend(contour_features)
        
        return np.array(features, dtype=np.float32)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        标准化图像到固定大小，并统一格式
        
        预处理输出格式：黑底(0)白字(255)，白色是前景（与OpenCV连通域分析兼容）
        训练数据格式：白底(255)黑字(0)，黑色是前景
        
        通过检测图像格式，自动转换为训练数据格式。
        
        Args:
            image: 输入图像
            
        Returns:
            标准化后的图像（统一为白底黑字格式）
        """
        target_size = self.config.symbol_size
        
        # 首先填充到目标大小
        padded = pad_image(image, target_size, pad_value=0)
        
        # 检测当前图像格式：通过边界像素判断背景色
        border_pixels = np.concatenate([
            padded[0, :], padded[-1, :],
            padded[:, 0], padded[:, -1]
        ])
        border_mean = np.mean(border_pixels)
        
        # 如果边界是黑色（<127），说明是黑底白字，需要反转
        if border_mean < 127:
            padded = 255 - padded
        
        return padded
    
    def _extract_geometric_features(self, image: np.ndarray) -> List[float]:
        """
        提取几何特征
        
        包括：
        - 宽高比
        - 填充率
        - 重心位置
        - 欧拉数
        
        Args:
            image: 标准化后的图像
            
        Returns:
            几何特征列表
        """
        h, w = image.shape
        
        # 像素统计
        foreground = np.sum(image > 0)
        total = h * w
        
        # 填充率
        fill_ratio = foreground / max(total, 1)
        
        # 计算矩
        moments = cv2.moments(image)
        
        # 重心位置（归一化到 [0, 1]）
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00'] / w
            cy = moments['m01'] / moments['m00'] / h
        else:
            cx, cy = 0.5, 0.5
        
        # 计算轮廓
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 欧拉数 = 连通区域数 - 孔洞数
        if hierarchy is not None:
            # 顶级轮廓数（连通区域）
            num_regions = np.sum(hierarchy[0, :, 3] == -1)
            # 孔洞数（有父轮廓的轮廓）
            num_holes = np.sum(hierarchy[0, :, 3] != -1)
            euler_number = num_regions - num_holes
        else:
            euler_number = 1
        
        # 计算外接矩形的宽高比
        if contours:
            all_points = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_points)
            aspect_ratio = bw / max(bh, 1)
        else:
            aspect_ratio = 1.0
        
        # 计算最小外接矩形的角度
        if contours and len(contours[0]) >= 5:
            rect = cv2.minAreaRect(np.vstack(contours))
            angle = rect[2] / 90.0  # 归一化到 [-1, 1]
        else:
            angle = 0.0
        
        return [fill_ratio, cx, cy, euler_number / 10.0, aspect_ratio, angle]
    
    def _extract_structural_features(self, image: np.ndarray) -> List[float]:
        """
        提取结构特征
        
        包括：
        - 端点数
        - 交叉点数
        - 环数
        - 水平/垂直穿越数
        
        Args:
            image: 标准化后的图像
            
        Returns:
            结构特征列表
        """
        # 骨架化
        try:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(image > 0).astype(np.uint8) * 255
        except ImportError:
            # 使用形态学细化
            kernel = np.ones((3, 3), np.uint8)
            skeleton = cv2.erode(image, kernel, iterations=1)
        
        # 端点和交叉点检测
        endpoints, crossings = self._detect_junction_points(skeleton)
        
        # 环数（使用欧拉数估计）
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        num_loops = 0
        if hierarchy is not None:
            num_loops = np.sum(hierarchy[0, :, 3] != -1)
        
        # 水平穿越数
        h_crossings = self._count_crossings(image, axis=0)
        
        # 垂直穿越数
        v_crossings = self._count_crossings(image, axis=1)
        
        # 归一化
        return [
            endpoints / 10.0,
            crossings / 5.0,
            num_loops / 3.0,
            h_crossings / 10.0,
            v_crossings / 10.0
        ]
    
    def _detect_junction_points(self, skeleton: np.ndarray) -> Tuple[int, int]:
        """
        检测骨架的端点和交叉点
        
        Args:
            skeleton: 骨架图像
            
        Returns:
            (端点数, 交叉点数)
        """
        # 定义端点和交叉点的模板
        endpoints = 0
        crossings = 0
        
        # 填充边界
        padded = np.pad(skeleton, 1, mode='constant', constant_values=0)
        
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                if padded[i, j] == 0:
                    continue
                
                # 8 邻域
                neighbors = [
                    padded[i-1, j], padded[i-1, j+1],
                    padded[i, j+1], padded[i+1, j+1],
                    padded[i+1, j], padded[i+1, j-1],
                    padded[i, j-1], padded[i-1, j-1]
                ]
                
                # 计算邻居数
                num_neighbors = sum(1 for n in neighbors if n > 0)
                
                # 计算连通性（0-1 跃迁数）
                transitions = 0
                for k in range(8):
                    if neighbors[k] == 0 and neighbors[(k+1) % 8] > 0:
                        transitions += 1
                
                if num_neighbors == 1:
                    endpoints += 1
                elif num_neighbors >= 3 and transitions >= 3:
                    crossings += 1
        
        return endpoints, crossings
    
    def _count_crossings(self, image: np.ndarray, axis: int) -> float:
        """
        计算穿越数
        
        Args:
            image: 输入图像
            axis: 轴向（0=水平，1=垂直）
            
        Returns:
            平均穿越数
        """
        if axis == 0:
            # 水平穿越：沿每一行计算
            lines = [image[i, :] for i in range(image.shape[0])]
        else:
            # 垂直穿越：沿每一列计算
            lines = [image[:, j] for j in range(image.shape[1])]
        
        total_crossings = 0
        for line in lines:
            # 计算 0->1 和 1->0 的跃迁
            binary = (line > 0).astype(int)
            crossings = np.sum(np.abs(np.diff(binary)))
            total_crossings += crossings
        
        return total_crossings / max(len(lines), 1)
    
    def _extract_hu_moments(self, image: np.ndarray) -> List[float]:
        """
        提取 Hu 不变矩
        
        Hu 矩具有平移、旋转和缩放不变性。
        
        Args:
            image: 输入图像
            
        Returns:
            Hu 矩特征（7 维）
        """
        hu_moments = calculate_hu_moments(image)
        return hu_moments.tolist()
    
    def _extract_hog_features(self, image: np.ndarray) -> List[float]:
        """
        提取 HOG（方向梯度直方图）特征
        
        Args:
            image: 输入图像
            
        Returns:
            HOG 特征
        """
        # 简化的 HOG 实现
        orientations = self.config.hog_orientations
        pixels_per_cell = self.config.hog_pixels_per_cell
        cells_per_block = self.config.hog_cells_per_block
        
        # 计算梯度
        gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180
        
        h, w = image.shape
        cell_h, cell_w = pixels_per_cell
        n_cells_y = h // cell_h
        n_cells_x = w // cell_w
        
        # 计算每个 cell 的直方图
        histograms = np.zeros((n_cells_y, n_cells_x, orientations))
        
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ang = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                
                # 构建直方图
                for m, a in zip(cell_mag.flatten(), cell_ang.flatten()):
                    bin_idx = int(a / 180 * orientations) % orientations
                    histograms[i, j, bin_idx] += m
        
        # 块归一化
        block_h, block_w = cells_per_block
        hog_features = []
        
        for i in range(n_cells_y - block_h + 1):
            for j in range(n_cells_x - block_w + 1):
                block = histograms[i:i+block_h, j:j+block_w, :].flatten()
                norm = np.sqrt(np.sum(block**2) + 1e-6)
                hog_features.extend(block / norm)
        
        # 如果特征太少，填充
        target_len = 36  # 预期的 HOG 特征长度
        if len(hog_features) < target_len:
            hog_features.extend([0.0] * (target_len - len(hog_features)))
        elif len(hog_features) > target_len:
            hog_features = hog_features[:target_len]
        
        return hog_features
    
    def _extract_grid_features(self, image: np.ndarray) -> List[float]:
        """
        提取网格特征
        
        将图像划分为网格，计算每个格子的像素密度。
        
        Args:
            image: 输入图像
            
        Returns:
            网格特征
        """
        grid_h, grid_w = self.config.grid_size
        h, w = image.shape
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        features = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                density = np.sum(cell > 0) / max(cell.size, 1)
                features.append(density)
        
        return features
    
    def _extract_projection_features(self, image: np.ndarray) -> List[float]:
        """
        提取投影特征
        
        计算水平和垂直投影的统计特征。
        
        Args:
            image: 输入图像
            
        Returns:
            投影特征
        """
        # 水平投影
        h_proj = np.sum(image > 0, axis=1).astype(np.float32)
        h_proj = h_proj / max(np.max(h_proj), 1)
        
        # 垂直投影
        v_proj = np.sum(image > 0, axis=0).astype(np.float32)
        v_proj = v_proj / max(np.max(v_proj), 1)
        
        # 采样投影（固定维度）
        n_samples = 8
        h_samples = np.interp(
            np.linspace(0, len(h_proj) - 1, n_samples),
            np.arange(len(h_proj)),
            h_proj
        )
        v_samples = np.interp(
            np.linspace(0, len(v_proj) - 1, n_samples),
            np.arange(len(v_proj)),
            v_proj
        )
        
        features = list(h_samples) + list(v_samples)
        
        # 投影统计特征
        features.append(np.mean(h_proj))
        features.append(np.std(h_proj))
        features.append(np.mean(v_proj))
        features.append(np.std(v_proj))
        
        return features
    
    def _extract_contour_features(self, image: np.ndarray) -> List[float]:
        """
        提取轮廓特征
        
        Args:
            image: 输入图像
            
        Returns:
            轮廓特征
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return [0.0] * 5
        
        # 使用最大轮廓
        largest = max(contours, key=cv2.contourArea)
        
        # 轮廓面积
        area = cv2.contourArea(largest)
        
        # 轮廓周长
        perimeter = cv2.arcLength(largest, True)
        
        # 圆形度
        circularity = 4 * np.pi * area / max(perimeter**2, 1)
        
        # 凸包
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        
        # 凸度
        convexity = area / max(hull_area, 1)
        
        # 矩形度
        x, y, w, h = cv2.boundingRect(largest)
        rectangularity = area / max(w * h, 1)
        
        return [
            area / max(image.size, 1),
            perimeter / max(image.shape[0] + image.shape[1], 1),
            circularity,
            convexity,
            rectangularity
        ]


class SymbolRecognizer:
    """
    符号识别器
    
    使用机器学习分类器识别符号类别。
    """
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        """
        初始化识别器
        
        Args:
            config: 识别配置
        """
        self.config = config or DEFAULT_CONFIG.recognition
        self.feature_extractor = FeatureExtractor(config)
        
        # 分类器
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.reverse_encoder = {}
        
        # 是否已训练
        self.is_trained = False
    
    def train(self, images: List[np.ndarray], labels: List[str],
              classifier_type: str = 'svm') -> Dict[str, Any]:
        """
        训练分类器
        
        Args:
            images: 符号图像列表
            labels: 标签列表
            classifier_type: 分类器类型 ('svm', 'rf', 'knn')
            
        Returns:
            训练结果报告
        """
        logger.info(f"开始训练，样本数: {len(images)}")
        
        # 构建标签编码器
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.reverse_encoder = {i: label for label, i in self.label_encoder.items()}
        
        # 提取特征
        logger.info("提取特征...")
        features = []
        for img in images:
            feat = self.feature_extractor.extract(img)
            features.append(feat)
        
        X = np.array(features)
        y = np.array([self.label_encoder[label] for label in labels])
        
        # 标准化
        X = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 创建分类器
        if classifier_type == 'svm':
            self.classifier = SVC(
                C=self.config.svm_c,
                gamma=self.config.svm_gamma,
                kernel='rbf',
                probability=True
            )
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                random_state=42
            )
        elif classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(
                n_neighbors=self.config.knn_n_neighbors
            )
        else:
            raise ValueError(f"未知的分类器类型: {classifier_type}")
        
        # 训练
        logger.info(f"训练 {classifier_type} 分类器...")
        self.classifier.fit(X_train, y_train)
        
        # 评估
        y_pred = self.classifier.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=unique_labels,
            output_dict=True
        )
        
        self.is_trained = True
        logger.info(f"训练完成，测试准确率: {report['accuracy']:.4f}")
        
        return report
    
    def predict(self, image: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        预测单个符号
        
        Args:
            image: 符号图像
            
        Returns:
            (预测标签, 置信度, top-k 候选列表)
        """
        if not self.is_trained:
            raise RuntimeError("分类器尚未训练")
        
        # 提取特征
        features = self.feature_extractor.extract(image)
        X = self.scaler.transform([features])
        
        # 预测
        pred_idx = self.classifier.predict(X)[0]
        pred_label = self.reverse_encoder[pred_idx]
        
        # 获取概率（如果支持）
        if hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(X)[0]
            confidence = probs[pred_idx]
            
            # 获取 top-k 候选
            top_k = self.config.top_k_candidates
            top_indices = np.argsort(probs)[-top_k:][::-1]
            candidates = [
                (self.reverse_encoder[idx], probs[idx])
                for idx in top_indices
            ]
        else:
            confidence = 1.0
            candidates = [(pred_label, 1.0)]
        
        return pred_label, confidence, candidates
    
    def recognize_symbols(self, symbols: List[Symbol]) -> List[Symbol]:
        """
        识别符号列表
        
        Args:
            symbols: 待识别的符号列表
            
        Returns:
            识别后的符号列表
        """
        for symbol in symbols:
            # 跳过特殊符号（已标记的分数线等）
            if symbol.is_fraction_line:
                symbol.label = 'fraction_line'
                symbol.confidence = 1.0
                continue
            
            if symbol.is_sqrt_symbol:
                symbol.label = 'sqrt'
                symbol.confidence = 1.0
                continue
            
            # 识别
            try:
                label, conf, candidates = self.predict(symbol.image)
                symbol.label = label
                symbol.confidence = conf
                symbol.candidates = candidates
            except Exception as e:
                logger.warning(f"符号识别失败: {e}")
                symbol.label = 'unknown'
                symbol.confidence = 0.0
        
        return symbols
    
    def resolve_ambiguity(self, symbol: Symbol, context: Dict[str, Any]) -> str:
        """
        消解符号歧义
        
        根据上下文信息解决易混淆符号的识别。
        
        Args:
            symbol: 符号
            context: 上下文信息
            
        Returns:
            最终标签
        """
        if symbol.confidence > 0.9:
            return symbol.label
        
        candidates = symbol.candidates
        if len(candidates) < 2:
            return symbol.label
        
        top1, top2 = candidates[0], candidates[1]
        
        # 检查是否是混淆对
        confusable = None
        for pair in CONFUSABLE_SYMBOLS:
            if {top1[0], top2[0]} == {pair[0], pair[1]}:
                confusable = pair
                break
        
        if confusable is None:
            return symbol.label
        
        # 根据上下文消解
        label1, label2 = confusable
        
        # 0/O 混淆
        if {label1, label2} == {'0', 'O'} or {label1, label2} == {'0', 'o'}:
            # 如果周围是数字，更可能是 0
            if context.get('neighbors_are_digits', False):
                return '0'
            # 如果周围是字母，更可能是 O
            if context.get('neighbors_are_letters', False):
                return 'O'
        
        # 1/l 混淆
        if {label1, label2} == {'1', 'l'} or {label1, label2} == {'1', 'I'}:
            if context.get('neighbors_are_digits', False):
                return '1'
            if context.get('in_variable_position', False):
                return 'l'
        
        # x/× 混淆
        if {label1, label2} == {'x', 'times'}:
            if context.get('between_numbers', False):
                return 'times'
            return 'x'  # 默认为变量 x
        
        return symbol.label
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'reverse_encoder': self.reverse_encoder,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.reverse_encoder = model_data['reverse_encoder']
        self.config = model_data.get('config', self.config)
        self.is_trained = True
        
        logger.info(f"模型已加载: {path}")


class TemplateMatchRecognizer:
    """
    模板匹配识别器（备选方案）
    
    使用模板匹配进行符号识别，适用于少量类别或作为验证。
    """
    
    def __init__(self):
        """初始化模板匹配识别器"""
        self.templates: Dict[str, List[np.ndarray]] = {}
    
    def add_template(self, label: str, image: np.ndarray) -> None:
        """
        添加模板
        
        Args:
            label: 标签
            image: 模板图像
        """
        if label not in self.templates:
            self.templates[label] = []
        
        # 标准化大小
        normalized = pad_image(image, (32, 32), pad_value=0)
        self.templates[label].append(normalized)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        使用模板匹配预测
        
        Args:
            image: 输入图像
            
        Returns:
            (预测标签, 匹配分数)
        """
        # 标准化输入
        normalized = pad_image(image, (32, 32), pad_value=0)
        
        best_label = None
        best_score = -1
        
        for label, templates in self.templates.items():
            for template in templates:
                # 使用归一化相关系数匹配
                result = cv2.matchTemplate(
                    normalized.astype(np.float32),
                    template.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED
                )
                score = result[0, 0]
                
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return best_label, best_score


def create_recognizer(model_path: Optional[str] = None,
                      config: Optional[RecognitionConfig] = None) -> SymbolRecognizer:
    """
    创建符号识别器
    
    Args:
        model_path: 预训练模型路径（可选）
        config: 识别配置
        
    Returns:
        符号识别器实例
    """
    recognizer = SymbolRecognizer(config)
    
    if model_path and os.path.exists(model_path):
        recognizer.load_model(model_path)
    
    return recognizer
