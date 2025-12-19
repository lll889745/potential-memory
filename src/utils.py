"""
工具函数模块
============

包含项目中通用的工具函数。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """边界框数据类"""
    x: int  # 左上角 x
    y: int  # 左上角 y
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        """右下角 x"""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """右下角 y"""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """中心点坐标"""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    @property
    def area(self) -> int:
        """面积"""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """宽高比"""
        return self.width / max(self.height, 1)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """转换为元组 (x, y, w, h)"""
        return (self.x, self.y, self.width, self.height)
    
    def to_rect(self) -> Tuple[int, int, int, int]:
        """转换为矩形格式 (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x2, self.y2)
    
    def expand(self, margin: int) -> 'BoundingBox':
        """扩展边界框"""
        return BoundingBox(
            max(0, self.x - margin),
            max(0, self.y - margin),
            self.width + 2 * margin,
            self.height + 2 * margin
        )
    
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """计算与另一个边界框的交集"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 < x2 and y1 < y2:
            return BoundingBox(x1, y1, x2 - x1, y2 - y1)
        return None
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """计算与另一个边界框的并集"""
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """计算 IoU（交并比）"""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        
        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        return intersection_area / max(union_area, 1)
    
    def vertical_overlap_ratio(self, other: 'BoundingBox') -> float:
        """计算垂直方向的重叠比例"""
        overlap_top = max(self.y, other.y)
        overlap_bottom = min(self.y2, other.y2)
        overlap_height = max(0, overlap_bottom - overlap_top)
        
        min_height = min(self.height, other.height)
        return overlap_height / max(min_height, 1)
    
    def horizontal_overlap_ratio(self, other: 'BoundingBox') -> float:
        """计算水平方向的重叠比例"""
        overlap_left = max(self.x, other.x)
        overlap_right = min(self.x2, other.x2)
        overlap_width = max(0, overlap_right - overlap_left)
        
        min_width = min(self.width, other.width)
        return overlap_width / max(min_width, 1)


# 用于生成唯一符号 ID 的计数器
_symbol_id_counter = 0


def _get_next_symbol_id() -> int:
    """获取下一个唯一的符号 ID"""
    global _symbol_id_counter
    _symbol_id_counter += 1
    return _symbol_id_counter


@dataclass(eq=False)
class Symbol:
    """符号数据类"""
    image: np.ndarray  # 符号图像
    bbox: BoundingBox  # 边界框
    label: Optional[str] = None  # 识别标签
    confidence: float = 0.0  # 识别置信度
    candidates: List[Tuple[str, float]] = field(default_factory=list)  # 候选标签
    
    # 特殊标记
    is_fraction_line: bool = False  # 是否为分数线
    is_sqrt_symbol: bool = False  # 是否为根号符号
    is_sum_symbol: bool = False  # 是否为求和符号
    is_integral_symbol: bool = False  # 是否为积分符号
    
    # 关联信息
    parent_id: Optional[int] = None  # 父符号 ID
    children_ids: List[int] = field(default_factory=list)  # 子符号 ID 列表
    
    # 唯一标识符
    _id: int = field(default_factory=_get_next_symbol_id)
    
    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center
    
    def __eq__(self, other: object) -> bool:
        """基于唯一 ID 比较符号"""
        if not isinstance(other, Symbol):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        """基于唯一 ID 计算哈希"""
        return hash(self._id)


@dataclass
class SyntaxNode:
    """语法树节点"""
    node_type: str  # 节点类型
    value: Optional[str] = None  # 节点值（对于叶子节点）
    children: List['SyntaxNode'] = field(default_factory=list)
    
    # 位置信息
    bbox: Optional[BoundingBox] = None
    
    def add_child(self, child: 'SyntaxNode') -> None:
        """添加子节点"""
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'type': self.node_type,
            'value': self.value,
        }
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        return result
    
    def __repr__(self) -> str:
        if self.value:
            return f"SyntaxNode({self.node_type}: {self.value})"
        return f"SyntaxNode({self.node_type}, children={len(self.children)})"


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """
    加载图像
    
    Args:
        path: 图像路径
        grayscale: 是否转换为灰度图
        
    Returns:
        图像数组
    """
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    
    return img


def save_image(img: np.ndarray, path: str) -> None:
    """保存图像"""
    cv2.imwrite(path, img)


def resize_image(img: np.ndarray, target_height: int, 
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        img: 输入图像
        target_height: 目标高度
        keep_aspect_ratio: 是否保持宽高比
        
    Returns:
        调整大小后的图像
    """
    h, w = img.shape[:2]
    
    if keep_aspect_ratio:
        scale = target_height / h
        target_width = int(w * scale)
    else:
        target_width = target_height
    
    return cv2.resize(img, (target_width, target_height), 
                      interpolation=cv2.INTER_LINEAR)


def pad_image(img: np.ndarray, target_size: Tuple[int, int],
              pad_value: int = 0) -> np.ndarray:
    """
    填充图像到目标大小
    
    Args:
        img: 输入图像
        target_size: 目标大小 (height, width)
        pad_value: 填充值
        
    Returns:
        填充后的图像
    """
    h, w = img.shape[:2]
    th, tw = target_size
    
    # 首先调整大小以适应目标尺寸
    scale = min(th / h, tw / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 对于二值图像（只有0和255），使用最近邻插值以保持锐利边缘
    unique_vals = np.unique(img)
    is_binary = len(unique_vals) <= 2 and (0 in unique_vals or 255 in unique_vals)
    
    if is_binary:
        # 二值图像使用最近邻插值
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        # 灰度图像使用线性插值
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建填充后的图像
    if len(img.shape) == 3:
        padded = np.full((th, tw, img.shape[2]), pad_value, dtype=img.dtype)
    else:
        padded = np.full((th, tw), pad_value, dtype=img.dtype)
    
    # 居中放置
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded


def calculate_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """计算图像直方图"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()


def find_contours(binary_img: np.ndarray) -> List[np.ndarray]:
    """查找轮廓"""
    contours, _ = cv2.findContours(
        binary_img, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def contour_to_bbox(contour: np.ndarray) -> BoundingBox:
    """将轮廓转换为边界框"""
    x, y, w, h = cv2.boundingRect(contour)
    return BoundingBox(x, y, w, h)


def calculate_moments(binary_img: np.ndarray) -> Dict[str, float]:
    """计算图像矩"""
    moments = cv2.moments(binary_img)
    return moments


def calculate_hu_moments(binary_img: np.ndarray) -> np.ndarray:
    """计算 Hu 不变矩"""
    moments = cv2.moments(binary_img)
    hu_moments = cv2.HuMoments(moments).flatten()
    # 对数变换以归一化
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments


def distance_between_bboxes(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """计算两个边界框中心点之间的距离"""
    c1 = bbox1.center
    c2 = bbox2.center
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def sort_symbols_by_position(symbols: List[Symbol], 
                             tolerance: float = 0.3) -> List[Symbol]:
    """
    按位置排序符号（从左到右，从上到下）
    
    Args:
        symbols: 符号列表
        tolerance: 同一行的 y 坐标容差（相对于平均高度）
        
    Returns:
        排序后的符号列表
    """
    if not symbols:
        return []
    
    # 计算平均高度
    avg_height = np.mean([s.bbox.height for s in symbols])
    y_tolerance = avg_height * tolerance
    
    # 按 y 坐标分组（同一行）
    sorted_by_y = sorted(symbols, key=lambda s: s.bbox.center_y)
    
    lines = []
    current_line = [sorted_by_y[0]]
    
    for symbol in sorted_by_y[1:]:
        if abs(symbol.bbox.center_y - current_line[-1].bbox.center_y) < y_tolerance:
            current_line.append(symbol)
        else:
            lines.append(current_line)
            current_line = [symbol]
    lines.append(current_line)
    
    # 每行按 x 坐标排序
    result = []
    for line in lines:
        line_sorted = sorted(line, key=lambda s: s.bbox.center_x)
        result.extend(line_sorted)
    
    return result


def visualize_symbols(img: np.ndarray, symbols: List[Symbol],
                      show_labels: bool = True) -> np.ndarray:
    """
    可视化符号边界框
    
    Args:
        img: 输入图像
        symbols: 符号列表
        show_labels: 是否显示标签
        
    Returns:
        可视化结果图像
    """
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 黄色
    ]
    
    for i, symbol in enumerate(symbols):
        color = colors[i % len(colors)]
        bbox = symbol.bbox
        
        # 绘制边界框
        cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 2)
        
        # 绘制标签
        if show_labels and symbol.label:
            label_text = f"{symbol.label} ({symbol.confidence:.2f})"
            cv2.putText(vis, label_text, (bbox.x, bbox.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis


def visualize_syntax_tree(root: SyntaxNode, depth: int = 0) -> str:
    """
    可视化语法树（文本形式）
    
    Args:
        root: 根节点
        depth: 当前深度
        
    Returns:
        树的文本表示
    """
    indent = "  " * depth
    
    if root.value:
        result = f"{indent}[{root.node_type}]: {root.value}\n"
    else:
        result = f"{indent}[{root.node_type}]\n"
    
    for child in root.children:
        result += visualize_syntax_tree(child, depth + 1)
    
    return result
