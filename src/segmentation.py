"""
符号分割模块
============

实现以下功能：
- 连通域分析
- 组件合并（处理分离符号如 i, =, %）
- 组件拆分（处理粘连符号）
- 特殊结构处理（根号、分数线、求和等）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

from .config import SegmentationConfig, DEFAULT_CONFIG
from .utils import BoundingBox, Symbol

logger = logging.getLogger(__name__)


class SymbolSegmenter:
    """
    符号分割器
    
    将预处理后的二值图像分割为独立的符号单元。
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        """
        初始化分割器
        
        Args:
            config: 分割配置
        """
        self.config = config or DEFAULT_CONFIG.segmentation
    
    def segment(self, binary: np.ndarray) -> List[Symbol]:
        """
        分割符号
        
        Args:
            binary: 二值图像（前景为白色）
            
        Returns:
            符号列表
        """
        logger.info("开始符号分割...")
        
        # 步骤1：连通域分析
        components = self._find_connected_components(binary)
        logger.info(f"检测到 {len(components)} 个连通域")
        
        # 步骤2：识别特殊结构
        components = self._identify_special_structures(components, binary)
        
        # 步骤3：合并分离的组件
        merged = self._merge_components(components, binary)
        logger.info(f"合并后剩余 {len(merged)} 个组件")
        
        # 步骤4：拆分粘连的组件
        split = self._split_connected(merged, binary)
        logger.info(f"拆分后共 {len(split)} 个符号")
        
        # 步骤5：提取符号图像
        symbols = self._extract_symbol_images(split, binary)
        
        return symbols
    
    def _find_connected_components(self, binary: np.ndarray) -> List[Dict]:
        """
        查找所有连通域
        
        Args:
            binary: 二值图像
            
        Returns:
            连通域信息列表
        """
        connectivity = self.config.connectivity
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=connectivity
        )
        
        components = []
        for i in range(1, num_labels):  # 跳过背景
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            # 创建该连通域的掩码
            mask = (labels == i).astype(np.uint8) * 255
            
            components.append({
                'id': i,
                'bbox': BoundingBox(x, y, w, h),
                'area': area,
                'centroid': (cx, cy),
                'mask': mask,
                'label_id': i
            })
        
        return components
    
    def _identify_special_structures(self, components: List[Dict], 
                                     binary: np.ndarray) -> List[Dict]:
        """
        识别特殊结构（分数线、根号等）
        
        Args:
            components: 连通域列表
            binary: 二值图像
            
        Returns:
            标记后的连通域列表
        """
        img_height, img_width = binary.shape
        
        for comp in components:
            bbox = comp['bbox']
            aspect_ratio = bbox.aspect_ratio
            height_ratio = bbox.height / img_height
            
            # 检测分数线：宽高比大，高度小
            if (aspect_ratio > self.config.fraction_line_aspect_ratio and
                height_ratio < self.config.fraction_line_height_ratio):
                comp['is_fraction_line'] = True
                logger.debug(f"检测到分数线: {bbox.to_tuple()}")
            else:
                comp['is_fraction_line'] = False
            
            # 检测根号符号
            comp['is_sqrt'] = self._detect_sqrt_symbol(comp, binary)
            
            # 检测求和/积分符号
            comp['is_large_operator'] = self._detect_large_operator(comp, binary)
        
        return components
    
    def _detect_sqrt_symbol(self, comp: Dict, binary: np.ndarray) -> bool:
        """
        检测根号符号
        
        根号的特征：
        - 左侧有一个向上的钩（V形结构）
        - 右侧有一条水平线（从钩顶延伸到右边界）
        - 整体宽度远大于高度
        - 主要分布在图像上部
        
        Args:
            comp: 连通域信息
            binary: 二值图像
            
        Returns:
            是否为根号
        """
        bbox = comp['bbox']
        
        # 根号必须是宽扁的形状（宽度应该大于高度的1.5倍以上）
        if bbox.width < bbox.height * 1.5:
            return False
        
        # 根号必须足够大
        if bbox.width < 20 or bbox.height < 10:
            return False
        
        # 提取组件区域
        region = binary[bbox.y:bbox.y2, bbox.x:bbox.x2]
        h, w = region.shape
        
        # 检测左侧的钩：左侧约1/4区域
        hook_width = max(3, int(w * self.config.sqrt_hook_ratio))
        hook_region = region[:, :hook_width]
        hook_density = np.sum(hook_region > 0) / (hook_region.size + 1)
        
        # 检测上方的水平线（从钩之后开始）：上部1/4区域
        top_height = max(1, h // 4)
        top_region = region[:top_height, hook_width:]
        top_density = np.sum(top_region > 0) / (top_region.size + 1)
        
        # 检测右侧区域应该相对空白（根号下面的内容区域）
        right_bottom = region[top_height:, int(w * 0.3):]
        right_bottom_density = np.sum(right_bottom > 0) / (right_bottom.size + 1)
        
        # 根号特征：有钩，有水平线，右下方相对空白
        # 数字"2"右下方会有笔画，不满足这个条件
        if hook_density > 0.15 and top_density > 0.4 and right_bottom_density < 0.2:
            return True
        
        return False
    
    def _detect_large_operator(self, comp: Dict, binary: np.ndarray) -> bool:
        """
        检测大型运算符（求和、积分等）
        
        大型运算符的特征：
        - 尺寸明显大于普通符号
        - 通常是高瘦的形状
        - 具有特殊的密度分布
        
        Args:
            comp: 连通域信息
            binary: 二值图像
            
        Returns:
            是否为大型运算符
        """
        bbox = comp['bbox']
        img_height = binary.shape[0]
        
        # 大型运算符必须占据图像高度的大部分（>60%）
        if bbox.height < img_height * 0.6:
            return False
        
        # 大型运算符通常比较大
        if bbox.area < 200:
            return False
        
        # 求和/积分符号通常比较高瘦（高度 > 宽度 * 1.2）
        if bbox.height < bbox.width * 1.2:
            return False
        
        # 检查是否具有典型的求和/积分形状
        region = binary[bbox.y:bbox.y2, bbox.x:bbox.x2]
        h, w = region.shape
        
        # 计算上、中、下三部分的密度
        h_third = max(1, h // 3)
        top = region[:h_third, :]
        middle = region[h_third:2*h_third, :]
        bottom = region[2*h_third:, :]
        
        top_density = np.sum(top > 0) / (top.size + 1)
        mid_density = np.sum(middle > 0) / (middle.size + 1)
        bottom_density = np.sum(bottom > 0) / (bottom.size + 1)
        
        # 求和符号：上下密度较高，中间较低
        if top_density > 0.3 and bottom_density > 0.3 and mid_density < 0.25:
            return True
        
        # 积分符号：S 形状，中间密度高，宽高比<0.5
        if mid_density > 0.2 and bbox.aspect_ratio < 0.5:
            return True
        
        return False
    
    def _merge_components(self, components: List[Dict], 
                          binary: np.ndarray) -> List[Dict]:
        """
        合并分离的组件
        
        处理以下情况：
        - i, j 的点和主体
        - = 的两条线
        - % 的斜线和圆圈
        - : 的两个点
        - 同一符号的断开笔画（水平距离很近）
        
        Args:
            components: 连通域列表
            binary: 二值图像
            
        Returns:
            合并后的组件列表
        """
        if not components:
            return []
        
        n = len(components)
        
        # 计算平均高度和宽度作为参考
        avg_height = np.mean([c['bbox'].height for c in components])
        avg_width = np.mean([c['bbox'].width for c in components])
        merge_distance = avg_height * self.config.merge_distance_threshold
        
        # 对于小图片（高度≤64），使用更积极的合并策略
        img_height, img_width = binary.shape
        is_small_image = img_height <= 64
        
        # 使用并查集来处理传递性合并
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 检查所有组件对是否需要合并
        for i in range(n):
            for j in range(i + 1, n):
                bbox_i = components[i]['bbox']
                bbox_j = components[j]['bbox']
                
                should_merge = False
                
                # 条件1：小点在大组件上方（i, j 的点）
                if self._is_dot_above(bbox_j, bbox_i, avg_height):
                    should_merge = True
                elif self._is_dot_above(bbox_i, bbox_j, avg_height):
                    should_merge = True
                
                # 条件2：两条对齐的水平线（= 符号）
                if self._is_aligned_horizontal_lines(bbox_i, bbox_j):
                    should_merge = True
                
                # 条件3：垂直对齐的两个点（: 符号）
                if self._is_colon_dots(bbox_i, bbox_j, avg_height):
                    should_merge = True
                
                # 条件4：% 符号的组成部分
                if self._is_percent_parts(components[i], components[j], components, avg_height):
                    should_merge = True
                
                # 条件5：对于小图片，只合并明显属于同一符号的断开笔画
                # 条件：水平有重叠 + 尺寸相似（都是小笔画片段）
                if is_small_image and not should_merge:
                    # 检查水平方向是否有重叠
                    h_overlap = bbox_i.horizontal_overlap_ratio(bbox_j)
                    
                    # 检查尺寸是否相似（同一符号的断开笔画尺寸应该接近）
                    area_ratio = min(bbox_i.area, bbox_j.area) / max(bbox_i.area, bbox_j.area)
                    
                    # 只有当水平有重叠，且尺寸相差不超过5倍时才合并
                    if h_overlap > 0.3 and area_ratio > 0.2:
                        should_merge = True
                
                if should_merge:
                    union(i, j)
        
        # 按照并查集结果分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # 合并每个组的组件
        result = []
        for indices in groups.values():
            if len(indices) == 1:
                result.append(components[indices[0]])
            else:
                # 合并多个组件
                merged_comp = components[indices[0]].copy()
                merged_bbox = merged_comp['bbox']
                merged_mask = merged_comp['mask'].copy()
                merged_area = merged_comp['area']
                
                for idx in indices[1:]:
                    comp = components[idx]
                    merged_bbox = merged_bbox.union(comp['bbox'])
                    merged_mask = cv2.bitwise_or(merged_mask, comp['mask'])
                    merged_area += comp['area']
                
                merged_comp['bbox'] = merged_bbox
                merged_comp['mask'] = merged_mask
                merged_comp['area'] = merged_area
                result.append(merged_comp)
        
        return result
    
    def _is_dot_above(self, small_bbox: BoundingBox, large_bbox: BoundingBox,
                      avg_height: float) -> bool:
        """
        检查小组件是否是大组件上方的点
        
        Args:
            small_bbox: 小组件边界框
            large_bbox: 大组件边界框
            avg_height: 平均高度
            
        Returns:
            是否满足点的条件
        """
        # 小组件必须足够小
        if small_bbox.area > large_bbox.area * self.config.dot_size_ratio:
            return False
        
        # 小组件在大组件正上方
        if small_bbox.y2 > large_bbox.y:
            return False
        
        # 水平位置大致对齐
        h_overlap = small_bbox.horizontal_overlap_ratio(large_bbox)
        if h_overlap < 0.3:
            return False
        
        # 距离不能太远
        vertical_gap = large_bbox.y - small_bbox.y2
        if vertical_gap > avg_height * 0.5:
            return False
        
        return True
    
    def _is_aligned_horizontal_lines(self, bbox1: BoundingBox, 
                                     bbox2: BoundingBox) -> bool:
        """
        检查两个组件是否是对齐的水平线（= 符号）
        
        Args:
            bbox1: 第一个边界框
            bbox2: 第二个边界框
            
        Returns:
            是否满足条件
        """
        # 两个都应该是宽扁的形状
        if bbox1.aspect_ratio < 2 or bbox2.aspect_ratio < 2:
            return False
        
        # 宽度应该相近
        width_ratio = min(bbox1.width, bbox2.width) / max(bbox1.width, bbox2.width)
        if width_ratio < 0.7:
            return False
        
        # 水平对齐
        h_overlap = bbox1.horizontal_overlap_ratio(bbox2)
        if h_overlap < 0.7:
            return False
        
        # 垂直方向有间隔但不重叠
        if bbox1.y2 > bbox2.y and bbox2.y2 > bbox1.y:
            return False  # 有垂直重叠
        
        # 间距合理
        gap = abs(min(bbox1.y, bbox2.y) - max(bbox1.y2, bbox2.y2))
        avg_height = (bbox1.height + bbox2.height) / 2
        if gap > avg_height * 3:
            return False
        
        return True
    
    def _is_colon_dots(self, bbox1: BoundingBox, bbox2: BoundingBox,
                       avg_height: float) -> bool:
        """
        检查两个组件是否是冒号的两个点
        
        Args:
            bbox1: 第一个边界框
            bbox2: 第二个边界框
            avg_height: 平均高度
            
        Returns:
            是否满足条件
        """
        # 两个都应该是小的、接近正方形的
        if bbox1.area > avg_height * avg_height * 0.2:
            return False
        if bbox2.area > avg_height * avg_height * 0.2:
            return False
        
        # 水平对齐
        h_overlap = bbox1.horizontal_overlap_ratio(bbox2)
        if h_overlap < 0.5:
            return False
        
        # 垂直排列
        v_gap = abs(bbox1.center_y - bbox2.center_y)
        if v_gap < avg_height * 0.2 or v_gap > avg_height * 0.8:
            return False
        
        return True
    
    def _is_percent_parts(self, comp1: Dict, comp2: Dict,
                          all_components: List[Dict],
                          avg_height: float) -> bool:
        """
        检查两个组件是否是百分号的组成部分
        
        % 符号由两个小圆圈和一条斜线组成
        
        Args:
            comp1: 第一个组件
            comp2: 第二个组件
            all_components: 所有组件
            avg_height: 平均高度
            
        Returns:
            是否满足条件
        """
        # 简化实现：检查是否是对角排列的两个小圆圈
        bbox1 = comp1['bbox']
        bbox2 = comp2['bbox']
        
        # 两个都应该是小的
        if bbox1.area > avg_height * avg_height * 0.15:
            return False
        if bbox2.area > avg_height * avg_height * 0.15:
            return False
        
        # 对角排列
        dx = abs(bbox1.center_x - bbox2.center_x)
        dy = abs(bbox1.center_y - bbox2.center_y)
        
        if dx < avg_height * 0.2 or dy < avg_height * 0.2:
            return False
        
        return True
    
    def _split_connected(self, components: List[Dict],
                         binary: np.ndarray) -> List[Dict]:
        """
        拆分粘连的组件
        
        使用投影分析和笔画分析检测并拆分粘连的符号。
        
        Args:
            components: 组件列表
            binary: 二值图像
            
        Returns:
            拆分后的组件列表
        """
        if not components:
            return []
        
        # 计算统计信息
        widths = [c['bbox'].width for c in components]
        heights = [c['bbox'].height for c in components]
        median_width = np.median(widths)
        median_height = np.median(heights)
        
        result = []
        
        for comp in components:
            bbox = comp['bbox']
            
            # 检查是否需要拆分（宽度异常大）
            if bbox.width > median_width * 2.5 and not comp.get('is_fraction_line', False):
                # 尝试拆分
                split_result = self._try_split(comp, binary, median_width)
                if len(split_result) > 1:
                    result.extend(split_result)
                    continue
            
            result.append(comp)
        
        return result
    
    def _try_split(self, comp: Dict, binary: np.ndarray,
                   expected_width: float) -> List[Dict]:
        """
        尝试拆分粘连组件
        
        使用垂直投影分析找到分割点。
        
        Args:
            comp: 组件信息
            binary: 二值图像
            expected_width: 预期单个符号宽度
            
        Returns:
            拆分后的组件列表
        """
        bbox = comp['bbox']
        region = binary[bbox.y:bbox.y2, bbox.x:bbox.x2]
        
        # 计算垂直投影
        v_projection = np.sum(region, axis=0)
        
        # 寻找投影的低谷作为分割点
        split_points = self._find_split_points(v_projection, expected_width)
        
        if not split_points:
            return [comp]
        
        # 执行分割
        result = []
        prev_x = 0
        
        for split_x in split_points + [bbox.width]:
            if split_x - prev_x < expected_width * 0.3:
                continue
            
            # 创建新的组件
            new_bbox = BoundingBox(
                bbox.x + prev_x,
                bbox.y,
                split_x - prev_x,
                bbox.height
            )
            
            new_mask = np.zeros_like(comp['mask'])
            new_mask[bbox.y:bbox.y2, bbox.x + prev_x:bbox.x + split_x] = \
                comp['mask'][bbox.y:bbox.y2, bbox.x + prev_x:bbox.x + split_x]
            
            new_area = np.sum(new_mask > 0)
            
            if new_area > 0:
                result.append({
                    'id': len(result),
                    'bbox': new_bbox,
                    'area': new_area,
                    'centroid': new_bbox.center,
                    'mask': new_mask,
                    'is_fraction_line': False,
                    'is_sqrt': False,
                    'is_large_operator': False
                })
            
            prev_x = split_x
        
        return result if len(result) > 1 else [comp]
    
    def _find_split_points(self, projection: np.ndarray,
                           expected_width: float) -> List[int]:
        """
        在投影中找到分割点
        
        Args:
            projection: 垂直投影
            expected_width: 预期宽度
            
        Returns:
            分割点位置列表
        """
        if len(projection) < 2:
            return []
        
        # 平滑投影
        kernel_size = max(3, int(expected_width * 0.1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(projection, np.ones(kernel_size) / kernel_size, 
                               mode='same')
        
        # 找到局部最小值
        threshold = np.max(smoothed) * 0.2
        
        split_points = []
        in_valley = False
        valley_start = 0
        
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] < threshold:
                if not in_valley:
                    valley_start = i
                    in_valley = True
            else:
                if in_valley:
                    # 记录谷的中点
                    valley_mid = (valley_start + i) // 2
                    
                    # 确保分割点间距合理
                    if not split_points or valley_mid - split_points[-1] > expected_width * 0.5:
                        split_points.append(valley_mid)
                    
                    in_valley = False
        
        return split_points
    
    def _extract_symbol_images(self, components: List[Dict],
                               binary: np.ndarray) -> List[Symbol]:
        """
        提取符号图像
        
        Args:
            components: 组件列表
            binary: 二值图像
            
        Returns:
            符号列表
        """
        symbols = []
        
        for i, comp in enumerate(components):
            bbox = comp['bbox']
            
            # 提取符号图像
            symbol_img = binary[bbox.y:bbox.y2, bbox.x:bbox.x2].copy()
            
            # 创建 Symbol 对象
            symbol = Symbol(
                image=symbol_img,
                bbox=bbox,
                is_fraction_line=comp.get('is_fraction_line', False),
                is_sqrt_symbol=comp.get('is_sqrt', False),
                is_sum_symbol=comp.get('is_large_operator', False)
            )
            
            symbols.append(symbol)
        
        return symbols
    
    def extract_sqrt_content(self, sqrt_symbol: Symbol, 
                             all_symbols: List[Symbol],
                             binary: np.ndarray) -> List[Symbol]:
        """
        提取根号内部的内容
        
        Args:
            sqrt_symbol: 根号符号
            all_symbols: 所有符号
            binary: 二值图像
            
        Returns:
            根号内部的符号列表
        """
        sqrt_bbox = sqrt_symbol.bbox
        
        # 根号覆盖的区域：根号右侧，被水平线覆盖的部分
        # 简化处理：取根号右边界右侧的符号
        
        inner_symbols = []
        for symbol in all_symbols:
            if symbol is sqrt_symbol:
                continue
            
            symbol_bbox = symbol.bbox
            
            # 检查是否在根号内部
            if (symbol_bbox.x > sqrt_bbox.x + sqrt_bbox.width * 0.3 and
                symbol_bbox.x < sqrt_bbox.x2 and
                symbol_bbox.center_y > sqrt_bbox.y and
                symbol_bbox.center_y < sqrt_bbox.y2):
                inner_symbols.append(symbol)
        
        return inner_symbols
    
    def extract_fraction_parts(self, fraction_line: Symbol,
                               all_symbols: List[Symbol]) -> Tuple[List[Symbol], List[Symbol]]:
        """
        提取分数的分子和分母
        
        Args:
            fraction_line: 分数线符号
            all_symbols: 所有符号
            
        Returns:
            (分子符号列表, 分母符号列表)
        """
        line_bbox = fraction_line.bbox
        
        numerator = []
        denominator = []
        
        for symbol in all_symbols:
            if symbol is fraction_line:
                continue
            
            symbol_bbox = symbol.bbox
            
            # 检查水平位置是否与分数线重叠
            h_overlap = symbol_bbox.horizontal_overlap_ratio(line_bbox)
            if h_overlap < 0.3:
                continue
            
            # 根据垂直位置分类
            if symbol_bbox.center_y < line_bbox.center_y:
                numerator.append(symbol)
            else:
                denominator.append(symbol)
        
        return numerator, denominator


# 便捷函数
def segment_symbols(binary: np.ndarray,
                    config: Optional[SegmentationConfig] = None) -> List[Symbol]:
    """
    分割符号的便捷函数
    
    Args:
        binary: 二值图像
        config: 分割配置
        
    Returns:
        符号列表
    """
    segmenter = SymbolSegmenter(config)
    return segmenter.segment(binary)
