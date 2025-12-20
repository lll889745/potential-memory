"""
结构分析模块
============

实现以下功能：
- 空间关系判定
- 语法树构建
- LaTeX 代码生成
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import StructureConfig, DEFAULT_CONFIG, SYMBOL_TO_LATEX, LATEX_TEMPLATES
from .utils import Symbol, BoundingBox, SyntaxNode, sort_symbols_by_position

logger = logging.getLogger(__name__)


class SpatialRelation(Enum):
    """空间关系类型"""
    RIGHT = "right"          # 右邻关系
    SUPERSCRIPT = "superscript"  # 上标关系
    SUBSCRIPT = "subscript"      # 下标关系
    ABOVE = "above"          # 正上方
    BELOW = "below"          # 正下方
    INSIDE = "inside"        # 内部关系
    NUMERATOR = "numerator"      # 分子关系
    DENOMINATOR = "denominator"  # 分母关系
    UNKNOWN = "unknown"      # 未知关系


class NodeType(Enum):
    """语法树节点类型"""
    EXPRESSION = "expression"    # 表达式
    EQUATION = "equation"        # 等式
    FRACTION = "fraction"        # 分数
    SQRT = "sqrt"                # 根号
    SUPERSCRIPT = "superscript"  # 上标
    SUBSCRIPT = "subscript"      # 下标
    SUM = "sum"                  # 求和
    PRODUCT = "product"          # 连乘
    INTEGRAL = "integral"        # 积分
    LIMIT = "limit"              # 极限
    SYMBOL = "symbol"            # 符号
    OPERATOR = "operator"        # 运算符
    GROUP = "group"              # 分组


class StructureAnalyzer:
    """
    结构分析器
    
    分析符号之间的空间关系，构建语法树，生成 LaTeX 代码。
    """
    
    def __init__(self, config: Optional[StructureConfig] = None):
        """
        初始化结构分析器
        
        Args:
            config: 结构分析配置
        """
        self.config = config or DEFAULT_CONFIG.structure
        
        # 运算符优先级
        self.operator_precedence = {
            '=': 1, '<': 1, '>': 1, 'leq': 1, 'geq': 1, 'neq': 1,
            '+': 2, '-': 2, 'pm': 2,
            '*': 3, 'times': 3, 'cdot': 3, '/': 3, 'div': 3,
            '^': 4, '_': 4,
        }
    
    def analyze(self, symbols: List[Symbol]) -> Tuple[SyntaxNode, str]:
        """
        分析符号结构并生成 LaTeX
        
        Args:
            symbols: 识别后的符号列表
            
        Returns:
            (语法树根节点, LaTeX 字符串)
        """
        if not symbols:
            return SyntaxNode(NodeType.EXPRESSION.value), ""
        
        logger.info(f"开始结构分析，符号数: {len(symbols)}")
        
        # 步骤1：计算空间关系矩阵
        relations = self._compute_relations(symbols)
        
        # 步骤2：处理特殊结构（分数、根号等）
        symbols, groups = self._process_special_structures(symbols, relations)
        
        # 步骤3：构建语法树
        syntax_tree = self._build_syntax_tree(symbols, relations, groups)
        
        # 步骤4：生成 LaTeX
        latex = self._generate_latex(syntax_tree)
        
        logger.info(f"生成的 LaTeX: {latex}")
        
        return syntax_tree, latex
    
    def _compute_relations(self, symbols: List[Symbol]) -> Dict[Tuple[int, int], SpatialRelation]:
        """
        计算符号间的空间关系
        
        Args:
            symbols: 符号列表
            
        Returns:
            空间关系字典 {(i, j): relation}
        """
        relations = {}
        
        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                if i == j:
                    continue
                
                relation = self._classify_relation(sym_a, sym_b)
                if relation != SpatialRelation.UNKNOWN:
                    relations[(i, j)] = relation
        
        return relations
    
    def _classify_relation(self, symbol_a: Symbol, symbol_b: Symbol) -> SpatialRelation:
        """
        判断 symbol_b 相对于 symbol_a 的空间关系
        
        Args:
            symbol_a: 基准符号
            symbol_b: 目标符号
            
        Returns:
            空间关系类型
        """
        bbox_a = symbol_a.bbox
        bbox_b = symbol_b.bbox
        
        # 计算相对位置
        dx = bbox_b.center_x - bbox_a.center_x
        dy = bbox_b.center_y - bbox_a.center_y
        
        # 计算尺寸比例
        size_a = max(bbox_a.width, bbox_a.height)
        size_b = max(bbox_b.width, bbox_b.height)
        height_a = bbox_a.height
        height_b = bbox_b.height
        size_ratio = size_b / max(size_a, 1)
        height_ratio = height_b / max(height_a, 1)
        
        # 计算重叠
        v_overlap = bbox_a.vertical_overlap_ratio(bbox_b)
        h_overlap = bbox_a.horizontal_overlap_ratio(bbox_b)
        
        # 归一化距离
        norm_dx = dx / max(size_a, 1)
        norm_dy = dy / max(height_a, 1)
        
        # 计算水平距离（从基础符号右边缘到目标符号左边缘）
        h_gap = bbox_b.x - bbox_a.x2
        norm_h_gap = h_gap / max(bbox_a.width, 1)
        
        # 获取最大水平距离限制
        max_h_distance = getattr(self.config, 'script_max_horizontal_distance', 2.5)
        
        # 分数线处理 - 优化：检查水平覆盖
        if symbol_a.is_fraction_line:
            # 检查符号是否在分数线的水平范围内
            h_coverage = self._compute_horizontal_coverage(bbox_a, bbox_b)
            
            # 使用绝对距离而不是归一化距离（因为分数线高度很小）
            abs_dy = abs(dy)
            # 分子/分母与分数线的距离应该在合理范围内（基于目标符号的高度）
            max_distance = max(bbox_b.height * 2, 50)  # 最大距离为目标符号高度的2倍或50像素
            
            if h_coverage > 0.2:  # 至少20%水平覆盖
                if dy < 0 and abs_dy < max_distance:
                    return SpatialRelation.NUMERATOR
                elif dy > 0 and abs_dy < max_distance:
                    return SpatialRelation.DENOMINATOR
        
        # 上标判定 - 优化后的逻辑
        is_superscript = self._check_superscript(
            bbox_a, bbox_b, dx, dy, norm_dx, norm_dy, 
            size_ratio, height_ratio, norm_h_gap, max_h_distance
        )
        if is_superscript:
            return SpatialRelation.SUPERSCRIPT
        
        # 下标判定 - 优化后的逻辑
        is_subscript = self._check_subscript(
            bbox_a, bbox_b, dx, dy, norm_dx, norm_dy,
            size_ratio, height_ratio, norm_h_gap, max_h_distance
        )
        if is_subscript:
            return SpatialRelation.SUBSCRIPT
        
        # 右邻判定
        if (norm_dx > 0.3 and v_overlap > self.config.vertical_overlap_threshold):
            return SpatialRelation.RIGHT
        
        # 正上方
        if (abs(norm_dx) < 0.5 and norm_dy < -0.5 and h_overlap > 0.3):
            return SpatialRelation.ABOVE
        
        # 正下方
        if (abs(norm_dx) < 0.5 and norm_dy > 0.5 and h_overlap > 0.3):
            return SpatialRelation.BELOW
        
        return SpatialRelation.UNKNOWN
    
    def _compute_horizontal_coverage(self, bbox_a: BoundingBox, bbox_b: BoundingBox) -> float:
        """
        计算 bbox_b 在 bbox_a 水平范围内的覆盖率
        
        Args:
            bbox_a: 基准边界框（如分数线）
            bbox_b: 目标边界框（如分子或分母）
            
        Returns:
            覆盖率 (0.0 - 1.0)
        """
        overlap_left = max(bbox_a.x, bbox_b.x)
        overlap_right = min(bbox_a.x2, bbox_b.x2)
        
        if overlap_right <= overlap_left:
            return 0.0
        
        overlap_width = overlap_right - overlap_left
        target_width = bbox_b.width
        
        return overlap_width / max(target_width, 1)
    
    def _check_superscript(self, bbox_a: BoundingBox, bbox_b: BoundingBox,
                           dx: float, dy: float, norm_dx: float, norm_dy: float,
                           size_ratio: float, height_ratio: float,
                           norm_h_gap: float, max_h_distance: float) -> bool:
        """
        检查是否为上标关系
        
        上标特征：
        1. 目标符号位于基准符号的右上方
        2. 目标符号的底部在基准符号中线以上或附近
        3. 目标符号通常较小（但不是必须的，如 x^2 中的 2 可能较大）
        4. 水平距离不能太远
        
        Args:
            bbox_a: 基准符号边界框
            bbox_b: 目标符号边界框
            dx, dy: 中心点偏移
            norm_dx, norm_dy: 归一化偏移
            size_ratio: 尺寸比例
            height_ratio: 高度比例
            norm_h_gap: 归一化水平间隙
            max_h_distance: 最大水平距离
            
        Returns:
            是否为上标
        """
        # 必须在右侧（中心点或左边缘在基准符号右边缘附近或之后）
        if dx <= 0 and bbox_b.x < bbox_a.x2 - bbox_a.width * 0.2:
            return False
        
        # 水平距离限制
        if norm_h_gap > max_h_distance:
            return False
        
        # 检查垂直位置：目标符号的底部应该在基准符号中线以上或附近
        base_midline = bbox_a.center_y
        target_bottom = bbox_b.y2
        
        # 上标的底部应该在基准符号的上半部分（或中线附近）
        vertical_position = (target_bottom - bbox_a.y) / max(bbox_a.height, 1)
        
        # 条件1：目标符号底部在基准符号上半部分（传统上标）
        # 必须同时满足：底部在上60%区域 + 中心明显偏上 + 尺寸偏小
        is_traditional_superscript = (
            vertical_position < 0.6 and  # 底部在上60%区域
            dy < -bbox_a.height * 0.15 and  # 中心明显在上方
            size_ratio < 0.9  # 尺寸需要偏小
        )
        
        # 条件2：目标符号明显偏小且在右上方向偏移
        is_small_superscript = (
            dy < -self.config.superscript_y_threshold * bbox_a.height and
            size_ratio < self.config.script_size_ratio and
            dx > 0
        )
        
        # 条件3：目标符号顶部明显高于基准符号（必须尺寸偏小）
        is_elevated = (
            bbox_b.y < bbox_a.y - bbox_a.height * 0.15 and  # 顶部明显更高
            dx > 0 and
            size_ratio < 0.85 and  # 必须尺寸偏小
            norm_h_gap < max_h_distance * 0.5
        )
        
        return is_traditional_superscript or is_small_superscript or is_elevated
    
    def _check_subscript(self, bbox_a: BoundingBox, bbox_b: BoundingBox,
                         dx: float, dy: float, norm_dx: float, norm_dy: float,
                         size_ratio: float, height_ratio: float,
                         norm_h_gap: float, max_h_distance: float) -> bool:
        """
        检查是否为下标关系
        
        下标特征：
        1. 目标符号位于基准符号的右下方
        2. 目标符号的顶部在基准符号中线以下或附近
        3. 目标符号通常较小（但不是必须的）
        4. 水平距离不能太远
        
        Args:
            bbox_a: 基准符号边界框
            bbox_b: 目标符号边界框
            dx, dy: 中心点偏移
            norm_dx, norm_dy: 归一化偏移
            size_ratio: 尺寸比例
            height_ratio: 高度比例
            norm_h_gap: 归一化水平间隙
            max_h_distance: 最大水平距离
            
        Returns:
            是否为下标
        """
        # 必须在右侧
        if dx <= 0 and bbox_b.x < bbox_a.x2 - bbox_a.width * 0.2:
            return False
        
        # 水平距离限制
        if norm_h_gap > max_h_distance:
            return False
        
        # 检查垂直位置：目标符号的顶部应该在基准符号中线以下或附近
        base_midline = bbox_a.center_y
        target_top = bbox_b.y
        
        # 下标的顶部应该在基准符号的下半部分
        vertical_position = (target_top - bbox_a.y) / max(bbox_a.height, 1)
        
        # 条件1：目标符号顶部在基准符号下半部分（传统下标）
        # 必须同时满足：顶部在下60%区域 + 中心明显偏下 + 尺寸偏小
        is_traditional_subscript = (
            vertical_position > 0.4 and  # 顶部在下60%区域
            dy > bbox_a.height * 0.15 and  # 中心明显在下方
            size_ratio < 0.9  # 尺寸需要偏小
        )
        
        # 条件2：目标符号明显偏小且在右下方向偏移
        is_small_subscript = (
            dy > self.config.subscript_y_threshold * bbox_a.height and
            size_ratio < self.config.script_size_ratio and
            dx > 0
        )
        
        # 条件3：目标符号底部明显低于基准符号（必须尺寸偏小）
        is_lowered = (
            bbox_b.y2 > bbox_a.y2 + bbox_a.height * 0.15 and  # 底部明显更低
            dx > 0 and
            size_ratio < 0.85 and  # 必须尺寸偏小
            norm_h_gap < max_h_distance * 0.5
        )
        
        return is_traditional_subscript or is_small_subscript or is_lowered
    
    def _process_special_structures(self, symbols: List[Symbol],
                                    relations: Dict) -> Tuple[List[Symbol], List[Dict]]:
        """
        处理特殊结构（分数、根号等）
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            (更新后的符号列表, 结构分组信息)
        """
        groups = []
        
        # 处理分数结构
        fraction_groups = self._find_fraction_groups(symbols, relations)
        groups.extend(fraction_groups)
        
        # 处理根号结构
        sqrt_groups = self._find_sqrt_groups(symbols, relations)
        groups.extend(sqrt_groups)
        
        # 处理求和/积分结构
        operator_groups = self._find_large_operator_groups(symbols, relations)
        groups.extend(operator_groups)
        
        return symbols, groups
    
    def _find_fraction_groups(self, symbols: List[Symbol],
                              relations: Dict) -> List[Dict]:
        """
        查找分数结构
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            分数分组列表
        """
        groups = []
        
        for i, symbol in enumerate(symbols):
            if symbol.is_fraction_line:
                numerator = []
                denominator = []
                
                fraction_bbox = symbol.bbox
                
                for j, other in enumerate(symbols):
                    if i == j:
                        continue
                    
                    # 首先检查已计算的关系
                    rel = relations.get((i, j))
                    if rel == SpatialRelation.NUMERATOR:
                        numerator.append(j)
                    elif rel == SpatialRelation.DENOMINATOR:
                        denominator.append(j)
                    else:
                        # 如果关系未知，使用直接的几何判断
                        other_bbox = other.bbox
                        
                        # 检查水平覆盖
                        h_coverage = self._compute_horizontal_coverage(fraction_bbox, other_bbox)
                        
                        if h_coverage > 0.2:  # 至少20%水平覆盖
                            # 使用目标符号的高度作为参考（因为分数线高度太小）
                            max_distance = max(other_bbox.height * 2, 50)
                            
                            # 检查垂直位置
                            if other_bbox.center_y < fraction_bbox.center_y:
                                # 在分数线上方
                                vertical_dist = fraction_bbox.y - other_bbox.y2
                                if vertical_dist < max_distance:
                                    numerator.append(j)
                            elif other_bbox.center_y > fraction_bbox.center_y:
                                # 在分数线下方
                                vertical_dist = other_bbox.y - fraction_bbox.y2
                                if vertical_dist < max_distance:
                                    denominator.append(j)
                
                # 只有当有分子或分母时才创建分数组
                if numerator or denominator:
                    groups.append({
                        'type': 'fraction',
                        'line_idx': i,
                        'numerator_indices': numerator,
                        'denominator_indices': denominator
                    })
        
        return groups
    
    def _find_sqrt_groups(self, symbols: List[Symbol],
                          relations: Dict) -> List[Dict]:
        """
        查找根号结构
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            根号分组列表
        """
        groups = []
        
        for i, symbol in enumerate(symbols):
            if symbol.is_sqrt_symbol:
                inner = []
                
                # 找到根号内部的符号
                sqrt_bbox = symbol.bbox
                for j, other in enumerate(symbols):
                    if i == j:
                        continue
                    
                    other_bbox = other.bbox
                    # 检查是否在根号覆盖范围内
                    if (other_bbox.center_x > sqrt_bbox.x + sqrt_bbox.width * 0.3 and
                        other_bbox.center_x < sqrt_bbox.x2 and
                        other_bbox.center_y > sqrt_bbox.y and
                        other_bbox.center_y < sqrt_bbox.y2):
                        inner.append(j)
                
                groups.append({
                    'type': 'sqrt',
                    'sqrt_idx': i,
                    'inner_indices': inner
                })
        
        return groups
    
    def _find_large_operator_groups(self, symbols: List[Symbol],
                                    relations: Dict) -> List[Dict]:
        """
        查找大型运算符结构（求和、积分等）
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            运算符分组列表
        """
        groups = []
        
        for i, symbol in enumerate(symbols):
            if symbol.is_sum_symbol or symbol.is_integral_symbol:
                upper_limit = []
                lower_limit = []
                operand = []
                
                for j, other in enumerate(symbols):
                    if i == j:
                        continue
                    
                    rel = relations.get((i, j))
                    if rel == SpatialRelation.ABOVE:
                        upper_limit.append(j)
                    elif rel == SpatialRelation.BELOW:
                        lower_limit.append(j)
                    elif rel == SpatialRelation.RIGHT:
                        operand.append(j)
                
                op_type = 'sum' if symbol.is_sum_symbol else 'integral'
                groups.append({
                    'type': op_type,
                    'operator_idx': i,
                    'upper_limit_indices': upper_limit,
                    'lower_limit_indices': lower_limit,
                    'operand_indices': operand
                })
        
        return groups
    
    def _build_syntax_tree(self, symbols: List[Symbol],
                           relations: Dict,
                           groups: List[Dict]) -> SyntaxNode:
        """
        构建语法树
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            groups: 结构分组信息
            
        Returns:
            语法树根节点
        """
        # 按位置排序符号
        sorted_symbols = sort_symbols_by_position(symbols)
        
        # 标记已处理的符号
        processed = set()
        
        # 首先处理分组结构（分数、根号等），收集组内符号
        fraction_line_indices = set()
        fraction_component_indices = set()  # 分数的分子和分母
        
        # 处理分组结构
        group_nodes = {}
        for group in groups:
            if group['type'] == 'fraction':
                line_idx = group.get('line_idx')
                fraction_line_indices.add(line_idx)
                fraction_component_indices.update(group.get('numerator_indices', []))
                fraction_component_indices.update(group.get('denominator_indices', []))
            
            node = self._create_group_node(group, symbols, relations, processed)
            group_nodes[group.get('line_idx') or group.get('sqrt_idx') or 
                       group.get('operator_idx')] = node
            
            # 标记组内符号为已处理
            for key in ['numerator_indices', 'denominator_indices', 
                       'inner_indices', 'upper_limit_indices',
                       'lower_limit_indices', 'operand_indices']:
                if key in group:
                    processed.update(group[key])
        
        # 识别上标/下标符号（排除分数组件和分数线）
        script_symbols = set()
        for (i, j), rel in relations.items():
            # 跳过分数线的关系
            if i in fraction_line_indices or j in fraction_line_indices:
                continue
            # 跳过分数组件之间的关系
            if i in fraction_component_indices or j in fraction_component_indices:
                continue
            if rel in [SpatialRelation.SUPERSCRIPT, SpatialRelation.SUBSCRIPT]:
                script_symbols.add(j)
        
        # 构建主表达式
        root = SyntaxNode(NodeType.EXPRESSION.value)
        
        for symbol in sorted_symbols:
            idx = symbols.index(symbol)
            
            if idx in processed:
                continue
            
            # 如果符号是上标/下标，跳过（它会在处理基础符号时被处理）
            if idx in script_symbols:
                continue
            
            if idx in group_nodes:
                root.add_child(group_nodes[idx])
            else:
                # 创建符号节点，传递 processed 集合
                node = self._create_symbol_node(symbol, idx, symbols, relations, processed)
                root.add_child(node)
            
            processed.add(idx)
        
        return root
    
    def _create_group_node(self, group: Dict, symbols: List[Symbol],
                           relations: Dict, processed: set = None) -> SyntaxNode:
        """
        创建分组节点（分数、根号等）
        
        Args:
            group: 分组信息
            symbols: 符号列表
            relations: 空间关系字典
            processed: 已处理符号索引集合
            
        Returns:
            语法树节点
        """
        if processed is None:
            processed = set()
            
        group_type = group['type']
        
        if group_type == 'fraction':
            node = SyntaxNode(NodeType.FRACTION.value)
            
            # 按位置排序分子和分母中的符号
            num_indices = sorted(group['numerator_indices'], 
                               key=lambda i: symbols[i].bbox.x)
            den_indices = sorted(group['denominator_indices'],
                               key=lambda i: symbols[i].bbox.x)
            
            # 将分数组件（分子和分母）标记为已处理
            # 注意：分数线索引不添加到 processed，它需要在遍历时通过 group_nodes 处理
            # 但我们需要确保 _create_symbol_node 不会将分数组件误认为上下标
            line_idx = group.get('line_idx')
            all_fraction_indices = set(num_indices + den_indices)
            if line_idx is not None:
                all_fraction_indices.add(line_idx)
            
            # 创建一个本地的 processed 副本用于阻止分数内部的上下标识别
            local_processed = processed.copy()
            local_processed.update(all_fraction_indices)
            # 将分子和分母标记为已处理（分数线不标记，它需要通过 group_nodes 添加）
            for idx in num_indices + den_indices:
                processed.add(idx)
            
            # 分子
            num_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in num_indices:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations, local_processed)
                num_node.add_child(child)
            node.add_child(num_node)
            
            # 分母
            den_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in den_indices:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations, local_processed)
                den_node.add_child(child)
            node.add_child(den_node)
            
        elif group_type == 'sqrt':
            node = SyntaxNode(NodeType.SQRT.value)
            
            # 按位置排序内部符号
            inner_indices = sorted(group['inner_indices'],
                                 key=lambda i: symbols[i].bbox.x)
            
            # 内部表达式
            inner_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in inner_indices:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations, processed)
                inner_node.add_child(child)
            node.add_child(inner_node)
            
        elif group_type in ['sum', 'integral']:
            node_type = NodeType.SUM if group_type == 'sum' else NodeType.INTEGRAL
            node = SyntaxNode(node_type.value)
            
            # 下限
            lower_node = SyntaxNode(NodeType.EXPRESSION.value)
            lower_indices = sorted(group.get('lower_limit_indices', []),
                                 key=lambda i: symbols[i].bbox.x)
            for idx in lower_indices:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations, processed)
                lower_node.add_child(child)
            node.add_child(lower_node)
            
            # 上限
            upper_node = SyntaxNode(NodeType.EXPRESSION.value)
            upper_indices = sorted(group.get('upper_limit_indices', []),
                                 key=lambda i: symbols[i].bbox.x)
            for idx in upper_indices:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations, processed)
                upper_node.add_child(child)
            node.add_child(upper_node)
            
            # 被操作数
            operand_node = SyntaxNode(NodeType.EXPRESSION.value)
            operand_indices = sorted(group.get('operand_indices', []),
                                   key=lambda i: symbols[i].bbox.x)
            for idx in operand_indices:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations, processed)
                operand_node.add_child(child)
            node.add_child(operand_node)
        else:
            node = SyntaxNode(NodeType.GROUP.value)
        
        return node
    
    def _create_symbol_node(self, symbol: Symbol, idx: int,
                            symbols: List[Symbol],
                            relations: Dict,
                            processed: set = None) -> SyntaxNode:
        """
        创建符号节点
        
        Args:
            symbol: 符号
            idx: 符号索引
            symbols: 所有符号
            relations: 空间关系字典
            processed: 已处理的符号索引集合（用于避免重复处理）
            
        Returns:
            语法树节点
        """
        if processed is None:
            processed = set()
        
        # 检查是否有上标或下标
        has_superscript = False
        has_subscript = False
        superscript_indices = []
        subscript_indices = []
        
        for (i, j), rel in relations.items():
            if i == idx and j not in processed:
                if rel == SpatialRelation.SUPERSCRIPT:
                    has_superscript = True
                    superscript_indices.append(j)
                elif rel == SpatialRelation.SUBSCRIPT:
                    has_subscript = True
                    subscript_indices.append(j)
        
        # 按位置排序上标和下标
        if superscript_indices:
            superscript_indices.sort(key=lambda j: symbols[j].bbox.x)
        if subscript_indices:
            subscript_indices.sort(key=lambda j: symbols[j].bbox.x)
        
        if has_superscript or has_subscript:
            # 创建带上下标的节点
            if has_superscript and has_subscript:
                node = SyntaxNode("subsuperscript")
            elif has_superscript:
                node = SyntaxNode(NodeType.SUPERSCRIPT.value)
            else:
                node = SyntaxNode(NodeType.SUBSCRIPT.value)
            
            # 基础符号
            base_node = SyntaxNode(NodeType.SYMBOL.value, value=symbol.label)
            node.add_child(base_node)
            
            # 标记上下标符号为已处理
            for j in superscript_indices + subscript_indices:
                processed.add(j)
            
            # 下标
            if has_subscript:
                sub_node = SyntaxNode(NodeType.EXPRESSION.value)
                for j in subscript_indices:
                    # 递归处理，以支持嵌套结构
                    child = self._create_symbol_node(symbols[j], j, symbols, relations, processed)
                    sub_node.add_child(child)
                node.add_child(sub_node)
            
            # 上标
            if has_superscript:
                sup_node = SyntaxNode(NodeType.EXPRESSION.value)
                for j in superscript_indices:
                    # 递归处理，以支持嵌套结构
                    child = self._create_symbol_node(symbols[j], j, symbols, relations, processed)
                    sup_node.add_child(child)
                node.add_child(sup_node)
            
            return node
        
        # 简单符号节点
        return SyntaxNode(NodeType.SYMBOL.value, value=symbol.label)
    
    def _generate_latex(self, node: SyntaxNode, level: int = 0) -> str:
        """
        从语法树生成 LaTeX 代码
        
        Args:
            node: 语法树节点
            level: 递归深度
            
        Returns:
            LaTeX 字符串
        """
        node_type = node.node_type
        
        # 符号节点
        if node_type == NodeType.SYMBOL.value:
            return self._symbol_to_latex(node.value)
        
        # 表达式节点
        if node_type == NodeType.EXPRESSION.value:
            parts = [self._generate_latex(child, level + 1) for child in node.children]
            return ' '.join(parts)
        
        # 分数节点
        if node_type == NodeType.FRACTION.value:
            if len(node.children) >= 2:
                num = self._generate_latex(node.children[0], level + 1)
                den = self._generate_latex(node.children[1], level + 1)
                return LATEX_TEMPLATES['frac'].format(num=num, den=den)
            return ''
        
        # 根号节点
        if node_type == NodeType.SQRT.value:
            if node.children:
                inner = self._generate_latex(node.children[0], level + 1)
                return LATEX_TEMPLATES['sqrt'].format(inner=inner)
            return r'\sqrt{}'
        
        # 上标节点
        if node_type == NodeType.SUPERSCRIPT.value:
            if len(node.children) >= 2:
                base = self._generate_latex(node.children[0], level + 1)
                exp = self._generate_latex(node.children[1], level + 1)
                return LATEX_TEMPLATES['superscript'].format(base=base, exp=exp)
            return ''
        
        # 下标节点
        if node_type == NodeType.SUBSCRIPT.value:
            if len(node.children) >= 2:
                base = self._generate_latex(node.children[0], level + 1)
                sub = self._generate_latex(node.children[1], level + 1)
                return LATEX_TEMPLATES['subscript'].format(base=base, sub=sub)
            return ''
        
        # 上下标同时存在
        if node_type == "subsuperscript":
            if len(node.children) >= 3:
                base = self._generate_latex(node.children[0], level + 1)
                sub = self._generate_latex(node.children[1], level + 1)
                exp = self._generate_latex(node.children[2], level + 1)
                return LATEX_TEMPLATES['subsuperscript'].format(base=base, sub=sub, exp=exp)
            return ''
        
        # 求和节点
        if node_type == NodeType.SUM.value:
            if len(node.children) >= 3:
                lower = self._generate_latex(node.children[0], level + 1)
                upper = self._generate_latex(node.children[1], level + 1)
                operand = self._generate_latex(node.children[2], level + 1)
                return LATEX_TEMPLATES['sum'].format(lower=lower, upper=upper) + ' ' + operand
            return r'\sum'
        
        # 积分节点
        if node_type == NodeType.INTEGRAL.value:
            if len(node.children) >= 3:
                lower = self._generate_latex(node.children[0], level + 1)
                upper = self._generate_latex(node.children[1], level + 1)
                operand = self._generate_latex(node.children[2], level + 1)
                return LATEX_TEMPLATES['int'].format(lower=lower, upper=upper) + ' ' + operand
            return r'\int'
        
        # 分组节点
        if node_type == NodeType.GROUP.value:
            parts = [self._generate_latex(child, level + 1) for child in node.children]
            return ' '.join(parts)
        
        # 未知节点类型
        logger.warning(f"未知节点类型: {node_type}")
        return ''
    
    def _symbol_to_latex(self, symbol: str) -> str:
        """
        将符号转换为 LaTeX
        
        Args:
            symbol: 符号标签
            
        Returns:
            LaTeX 表示
        """
        if symbol is None:
            return ''
        
        # 检查是否在映射表中
        if symbol in SYMBOL_TO_LATEX:
            return SYMBOL_TO_LATEX[symbol]
        
        # 处理 mathXXX{Y} 格式的符号（如 mathds{N}, mathfrak{A} 等）
        import re
        math_pattern = re.match(r'^(math[a-z]+)\{(.+)\}$', symbol)
        if math_pattern:
            cmd, arg = math_pattern.groups()
            return f'\\{cmd}{{{arg}}}'
        
        # 单个字符直接返回
        if len(symbol) == 1:
            # 特殊字符需要转义
            if symbol in '#$%&_{}':
                return '\\' + symbol
            return symbol
        
        # 数字直接返回
        if symbol.isdigit():
            return symbol
        
        # 其他情况
        return symbol


class LaTeXFormatter:
    """
    LaTeX 格式化器
    
    对生成的 LaTeX 代码进行格式化和美化。
    """
    
    def format(self, latex: str) -> str:
        """
        格式化 LaTeX 代码
        
        Args:
            latex: 原始 LaTeX 代码
            
        Returns:
            格式化后的 LaTeX 代码
        """
        # 去除多余空格
        latex = ' '.join(latex.split())
        
        # 修复括号
        latex = self._fix_brackets(latex)
        
        # 优化空格
        latex = self._optimize_spacing(latex)
        
        return latex
    
    def _fix_brackets(self, latex: str) -> str:
        """修复括号匹配"""
        # 简单的括号检查和修复
        open_count = latex.count('{') - latex.count('}')
        if open_count > 0:
            latex += '}' * open_count
        elif open_count < 0:
            latex = '{' * (-open_count) + latex
        
        return latex
    
    def _optimize_spacing(self, latex: str) -> str:
        """优化空格"""
        # 运算符周围添加适当空格
        operators = ['+', '-', '=', '<', '>']
        for op in operators:
            latex = latex.replace(f' {op} ', f' {op} ')
            latex = latex.replace(f'{op} ', f' {op} ')
            latex = latex.replace(f' {op}', f' {op} ')
        
        # 去除大括号内的多余空格
        import re
        latex = re.sub(r'\{\s+', '{', latex)
        latex = re.sub(r'\s+\}', '}', latex)
        
        return latex


def analyze_structure(symbols: List[Symbol],
                      config: Optional[StructureConfig] = None) -> Tuple[SyntaxNode, str]:
    """
    分析符号结构的便捷函数
    
    Args:
        symbols: 符号列表
        config: 结构分析配置
        
    Returns:
        (语法树根节点, LaTeX 字符串)
    """
    analyzer = StructureAnalyzer(config)
    return analyzer.analyze(symbols)
