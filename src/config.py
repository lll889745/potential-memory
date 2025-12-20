"""
配置文件
========

包含系统的所有可配置参数。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os


@dataclass
class PreprocessingConfig:
    """图像预处理配置"""
    
    # Sauvola 二值化参数
    sauvola_window_size: int = 25  # 局部窗口大小（奇数）
    sauvola_k: float = 0.2  # Sauvola 参数 k
    sauvola_r: float = 128  # 动态范围参数 R
    
    # 去噪参数
    denoise_kernel_size: int = 3  # 形态学去噪核大小
    min_component_area: int = 10  # 最小连通域面积（像素）
    
    # 倾斜校正参数
    hough_threshold: int = 100  # 霍夫变换阈值
    max_skew_angle: float = 15.0  # 最大允许倾斜角度（度）
    
    # 图像尺寸标准化
    target_height: int = 64  # 标准化高度


@dataclass
class SegmentationConfig:
    """符号分割配置"""
    
    # 连通域分析
    connectivity: int = 8  # 连通性（4或8）
    
    # 组件合并参数
    merge_distance_threshold: float = 0.3  # 合并距离阈值（相对于符号高度）
    dot_size_ratio: float = 0.25  # 点的尺寸比例阈值
    
    # 分数线检测参数
    fraction_line_aspect_ratio: float = 5.0  # 分数线最小宽高比
    fraction_line_height_ratio: float = 0.1  # 分数线最大高度比例
    
    # 根号检测参数
    sqrt_hook_ratio: float = 0.3  # 根号钩的宽度比例


@dataclass
class RecognitionConfig:
    """符号识别配置"""
    
    # 图像标准化
    symbol_size: Tuple[int, int] = (32, 32)  # 符号图像大小
    
    # 特征提取
    hog_orientations: int = 9  # HOG 方向数
    hog_pixels_per_cell: Tuple[int, int] = (8, 8)  # HOG 单元格大小
    hog_cells_per_block: Tuple[int, int] = (2, 2)  # HOG 块大小
    
    # 网格特征
    grid_size: Tuple[int, int] = (4, 4)  # 网格划分
    
    # 分类器参数
    svm_c: float = 10.0  # SVM 正则化参数
    svm_gamma: str = 'scale'  # SVM 核参数
    rf_n_estimators: int = 100  # 随机森林树数量
    knn_n_neighbors: int = 5  # k-NN 邻居数
    
    # 置信度阈值
    confidence_threshold: float = 0.7  # 最低置信度阈值
    top_k_candidates: int = 3  # 返回的候选数量


@dataclass
class StructureConfig:
    """结构分析配置"""
    
    # 空间关系判定阈值
    superscript_y_threshold: float = 0.25  # 上标 y 偏移阈值（提高以减少误检）
    subscript_y_threshold: float = 0.25  # 下标 y 偏移阈值（提高以减少误检）
    script_size_ratio: float = 0.75  # 上下标尺寸比例阈值（降低要求上下标必须明显偏小）
    vertical_overlap_threshold: float = 0.4  # 垂直重叠阈值
    
    # 上下标水平距离限制
    script_max_horizontal_distance: float = 2.0  # 上下标最大水平距离（相对于基础符号宽度）
    
    # 分数识别
    fraction_gap_threshold: float = 0.3  # 分数上下间距阈值
    fraction_line_vertical_tolerance: float = 2.0  # 分数线垂直关联容差
    
    # 根号识别
    sqrt_coverage_threshold: float = 0.8  # 根号覆盖范围阈值


@dataclass
class SemanticConfig:
    """语义理解配置"""
    
    # 计算超时
    computation_timeout: int = 30  # 计算超时时间（秒）
    
    # 简化选项
    simplify_trigonometric: bool = True  # 简化三角函数
    simplify_radicals: bool = True  # 简化根式
    
    # 错误检测
    check_division_by_zero: bool = True
    check_bracket_balance: bool = True
    check_undefined_symbols: bool = True


# 符号到 LaTeX 映射表
SYMBOL_TO_LATEX: Dict[str, str] = {
    # 希腊字母（小写）
    'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
    'delta': r'\delta', 'epsilon': r'\epsilon', 'zeta': r'\zeta',
    'eta': r'\eta', 'theta': r'\theta', 'iota': r'\iota',
    'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu',
    'nu': r'\nu', 'xi': r'\xi', 'omicron': r'o',
    'pi': r'\pi', 'rho': r'\rho', 'sigma': r'\sigma',
    'tau': r'\tau', 'upsilon': r'\upsilon', 'phi': r'\phi',
    'chi': r'\chi', 'psi': r'\psi', 'omega': r'\omega',
    
    # 希腊字母（大写）
    'Alpha': r'A', 'Beta': r'B', 'Gamma': r'\Gamma',
    'Delta': r'\Delta', 'Epsilon': r'E', 'Zeta': r'Z',
    'Eta': r'H', 'Theta': r'\Theta', 'Iota': r'I',
    'Kappa': r'K', 'Lambda': r'\Lambda', 'Mu': r'M',
    'Nu': r'N', 'Xi': r'\Xi', 'Omicron': r'O',
    'Pi': r'\Pi', 'Rho': r'P', 'Sigma': r'\Sigma',
    'Tau': r'T', 'Upsilon': r'\Upsilon', 'Phi': r'\Phi',
    'Chi': r'X', 'Psi': r'\Psi', 'Omega': r'\Omega',
    
    # 运算符
    'plus': '+', 'minus': '-', 'times': r'\times',
    'divide': r'\div', 'div': r'\div', 'pm': r'\pm', 'mp': r'\mp',
    'cdot': r'\cdot', 'ast': r'\ast', 'circ': r'\circ',
    'bullet': r'\bullet', 'star': r'\star',
    'oplus': r'\oplus', 'ominus': r'\ominus', 'otimes': r'\otimes',
    'odot': r'\odot', 'oslash': r'\oslash',
    'ltimes': r'\ltimes', 'rtimes': r'\rtimes',
    'boxplus': r'\boxplus', 'boxtimes': r'\boxtimes', 'boxdot': r'\boxdot',
    
    # 关系符号
    'eq': '=', 'neq': r'\neq', 'lt': '<', 'gt': '>',
    'less_than': '<', 'greater_than': '>',
    'leq': r'\leq', 'geq': r'\geq', 'approx': r'\approx',
    'leqslant': r'\leqslant', 'geqslant': r'\geqslant',
    'lesssim': r'\lesssim', 'gtrsim': r'\gtrsim',
    'equiv': r'\equiv', 'sim': r'\sim', 'simeq': r'\simeq',
    'cong': r'\cong', 'doteq': r'\doteq',
    'prec': r'\prec', 'succ': r'\succ',
    'preceq': r'\preceq', 'succeq': r'\succeq',
    'll': r'\ll', 'gg': r'\gg',
    'gtrless': r'\gtrless', 'lessgtr': r'\lessgtr',
    
    # 逻辑符号
    'forall': r'\forall', 'exists': r'\exists',
    'neg': r'\neg', 'lnot': r'\lnot',
    'land': r'\land', 'lor': r'\lor',
    'wedge': r'\wedge', 'vee': r'\vee',
    'implies': r'\implies', 'iff': r'\iff',
    'top': r'\top', 'bot': r'\bot',
    'therefore': r'\therefore', 'because': r'\because',
    
    # 特殊符号
    'sqrt': r'\sqrt', 'sum': r'\sum', 'prod': r'\prod',
    'coprod': r'\coprod',
    'int': r'\int', 'oint': r'\oint', 'iint': r'\iint', 'iiint': r'\iiint',
    'infty': r'\infty', 'partial': r'\partial', 'nabla': r'\nabla',
    'hbar': r'\hbar', 'ell': r'\ell', 'wp': r'\wp', 'Re': r'\Re', 'Im': r'\Im',
    'aleph': r'\aleph', 'beth': r'\beth',
    'degree': r'^\circ', 'celsius': r'^\circ\text{C}',
    'prime': r"'", 'dprime': r"''",
    'dag': r'\dag', 'ddag': r'\ddag',
    
    # 数集符号（黑板粗体 mathbb）
    'mathbb{N}': r'\mathbb{N}', 'mathbb{Z}': r'\mathbb{Z}',
    'mathbb{Q}': r'\mathbb{Q}', 'mathbb{R}': r'\mathbb{R}',
    'mathbb{C}': r'\mathbb{C}', 'mathbb{H}': r'\mathbb{H}',
    'mathbb{1}': r'\mathbb{1}',
    
    # 双线体 (mathds / dsfont)
    'mathds{1}': r'\mathds{1}', 'mathds{C}': r'\mathds{C}',
    'mathds{E}': r'\mathds{E}', 'mathds{N}': r'\mathds{N}',
    'mathds{P}': r'\mathds{P}', 'mathds{Q}': r'\mathds{Q}',
    'mathds{R}': r'\mathds{R}', 'mathds{Z}': r'\mathds{Z}',
    
    # 花体字母 (mathcal)
    'mathcal{A}': r'\mathcal{A}', 'mathcal{B}': r'\mathcal{B}',
    'mathcal{C}': r'\mathcal{C}', 'mathcal{D}': r'\mathcal{D}',
    'mathcal{E}': r'\mathcal{E}', 'mathcal{F}': r'\mathcal{F}',
    'mathcal{G}': r'\mathcal{G}', 'mathcal{H}': r'\mathcal{H}',
    'mathcal{L}': r'\mathcal{L}', 'mathcal{M}': r'\mathcal{M}',
    'mathcal{N}': r'\mathcal{N}', 'mathcal{O}': r'\mathcal{O}',
    'mathcal{P}': r'\mathcal{P}', 'mathcal{R}': r'\mathcal{R}',
    'mathcal{S}': r'\mathcal{S}', 'mathcal{T}': r'\mathcal{T}',
    'mathcal{U}': r'\mathcal{U}', 'mathcal{X}': r'\mathcal{X}',
    'mathcal{Z}': r'\mathcal{Z}',
    
    # 哥特体 (mathfrak)
    'mathfrak{A}': r'\mathfrak{A}', 'mathfrak{M}': r'\mathfrak{M}',
    'mathfrak{S}': r'\mathfrak{S}', 'mathfrak{X}': r'\mathfrak{X}',
    
    # 手写体 (mathscr)
    'mathscr{A}': r'\mathscr{A}', 'mathscr{C}': r'\mathscr{C}',
    'mathscr{D}': r'\mathscr{D}', 'mathscr{E}': r'\mathscr{E}',
    'mathscr{F}': r'\mathscr{F}', 'mathscr{H}': r'\mathscr{H}',
    'mathscr{L}': r'\mathscr{L}', 'mathscr{P}': r'\mathscr{P}',
    'mathscr{S}': r'\mathscr{S}',
    
    # 集合符号
    'in': r'\in', 'notin': r'\notin', 'ni': r'\ni',
    'subset': r'\subset', 'supset': r'\supset',
    'subseteq': r'\subseteq', 'supseteq': r'\supseteq',
    'subsetneq': r'\subsetneq', 'supsetneq': r'\supsetneq',
    'sqsubset': r'\sqsubset', 'sqsupset': r'\sqsupset',
    'sqsubseteq': r'\sqsubseteq', 'sqsupseteq': r'\sqsupseteq',
    'cup': r'\cup', 'cap': r'\cap',
    'bigcup': r'\bigcup', 'bigcap': r'\bigcap',
    'uplus': r'\uplus', 'sqcup': r'\sqcup',
    'emptyset': r'\emptyset', 'varnothing': r'\varnothing',
    'setminus': r'\setminus',
    
    # 箭头
    'rightarrow': r'\rightarrow', 'leftarrow': r'\leftarrow',
    'leftrightarrow': r'\leftrightarrow',
    'Rightarrow': r'\Rightarrow', 'Leftarrow': r'\Leftarrow',
    'Leftrightarrow': r'\Leftrightarrow',
    'longrightarrow': r'\longrightarrow', 'longleftarrow': r'\longleftarrow',
    'Longrightarrow': r'\Longrightarrow', 'Longleftarrow': r'\Longleftarrow',
    'Longleftrightarrow': r'\Longleftrightarrow',
    'mapsto': r'\mapsto', 'longmapsto': r'\longmapsto',
    'hookrightarrow': r'\hookrightarrow', 'hookleftarrow': r'\hookleftarrow',
    'uparrow': r'\uparrow', 'downarrow': r'\downarrow',
    'Uparrow': r'\Uparrow', 'Downarrow': r'\Downarrow',
    'updownarrow': r'\updownarrow', 'Updownarrow': r'\Updownarrow',
    'nearrow': r'\nearrow', 'searrow': r'\searrow',
    'nwarrow': r'\nwarrow', 'swarrow': r'\swarrow',
    'leadsto': r'\leadsto', 'curvearrowright': r'\curvearrowright',
    'circlearrowleft': r'\circlearrowleft', 'circlearrowright': r'\circlearrowright',
    
    # 点号
    'dots': r'\dots', 'cdots': r'\cdots', 'ldots': r'\ldots',
    'vdots': r'\vdots', 'ddots': r'\ddots', 'iddots': r'\iddots',
    'dotsc': r'\dotsc',
    
    # 括号
    'lparen': '(', 'rparen': ')', 'paren_open': '(', 'paren_close': ')',
    'lbracket': '[', 'rbracket': ']', 'bracket_open': '[', 'bracket_close': ']',
    'lbrace': r'\{', 'rbrace': r'\}', 'brace_open': r'\{', 'brace_close': r'\}',
    'langle': r'\langle', 'rangle': r'\rangle', 'angle': r'\angle',
    'lfloor': r'\lfloor', 'rfloor': r'\rfloor',
    'lceil': r'\lceil', 'rceil': r'\rceil',
    'llbracket': r'\llbracket', 'rrbracket': r'\rrbracket',
    'backslash': r'\backslash', 'forward_slash': '/',
    'double_vertical_bar': r'\|', 'vertical_bar': '|',
    
    # 杂项符号
    'checkmark': r'\checkmark', 'checked': r'\checkmark',
    'lightning': r'\lightning',
    'diamond': r'\diamond', 'diamondsuit': r'\diamondsuit',
    'heartsuit': r'\heartsuit', 'clubsuit': r'\clubsuit', 'spadesuit': r'\spadesuit',
    'flat': r'\flat', 'natural': r'\natural', 'sharp': r'\sharp',
    'lozenge': r'\lozenge', 'blacksquare': r'\blacksquare', 'square': r'\square',
    'blacktriangleright': r'\blacktriangleright',
    'triangleright': r'\triangleright', 'triangleleft': r'\triangleleft',
    'bowtie': r'\bowtie', 'asymp': r'\asymp',
    'between': r'\between', 'smile': r'\smile', 'frown': r'\frown',
    'vdash': r'\vdash', 'dashv': r'\dashv',
    'models': r'\models', 'perp': r'\perp', 'parallel': r'\parallel',
    'circledast': r'\circledast', 'circledcirc': r'\circledcirc',
    'circledR': r'\circledR', 'copyright': r'\copyright',
    'dollar': r'\$', 'hash': r'\#', 'ampersand': r'\&', 'percent': r'\%',
    'male': r'\male', 'female': r'\female', 'mars': r'\mars',
    'fullmoon': r'\fullmoon', 'leftmoon': r'\leftmoon', 'astrosun': r'\astrosun',
    'amalg': r'\amalg', 'barwedge': r'\barwedge', 'backsim': r'\backsim',
    
    # 其他
    'comma': ',', 'period': '.', 'colon': ':',
    'semicolon': ';', 'exclaim': '!', 'question': '?',
}

# LaTeX 模板
LATEX_TEMPLATES: Dict[str, str] = {
    'frac': r'\frac{{{num}}}{{{den}}}',
    'sqrt': r'\sqrt{{{inner}}}',
    'sqrt_n': r'\sqrt[{index}]{{{inner}}}',
    'superscript': r'{{{base}}}^{{{exp}}}',
    'subscript': r'{{{base}}}_{{{sub}}}',
    'subsuperscript': r'{{{base}}}_{{{sub}}}^{{{exp}}}',
    'sum': r'\sum_{{{lower}}}^{{{upper}}}',
    'sum_no_limits': r'\sum',
    'prod': r'\prod_{{{lower}}}^{{{upper}}}',
    'int': r'\int_{{{lower}}}^{{{upper}}}',
    'int_no_limits': r'\int',
    'lim': r'\lim_{{{var} \to {target}}}',
    'matrix': r'\begin{{pmatrix}} {content} \end{{pmatrix}}',
}

# 符号类别
SYMBOL_CATEGORIES: Dict[str, List[str]] = {
    'digits': list('0123456789'),
    'lowercase': list('abcdefghijklmnopqrstuvwxyz'),
    'uppercase': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
    'greek_lower': ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta',
                    'lambda', 'mu', 'pi', 'sigma', 'phi', 'omega'],
    'greek_upper': ['Gamma', 'Delta', 'Theta', 'Lambda', 'Pi', 'Sigma',
                    'Phi', 'Psi', 'Omega'],
    'operators': ['plus', 'minus', 'times', 'divide', 'pm', 'cdot'],
    'relations': ['eq', 'neq', 'lt', 'gt', 'leq', 'geq'],
    'special': ['sqrt', 'sum', 'prod', 'int', 'infty', 'partial'],
    'brackets': ['lparen', 'rparen', 'lbracket', 'rbracket', 'lbrace', 'rbrace'],
}

# 易混淆符号对
CONFUSABLE_SYMBOLS: List[Tuple[str, str]] = [
    ('0', 'O'), ('0', 'o'),
    ('1', 'l'), ('1', 'I'), ('1', '|'),
    ('2', 'Z'), ('2', 'z'),
    ('5', 'S'), ('5', 's'),
    ('6', 'b'), ('6', 'G'),
    ('8', 'B'),
    ('x', 'times'),
    ('minus', 'line'),  # 减号与分数线
]


@dataclass
class SystemConfig:
    """系统总配置"""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    
    # 模型路径
    model_dir: str = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # 日志配置
    log_level: str = 'INFO'
    log_file: str = 'formula_recognition.log'


# 默认配置实例
DEFAULT_CONFIG = SystemConfig()
