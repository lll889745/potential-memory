# 手写数学公式识别系统

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个基于传统图像处理和机器学习方法的手写数学公式识别系统，能够将手写数学公式图像转换为 LaTeX 代码。

## 环境要求

- Python 3.12+
- OpenCV 4.x
- NumPy, SciPy, scikit-learn, scikit-image
- matplotlib
- Pillow

## 项目结构

```
LaTex/
├── src/                          # 核心源代码
│   ├── __init__.py              # 模块初始化
│   ├── config.py                # 配置文件（符号映射、阈值参数等）
│   ├── utils.py                 # 工具函数
│   ├── preprocessing.py         # 图像预处理模块
│   ├── segmentation.py          # 符号分割模块
│   ├── recognition.py           # 符号识别模块
│   ├── structure_analysis.py    # 结构分析与 LaTeX 生成模块
│   └── semantic.py              # 语义理解模块
├── tests/                        # 单元测试
│   └── test_modules.py
├── examples/                     # 示例代码
│   └── demo_usage.py
├── scripts/                      # 工具脚本
│   ├── create_test_images.py    # 生成测试图片
│   ├── create_integral_sum_tests.py
│   └── debug_*.py               # 调试脚本
├── models/                       # 模型文件
│   └── *.pkl                    # 预训练模型
├── data/                         # 数据目录
│   ├── training/                # 训练数据
│   └── test/                    # 测试图像
├── data_raw/                     # 原始数据集
│   └── HASYv2/                  # HASYv2 原始数据
├── build_config/                 # 构建配置
│   ├── build_gui.spec           # GUI 打包配置
│   └── build_exe.spec           # 命令行打包配置
├── dist/                         # 打包输出
│   └── 公式识别系统.exe         # Windows 可执行程序
├── gui.py                        # GUI 主程序
├── main.py                       # 命令行主程序
├── train_model.py                # 模型训练脚本
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 模块详解

### 1. 图像预处理 (preprocessing.py)

实现自适应二值化（Sauvola 方法）、去噪、倾斜校正和骨架提取。

```python
from src.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()

# 完整预处理
binary = preprocessor.process(image)

# 获取中间结果
result = preprocessor.process(image, return_intermediate=True)
# result['binary']    - 二值化结果
# result['denoised']  - 去噪结果
# result['corrected'] - 倾斜校正结果
# result['skeleton']  - 骨架图像
```

### 2. 符号分割 (segmentation.py)

将预处理后的图像分割为独立符号，处理粘连和分离的符号，检测分数线结构。

```python
from src.segmentation import SymbolSegmenter

segmenter = SymbolSegmenter()
symbols = segmenter.segment(binary)

# 每个 symbol 包含：
# - bbox: 边界框 (x, y, width, height)
# - contour: 轮廓点
# - is_fraction_line: 是否为分数线
```

### 3. 符号识别 (recognition.py)

使用 HOG、Zernike 矩等特征和随机森林分类器进行符号识别。

```python
from src.recognition import SymbolRecognizer

# 加载预训练模型
recognizer = SymbolRecognizer()
recognizer.load_model('models/model_2025_12_19_19_05.pkl')

# 识别符号
symbols = recognizer.recognize_symbols(symbols)
# 每个 symbol 新增 label 和 confidence 属性
```

### 4. 结构分析 (structure_analysis.py)

分析符号间的空间关系（上标、下标、分数等），构建语法树，生成 LaTeX 代码。

```python
from src.structure_analysis import StructureAnalyzer

analyzer = StructureAnalyzer()
syntax_tree, latex = analyzer.analyze(symbols)

# 支持的结构：
# - 分数 (\frac{}{})
# - 上标 (^{})
# - 下标 (_{})
# - 积分 (\int)、求和 (\sum) 带上下限
# - 根号 (\sqrt{})
```

## 模型训练

如需重新训练模型：

```bash
python train_model.py
```

训练数据应放置在 `data/training/` 目录下，按符号类别分文件夹存放（参考 HASYv2 数据集格式）。

## 支持的符号

系统支持 370+ 种数学符号，包括：

- **数字**：0-9
- **拉丁字母**：a-z, A-Z
- **希腊字母**：α, β, γ, δ, θ, λ, μ, π, σ, φ, ω, Γ, Δ, Σ, Ω 等
- **运算符**：+, -, ×, ÷, =, ≠, <, >, ≤, ≥, ±, ∓
- **关系符号**：≈, ≡, ∈, ⊂, ⊃, ∪, ∩
- **特殊符号**：√, ∑, ∏, ∫, ∞, ∂, ∇, ℵ
- **括号**：(), [], {}, ⟨⟩
- **箭头**：→, ←, ↔, ⇒, ⇐, ⇔
- **特殊字体**：\mathbb, \mathcal, \mathfrak, \mathds 等

完整符号映射见 `src/config.py` 中的 `SYMBOL_TO_LATEX` 字典。

## 配置说明

在 `src/config.py` 中可以调整各模块的参数：

```python
from src.config import StructureConfig

# 结构分析参数
config = StructureConfig()
config.superscript_y_threshold = 0.25  # 上标检测阈值
config.subscript_y_threshold = 0.25    # 下标检测阈值
config.script_size_ratio = 0.75        # 上下标尺寸比例
config.fraction_gap_threshold = 0.3    # 分数间距阈值
```

## 运行测试

```bash
python -m pytest tests/test_modules.py -v
```

## 许可证

MIT License

## 参考资料

- [HASYv2 数据集](https://zenodo.org/record/259444) - 训练数据来源
- [OpenCV 文档](https://docs.opencv.org/)
- [scikit-learn 文档](https://scikit-learn.org/)
- [matplotlib LaTeX 渲染](https://matplotlib.org/stable/tutorials/text/mathtext.html)
