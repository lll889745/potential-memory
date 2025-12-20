# 模型存储目录

本目录存放符号识别模型文件（.pkl 格式）。

## 当前模型

- **model_2025_12_19_19_05.pkl** - 预训练的随机森林分类器
  - 基于 HASYv2 数据集训练
  - 支持 370+ 种数学符号
  - 使用 HOG、Zernike 矩等特征

## 使用方法

加载模型进行识别：
```python
from src.recognition import SymbolRecognizer

recognizer = SymbolRecognizer()
recognizer.load_model('models/model_2025_12_19_19_05.pkl')

# 识别符号
symbols = recognizer.recognize_symbols(segmented_symbols)
```

## 重新训练

如需重新训练模型：

```bash
python train_model.py
```

训练数据应放置在 `data/training/` 目录下，按符号类别分文件夹存放。

新模型将自动保存为 `models/model_YYYY_MM_DD_HH_MM.pkl` 格式。
