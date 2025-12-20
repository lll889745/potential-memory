# -*- mode: python ; coding: utf-8 -*-
"""
LaTeX公式识别系统打包配置
"""

import os
import sys

block_cipher = None

# 获取当前目录
CURR_DIR = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['main.py'],
    pathex=[CURR_DIR],
    binaries=[],
    datas=[
        # 包含模型文件
        ('models/model_2025_12_19_19_05.pkl', 'models'),
        # 包含示例测试图像
        ('data/test/*.png', 'data/test'),
    ],
    hiddenimports=[
        'sklearn',
        'sklearn.svm',
        'sklearn.ensemble',
        'sklearn.neighbors',
        'sklearn.preprocessing',
        'sklearn.model_selection',
        'sklearn.metrics',
        'sklearn.utils._cython_blas',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors._partition_nodes',
        'cv2',
        'numpy',
        'sympy',
        'sympy.parsing',
        'sympy.parsing.latex',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LaTeX_Formula_Recognizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 控制台程序
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标: icon='icon.ico'
)
