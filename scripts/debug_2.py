import cv2
import numpy as np

# 可视化训练数据"A"
train = cv2.imread('data/test/debug_A_training.png', 0)
print('Training A (32x32):')
for r in range(32):
    row = ''
    for c in range(32):
        row += '.' if train[r,c] == 0 else '#'  # 黑色笔画用'.'
    print(row)

print()
print('=' * 40)
print()

# 可视化规范化后的分割A
normalized = cv2.imread('data/test/debug_A_normalized.png', 0)
print('Normalized segmented A (32x32):')
for r in range(32):
    row = ''
    for c in range(32):
        row += '.' if normalized[r,c] == 0 else '#'
    print(row)
