# -*- coding: utf-8 -*-
# @Time : 2024/10/20 16:38
# @Author : CSR
# @File : test_attention.py

import matplotlib.pyplot as plt
import numpy as np

# 假设Q、K、V是2D numpy数组，维度为(batch_size, sequence_length)
Q = np.random.rand(3, 12)
K = np.random.rand(3, 12)
V = np.random.rand(3, 12)

# 计算注意力权重矩阵
attention_scores = np.matmul(Q, K.T) / np.sqrt(np.sum(K**2, axis=1, keepdims=True))
output = np.matmul(attention_scores, V)

# 可视化注意力权重矩阵
plt.imshow(output, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()