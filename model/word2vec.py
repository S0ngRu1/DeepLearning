# -*- coding: utf-8 -*-
# @Time : 2024/10/19 23:22
# @Author : CSR
# @File : word2vec.py

import numpy as np

class Word2Vec:
    def __init__(self, vocabulary, window_size):
        self.vocabulary = vocabulary
        self.window_size = window_size
        self.word_vectors = {}
        self.build_word_vectors()

# 构建词向量
def build_word_vectors(self):
    word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
    vocab_size = len(vocabulary)

    self.word_vectors = np.random.randn(vocab_size, vocab_size) / np.sqrt(vocab_size)

# 训练
def train(self, learning_rate=0.025, num_epochs=5, min_count=5):
    window_size = self.window_size
    vocab_size = len(self.vocabulary)
    word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
    context_size = 2 * window_size + 1

    # 初始化输入矩阵和输出矩阵
    X = np.zeros((vocab_size, context_size, vocab_size))
    Y = np.zeros((vocab_size, context_size, vocab_size))

    for center_word in self.vocabulary:
        if center_word in word_to_index:
            index = word_to_index[center_word]

            for context_word in self.get_context_words(center_word):
                if context_word in word_to_index:
                    context_index = word_to_index[context_word]

                    for i in range(-window_size, window_size + 1):
                        if i == 0 or i == window_size:
                            continue

                        context_index_adj = window_size + i
                        if context_index_adj >= 0 and context_index_adj < context_size:
                            X[index, context_index_adj, :] = 1
                            Y[index, context_index_adj, :] = 1

            self.update_word_vectors(index, learning_rate)

    # 更新词向量
    def update_word_vectors(self, center_index, learning_rate=0.025):
        X_center = X[center_index]
        Y_center = Y[center_index]

        for context_index, context_vector in enumerate(X_center):
            self.word_vectors[center_index] += learning_rate * np.outer(context_vector, Y_center[context_index])

# 获取上下文词
def get_context_words(self, center_word):
    context_words = []
    for i in range(-self.window_size, self.window_size + 1):
        if i == 0 or i == self.window_size:  # 跳过中心词自身
            continue
        context_words.append(self.vocabulary[(i + self.window_size + 1) % (2 * self.window_size + 1)])
    return context_words

