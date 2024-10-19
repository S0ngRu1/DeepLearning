# -*- coding: utf-8 -*-
# @Time : 2024/9/20 20:21
# @Author : CSR
# @File : NeuralNetwork.py
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, loss_func='mse'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss_func = loss_func

        self.weights1 = np.random.randn(self.hidden_size, self.input_size)
        self.bias1 = np.zeros((self.hidden_size, 1))
        self.weights2 = np.random.randn(self.output_size, self.hidden_size)
        self.bias2 = np.zeros((self.output_size, 1))

        self.train_loss = []
        self.test_loss = []

    def forward(self, X):
        self.z1 = np.dot(self.weights1, X) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.weights2, self.a1) + self.bias2
        if self.loss_func == 'categorical_crossentropy':
            self.a2 = self.softmax(self.z2)
        else:
            self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate=0.01):
        if self.loss_func == 'categorical_crossentropy':
            delta2 = self.a2 - y
        elif self.loss_func == 'mse':
            delta2 = 2 * (self.a2 - y)
        else:
            raise ValueError('Invalid loss function')
        # 计算第二层权重的梯度
        self.dw2 = np.dot(delta2, self.a1.T)
        # 计算第二层偏置的梯度
        self.db2 = np.sum(delta2, axis=1, keepdims=True)
        # 计算第一层激活值的梯度
        self.dz1 = np.dot(self.weights2.T, delta2)
        # 计算第一层权重的梯度
        self.dw1 = np.dot(self.dz1 * self.sigmoid_derivative(self.z1), X.T)
        # 计算第一层偏置的梯度
        self.db1 = np.sum(self.dz1 * self.sigmoid_derivative(self.z1), axis=1, keepdims=True)
        # Update weights and biases
        self.weights1 -= learning_rate * self.dw1
        self.bias1 -= learning_rate * self.db1
        self.weights2 -= learning_rate * self.dw2
        self.bias2 -= learning_rate * self.db2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)



