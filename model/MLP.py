# -*- coding: utf-8 -*-
# @Time : 2024/9/18 20:57
# @Author : CSR
# @File : MLP.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X = torch.randn(1000, 20)  # 100个样本，每个样本20个特征
Y = (X.sum(dim=1) > 10).float().unsqueeze(1) # 1表示正样本，0表示负样本
train_x, test_x = X[:800], X[800:]
train_y, test_y = Y[:800], Y[800:]

train_data = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model = MLP(input_dim=20, hidden_dim=16, output_dim=1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        # 用于根据计算得到的梯度更新模型的参数。
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

with torch.no_grad():
    test_outputs = model(test_x)
    test_loss = criterion(test_outputs, test_y)
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Accuracy: {(test_outputs.round() == test_y).sum().item() / len(test_y):.4f}")


