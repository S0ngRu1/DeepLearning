# -*- coding: utf-8 -*-
# @Time : 2024/10/21 19:44
# @Author : CSR
# @File : display_MMIST.py

# Imports we need
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define a transformation for the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load the MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)

# Get a batch of data
# 将 train_loader（一个 DataLoader 对象）转换为一个迭代器 data_iter。迭代器允许你逐步访问数据。
data_iter = iter(train_loader)
# 从迭代器中获取下一批数据。在这里，images 是一个张量，包含一批图像数据，labels 是一个张量，包含对应的标签。
images, labels = next(data_iter)

# Convert the image tensor to numpy for plotting
image = images[0].numpy()[0]  # Get the first image from the batch
label = labels.item()  # Get the label

# Plot the image
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')  # Hide axes
plt.show()
