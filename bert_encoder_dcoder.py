# -*- coding: utf-8 -*-
# @Time : 2024/10/21 20:12
# @Author : CSR
# @File : bert_encoder_dcoder.py

import os
import torch
from transformers import BertTokenizer

# 设置数据目录
data_dir = "./data"  # 本地存储目录

# 下载词汇表文件（假设你已经有了词汇表）
vocab_name = "vocab.translate_ende_wmt32k.32768.subwords"  # 替换为实际词汇表文件名
vocab_file = os.path.join(data_dir, vocab_name)

# 加载 BERT 分词器（可以根据需要使用其他模型）
tokenizer = BertTokenizer.from_pretrained("pretrained/bert/cased_L-12_H-768_A-12")  # 或者使用自定义词汇表

# 定义编码和解码的辅助函数
def encode(input_str, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = tokenizer.encode(input_str, add_special_tokens=True)  # add EOS id
    inputs_tensor = torch.tensor(inputs).unsqueeze(0)  # Make it 2D tensor (batch_size, sequence_length)
    return {"inputs": inputs_tensor}

def decode(integers):
    """List of ints to str"""
    integers = list(integers.squeeze().numpy())  # 转为列表
    if 1 in integers:  # 假设 1 是 EOS token 的 ID
        integers = integers[:integers.index(1)]
    return tokenizer.decode(integers)

# 测试编码和解码
input_str = "Hello, how are you?"
encoded = encode(input_str)
print("Encoded:", encoded)

decoded = decode(encoded["inputs"])
print("Decoded:", decoded)
