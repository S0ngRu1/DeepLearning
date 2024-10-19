# -*- coding: utf-8 -*-
# @Time : 2024/10/19 20:17
# @Author : CSR
# @File : test_one_hot.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 示例中文文本
texts = ['我爱北京天安门', '天安门上太阳升', '伟大领袖毛主席', '指引我们向前进']

# 将中文文本转换为列表形式，每个元素是一个字
text_list = list(''.join(texts))
text_set = set(text_list)
print(len(text_set))
# 创建DataFrame
df = pd.DataFrame(text_list, columns=['text'])

# 初始化OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# 对文本进行One-hot编码
# 由于One-hot编码通常用于分类变量，我们需要将文本转换为分类变量
# 这里我们使用get_dummies函数将每个字转换为一个二进制特征
one_hot_encoded = pd.get_dummies(df['text'].astype('category'))

print(one_hot_encoded)