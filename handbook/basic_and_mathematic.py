# -*- coding:utf-8 -*-
"""
@Time  : 3/14/20 2:43 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : basic_and_mathematic.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter2/2.2-deep-learning-basic-mathematics.ipynb
"""

import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

"""
深度学习基础及数学原理
监督学习、无监督学习、半监督学习、强化学习是我们日常接触到的常见的四个机器学习方法：
监督学习：
    通过已有的训练样本（即已知数据以及其对应的输出）去训练得到一个最优模型（这个模型属于某个函数的集合，最优则表示在某个评价准则下是最佳的），
    再利用这个模型将所有的输入映射为相应的输出。
无监督学习：
    它与监督学习的不同之处，在于我们事先没有任何训练样本，而需要直接对数据进行建模。
半监督学习：
    在训练阶段结合了大量未标记的数据和少量标签数据。与使用所有标签数据的模型相比，使用训练集的训练模型在训练时可以更为准确。
强化学习：
    我们设定一个回报函数（reward function），通过这个函数来确认否越来越接近目标，类似我们训练宠物，如果做对了就给他奖励，做错了就给予惩罚，
    最后来达到我们的训练目的。

线性回归 （Linear Regreesion）
线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。其表达形式为y = w'x+e，
e为 误差服从 均值为0的正态分布。回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。
如果回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。

简单的说： 线性回归对于输入x与输出y有一个映射f，y=f(x),而f的形式为aX+b。其中a和b是两个可调的参数，我们训练的时候就是训练a，b这两个参数。
下面我们来用pyTorch的代码来做一个详细的解释
"""

# 下面定义一个线性函数，这里使用 y = 5x + 7，这里的5和7就是上面说到的参数a和b，我们先使用matplot可视化一下这个函数
x = np.linspace(1, 20, 500)
"""
Return evenly spaced numbers over a specified interval.
np.linspace(start, end, num)
"""
y = 5 * x + 7
plt.plot(x, y)
# plt.show()

# 下面我生成一些随机的点，来作为我们的训练数据
x = np.random.rand(256)
noise = np.random.randn(256) / 4
y = x * 5 + 7 + noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y

# 在图上显示下我们生成的数据
sns.lmplot(x='x', y='y', data=df)
# sns.scatterplot(x="total_bill", y="tip", data=tips)
# plt.show()

"""
我们随机生成了一些点，下面将使用PyTorch建立一个线性的模型来对其进行拟合，这就是所说的训练的过程，由于只有一层线性模型，所以我们就直接使用了
其中参数(1, 1)代表输入输出的特征(feature)数量都是1. Linear 模型的表达式是 y=w * x+b，其中 w 代表权重， b 代表偏置
损失函数我们使用均方损失函数：MSELoss，这个后面会详细介绍
优化器我们选择最常见的优化方法 SGD，就是每一次迭代计算 mini-batch 的梯度，然后对参数进行更新，学习率 0.01 ，优化器本章后面也会进行介绍
"""
model = Linear(1, 1)
criterion = MSELoss()
optim = SGD(model.parameters(), lr=0.01)

epochs = 3000  # 训练3000次

"""
准备训练数据: x_train, y_train 的形状是 (256, 1)， 代表 mini-batch 大小为256， feature 为1. 
astype('float32') 是为了下一步可以直接转换为 torch.float.
"""

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

# 开始训练

"""
for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    #使用模型进行预测
    outputs = model(inputs)
    #梯度置0，否则会累加
    optim.zero_grad()
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 使用优化器默认方法优化
    optim.step()
    if (i%100==0):
        #每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i,loss.data.item()))
"""

for epoch in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # 前向传播，计算预测结果
    outputs = model(inputs)
    # 在计算loss之前需要将优化器的梯度置0，否则会累加
    optim.zero_grad()
    # 计算loss
    loss = criterion(outputs, labels)
    # 误差loss反向传播
    loss.backward()
    # 使用优化器默认方法优化, 更新参数
    optim.step()
    if epoch % 100 == 0:  # 每隔100步打印一次loss
        print('epoch {}, loss {:1.4f}'.format(epoch, loss.data.item()))

"""
训练完成了，看一下训练的成果是多少。用 model.parameters() 提取模型参数。 w， b 是我们所需要训练的模型参数 
我们期望的数据 w=5，b=7 可以做一下对比
"""

w, b = model.parameters()
print(w.data.item(), b.data.item())  # 4.909689903259277 7.041140556335449

"""
再次可视化一下我们的模型，看看我们训练的数据，如果你不喜欢seaborn，可以直接使用matplot
"""

predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
plt.plot(x_train, predicted, label='predicted', alpha=1)
plt.legend()
plt.show()
