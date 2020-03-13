# -*- coding:utf-8 -*-
"""
@Time  : 3/12/20 5:03 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : nn_optim.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter2/2.1.3-pytorch-basics-nerual-network.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""神经网络包nn和优化器optim
torch.nn是专门为神经网络设计的模块化接口。nn构建于 Autograd之上，可用来定义和运行神经网络。
约定： torch.nn 我们为了方便使用，会为他设置别名为nn
除了nn别名以外，我们还引用了nn.functional，这个包中包含了神经网络中使用的一些常用函数，这些
函数的特点是，不具有可学习的参数(如ReLU，pool，DropOut等)，这些函数可以放在构造函数中，也可
以不放，但是这里建议不放。
一般情况下我们会将nn.functional 设置为大写的F，这样缩写方便调用
"""

# ------------------------- 1.定义一个网络 -------------------------
"""
PyTorch中已经为我们准备好了现成的网络模型，只要继承nn.Module，并实现它的forward方法，
PyTorch会根据autograd，自动实现backward函数，在forward函数中可使用任何tensor支持的函数，
还可以使用if、for循环、print、log等Python语法，写法和标准的Python写法一致。
"""


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道， '6'表示输出通道数，'3'表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 线性层，输入1350个特征，输出10个特征
        self.fc1 = nn.Linear(1350, 10)  # 这里的1350是如何计算的呢？这就要看后面的forward函数

    # 正向传播
    def forward(self, x):
        # print(x.size())  # 结果：[1, 1, 32, 32]
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x)  # 根据卷积的尺寸计算公式，计算结果是30，具体计算公式后面第二章第四节 卷积神经网络 有详细介绍。
        x = F.relu(x)
        # print(x.size())  # 结果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2))  # 我们使用池化层，计算结果是15
        x = F.relu(x)
        # print(x.size())  # 结果：[1, 6, 15, 15]
        # reshape，‘-1’表示自适应
        # 这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1)
        # print(x.size())  # 这里就是fc1层的的输入1350, torch.Size([1, 1350])
        x = self.fc1(x)
        return x


net = Net()
"""
>>> print(net)
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=1350, out_features=10, bias=True)
)
"""

"""
# 网络的可学习参数通过net.parameters()返回
for parameter in net.parameters():
    print(parameter)
Parameter containing:
tensor([[[[ 0.0599,  0.1069, -0.2572],
          [ 0.2024, -0.3314, -0.3130],
          [-0.3284,  0.1664, -0.2119]]],


        [[[-0.0287, -0.3178, -0.2398],
          [ 0.0460,  0.0305, -0.1894],
          [-0.1000, -0.3187, -0.0902]]],


        [[[ 0.3172,  0.2797,  0.0717],
          [-0.3321, -0.0825, -0.1000],
          [ 0.0464,  0.3328,  0.0810]]],


        [[[ 0.2170,  0.0275,  0.1652],
          [-0.0496, -0.1602,  0.2351],
          [ 0.3299, -0.0015, -0.0091]]],


        [[[ 0.3093,  0.2849, -0.0814],
          [-0.1810,  0.0747, -0.2451],
          [-0.0062, -0.1523,  0.0507]]],


        [[[ 0.0547, -0.1621, -0.3325],
          [-0.3027, -0.1189, -0.1728],
          [-0.1324, -0.3294, -0.2991]]]], requires_grad=True)
Parameter containing:
tensor([ 0.0834, -0.0405, -0.1260,  0.0395, -0.1728, -0.0301],
       requires_grad=True)
Parameter containing:
tensor([[ 0.0158, -0.0124,  0.0040,  ...,  0.0160, -0.0070, -0.0261],
        [-0.0235,  0.0008, -0.0201,  ...,  0.0093, -0.0032,  0.0101],
        [ 0.0181,  0.0232,  0.0075,  ..., -0.0072,  0.0260, -0.0266],
        ...,
        [-0.0089, -0.0109,  0.0087,  ...,  0.0104,  0.0166, -0.0095],
        [ 0.0129,  0.0206, -0.0269,  ...,  0.0073,  0.0252, -0.0124],
        [ 0.0070,  0.0080, -0.0232,  ..., -0.0263,  0.0118, -0.0227]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0251,  0.0155,  0.0030, -0.0047,  0.0054,  0.0185,  0.0231,  0.0214,
        -0.0173, -0.0197], requires_grad=True)
"""

"""
# net.named_parameters可同时返回可学习的参数及名称。
for name, parameter in net.named_parameters():
    print(name, ": ", parameter.size())

conv1.weight :  torch.Size([6, 1, 3, 3])
conv1.bias :  torch.Size([6])
fc1.weight :  torch.Size([10, 1350])
fc1.bias :  torch.Size([10])
"""

# ------------------------- 2.初始化网络，进行前向传播 ----------------
# forward函数的输入和输出都是Tensor
input = torch.randn(1, 1, 32, 32)  # 这里的对应前面forward的输入是32
out = net(input)
"""
>>> print(out)
tensor([[ 0.2332,  0.1621, -0.0448, -0.4732,  0.4609,  0.1004, -0.0490, -0.2577,
          0.4698, -0.5834]], grad_fn=<AddmmBackward>)
>>> print(out.size())
torch.Size([1, 10])
"""

# ------------------------- 3.反向传播 ----------------------------
# 在反向传播前，先要将所有参数的梯度清零
net.zero_grad()
# 反向传播的实现是PyTorch自动实现的，我们只要调用这个函数即可, 这里为了简单，直接随机一个误差进行反向传播
# 注意 out 中包含所有可训练变量（作为叶子节点），所以使用 out 去调用 backward()，进行反向传播
out.backward(torch.ones(1, 10))

"""
>>> print(input.size())  # torch.Size([1, 1, 32, 32])

注意:torch.nn只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个batch。
也就是说，就算我们输入一个样本，也会对样本进行分批，所以，所有的输入都会增加一个维度，我们对比下刚才的input，
nn中定义为3维，但是我们人工创建时多增加了一个维度，变为了4维，最前面的1即为batch-size
"""

# ------------------------- 4.损失函数 ----------------------------
# 在nn中PyTorch还预制了常用的损失函数，下面我们用MSELoss用来计算均方误差
y = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
# loss是个scalar，我们可以直接用item获取到他的python类型的数值
# print(loss.item())  # 27.606237411499023

# ------------------------- 5.优化器 ------------------------------
"""
在反向传播计算完所有参数的梯度后，还需要使用优化方法来更新网络的权重和参数，例如随机梯度下降法(SGD)的更新策略如下：
weight = weight - learning_rate * gradient
在torch.optim中实现大多数的优化方法，例如RMSProp、Adam、SGD等，下面我们使用SGD做个简单的样例
"""

# 新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()
# 开始反向传播
loss.backward()
# 更新参数
optimizer.step()

