# -*- coding:utf-8 -*-
"""
@Time  : 3/9/20 3:22 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : nn.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/3_neural_networks_tutorial.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    """
    使用torch.nn包来构建神经网络。
    autograd，nn包依赖autograd包来定义模型并求导。 一个nn.Module包含各个层和一个forward(input)方法，该方法返回output。
    :return:
    """

    """
    神经网络的典型训练过程如下：
    1.定义包含一些可学习的参数(或者叫权重)神经网络模型；
    2.在数据集上迭代；
    3.通过神经网络处理输入；
    4.计算损失(输出结果和正确值的差值大小)；
    5.将梯度反向传播回网络的参数；
    6.更新网络的参数，主要使用如下简单的更新原则：  weight = weight - learning_rate * gradient
    """

    def __init__(self):
        """
        super() 函数是用于调用父类(超类)的一个方法。
        super 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。
        MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表。
        Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx
        super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象
        """
        super(Net, self).__init__()
        """
        nn.Conv2d(         1,                   6,                    5)
                  1 input image channel, 6 output channels, 5x5 square convolution
        """
        # 卷积层
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全连接层
        # an affine operation: y = Wx +b, 仿射函数，即nn.Linear()函数中进行的操作是该方程表示的操作
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16个channels，5×5的卷积核，全连接层输入维度input=16×5×5, 输出维度output=120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        输入是 x， 激活函数使用 RELU

        在模型中必须要定义 forward 函数，backward 函数（用来计算梯度）会被autograd自动创建。 可以在 forward 函数中使用任何针对 Tensor 的操作。
        """
        # 在 2×2 的窗口上进行最大池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 加入 size 是一个正方形，可以只指定一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # 进行拉平操作，即flatten操作(目的是作为后边全连接层的输入)， view()相当于reshape()函数
        x = x.view(-1, self.num_flat_features(x))

        # 全连接层进行仿射变换, 前两层全连接层都是用relu()激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """
        计算特征数量，tensor中每一个元素都是一个特征
        """
        size = x.size()[1:]  # 处理batch以外的所有维度值
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    """ 主要过程
    1.定义一个网络
    2.处理输入，调用backword
    3.计算损失
    4.更新网络权重
    """

    net = Net()

    # --------------------------- 打印模型架构 ---------------------------
    print(net)  # 打印模型架构
    """
    Net(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )
    """

    # --------------------------- 打印模型参数 ---------------------------
    # net.parameters()返回可被学习的参数（权重）列表和值
    params = list(net.parameters())
    # print(params)
    print(len(params))  # 10
    print(params[0].size())  # conv1's .weight, torch.Size([6, 1, 5, 5])
    for num, param in enumerate(params):
        print(num, param.size())
    """ W 和 b 交叉排列
    0 torch.Size([6, 1, 5, 5])
    1 torch.Size([6])
    2 torch.Size([16, 6, 5, 5])
    3 torch.Size([16])
    4 torch.Size([120, 400])
    5 torch.Size([120])
    6 torch.Size([84, 120])
    7 torch.Size([84])
    8 torch.Size([10, 84])
    9 torch.Size([10])
    """

    # ----------------------------- 测试模型 -----------------------------
    """
    注意：
        ``torch.nn`` 只支持小批量输入。整个 ``torch.nn`` 包都只支持小批量样本，而不支持单个样本。 
        例如，``nn.Conv2d`` 接受一个4维的张量， ``每一维分别是sSamples * nChannels * Height * Width（样本数*通道数*高*宽）``。 
        如果你有单个样本，只需使用 ``input.unsqueeze(0)`` 来添加其它的维数
    """
    # 测试随机输入32×32。 注：这个网络（LeNet）期望的输入大小是32×32，如果使用MNIST数据集来训练这个网络，请把图片大小重新调整到32×32。
    input = torch.randn(1, 1, 32, 32)  # torch.Size([1, 1, 32, 32])
    out = net(input)  # torch.Size([1, 10])
    """
    >>> print(out)
    tensor([[ 0.1021,  0.0050, -0.0766,  0.1305, -0.0788,  0.0296,  0.1538, -0.0124,
          0.0137, -0.1622]], grad_fn=<AddmmBackward>)
    """

    # 将所有参数的梯度缓存清零，然后进行随机梯度的的反向传播
    net.zero_grad()
    out.backward(torch.randn(1, 10))  # 随机一个shape为(1, 10)的梯度，进行反向传播

    """
    torch.Tensor：一个用过自动调用 backward()实现支持自动梯度计算的 多维数组 ， 并且保存关于这个向量的梯度 w.r.t.
    nn.Module：神经网络模块。封装参数、移动到GPU上运行、导出、加载等。
    nn.Parameter：一种变量，当把它赋值给一个Module时，被 自动 地注册为一个参数。
    autograd.Function：实现一个自动求导操作的前向和反向定义，每个变量操作至少创建一个函数节点，每一个Tensor的操作都会创建一个接到创建Tensor和 
    编码其历史的函数的Function节点。
    """

    # ----------------------------- 损失函数 -----------------------------
    """
    一个损失函数接受一对 (output, target) 作为输入，计算一个值来估计网络的输出和目标值相差多少。
    nn包中有很多不同的损失函数。 nn.MSELoss是一个比较简单的损失函数，它计算输出和目标间的均方误差.
    """
    output = net(input)  # torch.Size([1, 10])
    target = torch.randn(10)  # 随机值作为样例
    target = target.view(1, -1)  # 使target和output的shape相同
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    # print(loss)  # tensor(1.4147, grad_fn=<MseLossBackward>)
    """
    现在，如果在反向过程中跟随loss ， 使用它的 .grad_fn 属性，将看到如下所示的计算图。
    
    input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
    
    所以，当我们调用 loss.backward()时,整张计算图都会 根据loss进行微分，而且图中所有设置为requires_grad=True的张量 
    将会拥有一个随着梯度累积的.grad 张量。
    """
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
    """
    <MseLossBackward object at 0x7fb89d580fd0>
    <AddmmBackward object at 0x7fb89c10d240>
    <AccumulateGrad object at 0x7fb89d580fd0>
    """

    # ----------------------------- 反向传播 -----------------------------
    """
    调用loss.backward()获得反向传播的误差。
    但是在调用前需要清除已存在的梯度，否则梯度将被累加到已存在的梯度。
    现在，我们将调用loss.backward()，并查看conv1层的偏差（bias）项在反向传播前后的梯度。
    """
    net.zero_grad()  # 清除梯度

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    """
    conv1.bias.grad before backward
    tensor([0., 0., 0., 0., 0., 0.])
    """

    loss.backward()  # loss进行反向传播

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)
    """
    conv1.bias.grad after backward
    tensor([-0.0062, -0.0113,  0.0097,  0.0062,  0.0071, -0.0101])
    """

    # ------------------------- 更新权重（自定义方法） -------------------------
    """ 自定义优化方法
    在实践中最简单的权重更新规则是随机梯度下降（SGD）： ``weight = weight - learning_rate * gradient``
    
    我们可以使用简单的Python代码实现这个规则：
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)
    
    The .sub_() method indicates with the “_” suffix that it is done in-place.
        1.Actual symbols +,-,x,/ e.g. a = a + b
        2.Not in-place operators add(),sub() e.g. a = a.add(b)
        3.In-place operators add_(),sub_() e.g. a.add_(b)
        
    f.data.sub_(f.grad.data * learning_rate) 等价于 f.data = f.data -(f.grad.data * learning_rate)
    """
    learning_rate = 0.01
    for f in net.parameters():
        # print(f)  # 获取 W 或者 b 的数据, 带requires_grad=True参数
        # print(f.data)  # 获取 W 或者 b 的数据， 不带requires_grad=True参数
        # print(f.grad)  # 获取 W 或者 b 的数据， 不带requires_grad=True参数
        # print(f.grad.data)  # 获取对应 W 或者 b 的梯度数据， 不带requires_grad=True参数
        f.data.sub_(f.grad.data * learning_rate)

    # -------------------------- 更新权重（使用optim包） -------------------------
    """
    注意:
        观察如何使用``optimizer.zero_grad()``手动将梯度缓冲区设置为零。
        这是因为梯度是按Backprop部分中的说明累积的。
    """
    # 创建优化器
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # 在训练循环中
    optimizer.zero_grad()  # 清空梯度缓存
    output = net(input)  # 前向传播后，得到输出
    loss = criterion(output, target)  # 计算前向传播后得到的输出与目标之间的差异（使用均方根误差计算该损失）
    loss.backward()  # 将该loss进行方向传播，来更新权重
    optimizer.step()  # 使用优化器来更新权重

