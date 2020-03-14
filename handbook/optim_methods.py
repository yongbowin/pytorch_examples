# -*- coding:utf-8 -*-
"""
@Time  : 3/14/20 5:07 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : optim_methods.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter2/2.2-deep-learning-basic-mathematics.ipynb
        优化方法
"""

import torch.nn as nn
import torch.optim as optim

"""
在介绍损失函数的时候我们已经说了，梯度下降是一个使损失函数越来越小的优化算法，在无求解机器学习算法的模型参数，即约束优化问题时，
梯度下降（Gradient Descent）是最常采用的方法之一。所以梯度下降是我们目前所说的机器学习的核心，了解了它的含义，也就了解了机器
学习算法的含义。

梯度 : 
    在微积分里面，对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。 例如函数f(x,y), 分别对
    x，y求偏导数，求得的梯度向量就是(∂f/∂x, ∂f/∂y)T，简称grad f(x,y)或者▽f(x,y)。
    几何上讲，梯度就是函数变化增加最快的地方，沿着梯度向量的方向，更加容易找到函数的最大值。反过来说，沿着梯度向量相反的方向梯
    度减少最快，也就是更加容易找到函数的最小值。我们需要最小化损失函数，可以通过梯度下降法来一步步的迭代求解，得到最小化的损失
    函数，和模型参数值。

梯度下降法直观解释 : 
    梯度下降法就好比下山，我们并不知道下山的路，于是决定走一步算一步，每走到一个位置的时候，求解当前位置的梯度，沿着梯度的负方向，
    也就是当前最陡峭的位置向下走一步，然后继续求解当前位置梯度，向这一步所在位置沿着最陡峭最易下山的位置走一步。这样一步步的走下
    去，一直走到觉得我们已经到了山脚。
    这样走下去，有可能我们不能走到山脚，而是到了某一个局部的山峰低处（局部最优解）。
    这个问题在以前的机器学习中可能会遇到，因为机器学习中的特征比较少，所以导致很可能陷入到一个局部最优解中出不来，但是到了深度学习，
    动辄百万甚至上亿的特征，出现这种情况的概率几乎为0，所以我们可以不用考虑这个问题。

Mini-batch 的梯度下降法 : 
    对整个训练集进行梯度下降法的时候，我们必须处理整个训练数据集，然后才能进行一步梯度下降，即每一步梯度下降法需要对整个训练集进行
    一次处理，如果训练数据集很大的时候处理速度会很慢，而且也不可能一次的载入到内存或者显存中，所以我们会把大数据集分成小数据集，
    一部分一部分的训练，这个训练子集即称为 Mini-batch。 
    在PyTorch中就是使用这种方法进行的训练，可以看看上一章中关于dataloader的介绍里面的batch_size就是我们一个Mini-batch的大小。
    对于普通的梯度下降法，一个epoch只能进行一次梯度下降；而对于Mini-batch梯度下降法，一个epoch可以进行Mini-batch的个数次梯度下降。

    如果训练样本的大小比较小时，能够一次性的读取到内存中，那我们就不需要使用Mini-batch，
    如果训练样本的大小比较大时，一次读入不到内存或者现存中，那我们必须要使用 Mini-batch来分批的计算
    Mini-batch size的计算规则如下，在内存允许的最大情况下使用2的N次方个size

torch.optim是一个实现了各种优化算法的库。大部分常用优化算法都有实现，我们直接调用即可。
"""

# 1.随机梯度下降算法，带有动量（momentum）的算法作为一个可选参数可以进行设置，样例如下：
"""
lr参数为学习率，对于SGD来说一般选择0.1 0.01.0.001，如何设置会在后面实战的章节中详细说明
##如果设置了momentum，就是带有动量的SGD，可以不设置

`SGD` : 
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
"""
model = nn.Linear(1, 1)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

"""
除了以上的带有动量Momentum梯度下降法外，RMSprop（root mean square prop）也是一种可以加快梯度下降的算法，利用RMSprop算法，
可以减小某些维度梯度更新波动较大的情况，使其梯度下降的速度变得更快
"""
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

"""
Adam 优化算法的基本思想就是将 Momentum 和 RMSprop 结合起来形成的一种适用于不同深度学习结构的优化算法

`Adam` : 
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
"""
optimizer_adam = optim.Adam(model.parameters(), lr=0.01)

