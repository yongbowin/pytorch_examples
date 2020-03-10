# -*- coding:utf-8 -*-
"""
@Time  : 3/9/20 9:52 AM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : autograd.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/2_autograd_tutorial.ipynb
"""

import torch


def auto_grad():
    """
    Autograd: 自动求导机制
        PyTorch 中所有神经网络的核心是 autograd 包。 我们先简单介绍一下这个包，然后训练第一个简单的神经网络。
        autograd包为张量上的所有操作提供了自动求导。 它是一个在运行时定义的框架，这意味着反向传播是根据你的代码来确定如何运行，并且每次迭代可以是不同的。
    """
    """
    示例: 张量（Tensor）
    torch.Tensor是这个包的核心类。如果设置 .requires_grad 为 True，那么将会追踪所有对于该张量的操作。 当完成计算后通过调用 .backward()，
    自动计算所有的梯度， 这个张量的所有梯度将会自动积累到 .grad 属性。
    要阻止张量跟踪历史记录，可以调用.detach()方法将其与计算历史记录分离，并禁止跟踪它将来的计算记录。
    为了防止跟踪历史记录（和使用内存），可以将代码块包装在with torch.no_grad()：中。 在评估模型时特别有用，因为模型可能具有requires_grad = True
    的可训练参数，但是我们不需要梯度计算。
    在自动梯度计算中还有另外一个重要的类Function.
    Tensor 和 Function互相连接并生成一个非循环图，它表示和存储了完整的计算历史。 每个张量都有一个.grad_fn属性，这个属性引用了一个创建了Tensor的
    Function（除非这个张量是用户手动创建的，即，这个张量的 grad_fn 是 None）。
    如果需要计算导数，你可以在Tensor上调用.backward()。 如果Tensor是一个标量（即它包含一个元素数据）则不需要为backward()指定任何参数， 但是如果
    它有更多的元素，你需要指定一个gradient 参数来匹配张量的形状。
    
    注：在其他的文章中你可能会看到说将Tensor包裹到Variable中提供自动梯度计算，Variable 这个在0.41版中已经被标注为过期了，现在可以直接使用Tensor
    """
    # 创建一个张量并设置 requires_grad=True 用来追踪他的计算历史
    x = torch.ones(2, 2, requires_grad=True)
    """
    >>> print(x)
    tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
    """

    # 对张量进行操作
    y = x + 2
    """结果y已经被计算出来了，所以，grad_fn已经被自动生成了。
    >>> print(y)
    tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
    >>> print(y.grad_fn)
    <AddBackward0 object at 0x7ff3e82611d0>
    """

    # 对y进行一个操作
    z = y * y * 3
    out = z.mean()
    """
    >>> print(z, out)
    tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
    """

    # .requires_grad_( ... ) 可以改变现有张量的 requires_grad属性。 如果没有指定的话，默认输入的flag是 False。
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    a.requires_grad_(True)
    b = (a * a).sum()
    """
    >>> print(a.requires_grad)
    False
    >>> print(a.requires_grad)
    True
    >>> print(b.grad_fn)
    <SumBackward0 object at 0x7f893aeb61d0>
    """

    # 梯度: 反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))。
    # print(x.grad)  # None, 在设置反向传播之前x的梯度还没有计算，是None

    out.backward()
    # print gradients d(out)/dx
    """
    >>> print(x.grad)
    tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
    """

    """
    x = torch.randn(2, 3, requires_grad=True)

    y = x * 2
    print(x)
    print(y, type(y))
    print(y.data, type(y.data), y.data.shape)
    print(y.data.norm())  # default p=2, dim是全体的元素都计算，只输出一个值
    while y.data.norm() < 1000:
        y = y * 2
    print(y)
    
    ------
    tensor([[-0.6695,  0.2913,  0.6975],
        [ 0.0571, -0.9945,  1.3549]], requires_grad=True)
    tensor([[-1.3390,  0.5825,  1.3950],
        [ 0.1142, -1.9890,  2.7098]], grad_fn=<MulBackward0>) <class 'torch.Tensor'>
    tensor([[-1.3390,  0.5825,  1.3950],
        [ 0.1142, -1.9890,  2.7098]]) <class 'torch.Tensor'> torch.Size([2, 3])
    tensor(3.9231)
    tensor([[-342.7727,  149.1271,  357.1263],
        [  29.2391, -509.1932,  693.7017]], grad_fn=<MulBackward0>)
    """

    """
    在这个情形中，y不再是个标量。torch.autograd无法直接计算出完整的雅可比行列，但是如果我们只想要vector-Jacobian product，
    只需将向量作为参数传入backward
    """
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(gradients)

    # print(x.grad)  # tensor([5.1200e+01, 5.1200e+02, 5.1200e-02])

    """
    如果.requires_grad=True但是你又不希望进行autograd的计算， 那么可以将变量包裹在 with torch.no_grad()中
    """
    # print(x.requires_grad)  # True
    # print((x ** 2).requires_grad)  # True

    with torch.no_grad():
        print((x ** 2).requires_grad)  # False


if __name__ == '__main__':
    auto_grad()
