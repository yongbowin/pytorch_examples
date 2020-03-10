# -*- coding:utf-8 -*-
"""
@Time  : 3/8/20 5:29 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : functions.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/1_tensor_tutorial.ipynb
        for more details https://pytorch.org/docs/stable/torch.html
"""

import torch
import numpy as np


def randn_rand_normal_linspace():
    """
    产生一个随即tensor

    Functions:
        torch.randn():
            torch.randn(*sizes, out=None) → Tensor
            返回一个张量，包含了从标准正态分布(均值为0，方差为 1，即高斯白噪声)中抽取一组随机数，形状由可变参数sizes定义。
        torch.rand():
            torch.rand(*sizes, out=None) → Tensor
            返回一个张量，包含了从区间[0,1)的均匀分布中抽取的一组随机数，形状由可变参数sizes 定义。
        torch.normal():
            torch.normal(mean, std, out=None)
            返回一个张量，包含从给定参数means,std的离散正态分布中抽取随机数。 均值means是一个张量，包含每个输出元素相关
            的正态分布的均值。 std是一个张量，包含每个输出元素相关的正态分布的标准差。 均值和标准差的形状不须匹配，但每个
            张量的元素个数须相同。
            torch.normal(mean=0.0, std, out=None) 所有抽取的样本共享均值。
            torch.normal(mean, std=1.0, out=None) 所有抽取的样本共享标准差。

        torch.linspace():
            torch.linspace(start, end, steps=100, out=None) → Tensor
            返回一个1维张量，包含在区间start 和 end 上均匀间隔的steps个点。 输出1维张量的长度为steps。
                start (float) – 序列的起始点
                end (float) – 序列的最终值
                steps (int) – 在start 和 end间生成的样本数
                out (Tensor, optional) – 结果张量
    :return:
    """
    # torch.randn()

    # print(torch.rand())  # invalid
    a1 = torch.rand(4)
    a2 = torch.rand(2, 3)
    a3 = torch.rand(2, 3, 4)
    """
    >>> print(a1, a1.shape)
    outputs:
        tensor([0.8494, 0.4217, 0.1465, 0.3048]) torch.Size([4])
    
    >>> print(a2, a2.shape)
    outputs:
        tensor([[0.1952, 0.2853, 0.9431],
            [0.8105, 0.5219, 0.2616]]) torch.Size([2, 3])
    
    >>> print(a3, a3.shape)
    outputs:
        tensor([[[0.9637, 0.9078, 0.9330, 0.0186],
             [0.1283, 0.5353, 0.1693, 0.7082],
             [0.9060, 0.1584, 0.7350, 0.8504]],
    
            [[0.2239, 0.4077, 0.1403, 0.9761],
             [0.9001, 0.2220, 0.7728, 0.4138],
             [0.8184, 0.0965, 0.3094, 0.4910]]]) torch.Size([2, 3, 4])
    """

    c1 = torch.normal(mean=torch.FloatTensor(np.array(torch.arange(1, 11))),
                      std=torch.FloatTensor(np.array(torch.arange(1, 0, -0.1))))

    c2 = torch.normal(mean=0.5, std=torch.FloatTensor(np.array(torch.arange(1, 6))))

    c3 = torch.normal(mean=torch.FloatTensor(np.array(torch.arange(1, 6))))

    """
    RuntimeError: _th_normal not supported on CPUType for Long
    在Pytorch下的torch的exp运算是针对浮点数据类型的。
    
    进行强制类型转换：
    torch.FloatTensor(np.array(torch.arange(1, 11)))
    
    >>> print(c1, c1.shape)
    outputs:
        tensor([ 0.5777,  1.6084,  3.7288,  4.9057,  4.9233,  5.5791,  6.8800,  8.0431,
             8.9361, 10.0330]) torch.Size([10])
    >>> print(c2, c2.shape)
    outputs:
        tensor([ 0.7069,  2.6835,  4.4622, -3.7009,  9.2297]) torch.Size([5])
    >>> print(c3, c3.shape)
    outputs:
        tensor([0.0644, 1.6956, 3.2060, 4.6170, 4.3594]) torch.Size([5])

    """

    b1 = torch.linspace(3, 10, steps=5)
    b2 = torch.linspace(-10, 10, steps=5)
    b3 = torch.linspace(start=-10, end=10, steps=5)
    """
    >>> print(b1, b1.shape)
    outputs:
        tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000]) torch.Size([5])

    >>> print(b2, b2.shape)
    outputs:
        tensor([-10.,  -5.,   0.,   5.,  10.]) torch.Size([5])

    >>> print(b3, b3.shape)
    outputs:
        tensor([-10.,  -5.,   0.,   5.,  10.]) torch.Size([5])
    """


def arange_range():
    """
    产生一组固定尺寸和固定范围的tensor, 警告：建议使用函数 torch.arange()

    Functions:
        torch.arange(): 左闭右开区间 [ )
            torch.arange(start, end, step=1, out=None) → Tensor
            返回一个1维张量，长度为 floor((end−start)/step)。包含从start到end，以step为步长的一组序列值(默认步长为1)。
        torch.range(): 左闭右闭区间 [ ]
            torch.range(start, end, step=1, out=None) → Tensor
            返回一个1维张量，有 floor((end−start)/step)+1 个元素。包含在半开区间[start, end）从start开始，以step为步长的一组值。
    :return:
    """
    a1 = torch.arange(1, 4)
    a2 = torch.arange(1, 2.5, 0.5)
    """
    >>> print(a1, a1.shape)
    outputs:
        tensor([1, 2, 3]) torch.Size([3])
    >>> print(a2, a2.shape)
    outputs:
        tensor([1.0000, 1.5000, 2.0000]) torch.Size([3])
    """

    b1 = torch.range(1, 4)
    b2 = torch.range(1, 2.5, 0.5)
    """
    >>> print(b1, b1.shape)
    outputs:
        tensor([1., 2., 3., 4.]) torch.Size([4])
    >>> print(b2, b2.shape)
    outputs:
        tensor([1.0000, 1.5000, 2.0000, 2.5000]) torch.Size([4])
    """


def new_tensor():
    """
    根据要求新建一个tensor
    :return:
    """
    x1 = torch.empty(5, 3)
    x2 = torch.zeros(5, 3, dtype=torch.long)
    x3 = torch.tensor([5.5, 3])

    x4 = x3.new_ones(5, 3, dtype=torch.double)  # new_* 方法来创建对象
    x5 = torch.randn_like(x4, dtype=torch.float)  # 覆盖 dtype! 对象的size 是相同的，只是值和类型发生了变化
    """
    1.创建一个 5x3 矩阵, 但是未初始化
    
    >>> print(x1, x1.shape)
    outputs:
        tensor([[-1.8716e+33,  4.5706e-41, -3.9128e+08],
            [ 3.0688e-41, -7.2982e+14,  4.5706e-41],
            [-6.1707e+24,  4.5706e-41, -7.9571e+14],
            [ 4.5706e-41, -1.7463e+25,  4.5706e-41],
            [-7.9571e+14,  4.5706e-41, -7.9572e+14]]) torch.Size([5, 3])
    
    2.创建一个0填充的矩阵，数据类型为long
    >>> print(x2, x2.shape)
    outputs:
        tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]) torch.Size([5, 3])
    
    3.创建tensor并使用现有数据初始化
    >>> print(x3, x3.shape)
    outputs:
        tensor([5.5000, 3.0000]) torch.Size([2])
    
    4.xx
    >>> print(x4, x4.shape)
    tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64) torch.Size([5, 3])
    >>> print(x5, x5.shape)
    tensor([[-1.5502,  1.4383,  0.4045],
        [-0.7093,  0.4394, -0.6974],
        [-1.1750, -0.0560,  1.3077],
        [-0.2103,  0.3441,  0.7604],
        [ 1.6525,  0.4058,  0.1853]]) torch.Size([5, 3])
    """


def size_shape():
    """
    Functions:
        size()和shape相同， 使用size方法与Numpy的shape属性返回的相同，张量也支持shape属性，后面会详细介绍
    :return:
    """
    # 'torch.Size' 返回值是tuple类型, 所以它支持tuple类型的所有操作.
    pass


def operations():
    """
    tensor的加减乘除运算
    :return:
    """
    # 加法
    x1 = torch.rand(5, 3)
    x2 = torch.rand(5, 3)

    print(x1 + x2)
    print(torch.add(x1, x2))

    # 提供输出tensor作为参数
    result = torch.empty(5, 3)
    torch.add(x1, x2, out=result)
    print(result)

    # 替换，x2的值发生改变
    # adds x2 to x1, 任何 以``_`` 结尾的操作都会用结果替换原变量. 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``.
    x1.add_(x2)
    print(x1)


def slice_ops():
    """
    可以使用与NumPy索引方式相同的操作来进行对张量的操作, 切片操作
    :return:
    """
    x1 = torch.rand(5, 3)
    print(x1)
    print(x1[:, 1])


def change_dim():
    """
    改变张量的维度和大小：
        torch.view: 可以改变张量的维度和大小, 与Numpy的reshape类似
    :return:
    """

    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)  # size -1 从其他维度推断
    """
    print(x.size(), y.size(), z.size())
    torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
    """


def get_val():
    """
    如果你有只有一个元素的张量，使用.item()来得到Python数据类型的数值
    :return:
    """
    x = torch.randn(1)
    print(x)
    print(x.item())
    """
    tensor([-0.5492])
    -0.549225926399231
    """


def trans_numpy():
    """
    tensor与numpy之间的转换：
        Torch Tensor与NumPy数组共享底层内存地址，修改一个会导致另一个的变化。
    :return:
    """
    # 1.torch --> numpy
    x = torch.ones(5)
    y = x.numpy()

    # tensor([1., 1., 1., 1., 1.]) torch.Size([5]) <class 'torch.Tensor'>
    print(x, x.shape, type(x))

    # [1.  1.  1.  1.  1.] <class 'numpy.ndarray'>
    print(y, type(y))

    # 观察numpy数组的值是如何改变的。
    x.add_(1)  # 任何 以``_`` 结尾的操作都会用结果替换原变量. 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``.
    print(x)
    print(y)
    """ x和y都发生了变化
    tensor([2., 2., 2., 2., 2.])
    [ 2.  2.  2.  2.  2.]
    """

    # tensor转化为numpy
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)
    """ a和b都发生了变化
    [ 2.  2.  2.  2.  2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    
    所有的 Tensor 类型默认都是基于CPU， CharTensor 类型不支持到 NumPy 的转换.
    """


def cuda_tensor():
    """
    将tensor加载到CUDA中, 使用.to 方法 可以将Tensor移动到任何设备中
    :return:
    """
    x = torch.randn(2, 3)
    print(x)

    # is_available 函数判断是否有cuda可以使用
    # ``torch.device``将张量移动到指定的设备中
    if torch.cuda.is_available():
        device = torch.device("cuda")  # a CUDA 设备对象
        y = torch.ones_like(x, device=device)  # 直接从GPU创建张量      (方法一)
        print(y)
        x = x.to(device)  # 或者直接使用``.to("cuda")``将张量移动到cuda中  (方法二)
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # ``.to`` 也会对变量的类型做更改

    """
    >>> print(x)
    tensor([[-0.9498,  0.2693, -2.0094],
        [ 2.4877, -1.5897,  0.8316]])
    >>> print(y)
    tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
    >>> print(z)
    tensor([[ 0.0502,  1.2693, -1.0094],
        [ 3.4877, -0.5897,  1.8316]], device='cuda:0')
    >>> print(z.to("cpu", torch.double))
    tensor([[ 0.0502,  1.2693, -1.0094],
        [ 3.4877, -0.5897,  1.8316]], dtype=torch.float64)
    """


def _norm():
    """
    torch.norm(input, p, dim, out=None) → Tensor
        input (Tensor) – 输入张量
        p (float) – 范数计算中的幂指数值, default=2
        dim (int) – 缩减的维度
        out (Tensor, optional) – 结果张量
    :return:
    """
    a = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]).float()
    """
    >>> print(a, type(a))
    tensor([[1., 2., 3., 4.],
        [1., 2., 3., 4.]]) <class 'torch.Tensor'>
    >>> print(a.shape)
    torch.Size([2, 4])
    """
    a0 = torch.norm(a, p=2, dim=0)  # 按0维度求2范数
    a1 = torch.norm(a, p=2, dim=1)  # 按1维度求2范数
    """
    >>> print(a0)
    tensor([1.4142, 2.8284, 4.2426, 5.6569])
    >>> print(a1)
    tensor([5.4772, 5.4772])
    """

    # xx
    a = torch.rand((2, 3, 4))
    """
    >>> print(a, a.shape)
    tensor([[[0.9407, 0.4680, 0.9844, 0.5807],
         [0.1841, 0.5339, 0.5920, 0.9322],
         [0.1724, 0.2696, 0.3965, 0.8955]],

        [[0.8910, 0.5273, 0.8243, 0.7536],
         [0.6296, 0.7634, 0.2678, 0.3650],
         [0.8184, 0.1992, 0.5086, 0.3332]]]) torch.Size([2, 3, 4])
    """
    at = torch.norm(a, p=2, dim=1, keepdim=True)  # 保持维度
    af = torch.norm(a, p=2, dim=1, keepdim=False)  # 不保持维度
    """
    >>> print(at, at.shape)
    tensor([[[1.4586, 1.0426, 1.1850, 1.1981]],
            [[0.5215, 1.2621, 1.3342, 1.0673]]]) torch.Size([2, 1, 4]) 保持维度
    >>> print(af, af.shape)
    tensor([[1.4586, 1.0426, 1.1850, 1.1981],
        [0.5215, 1.2621, 1.3342, 1.0673]]) torch.Size([2, 4]) 不保持维度
    
    可以发现，当keepdim=False时，输出比输入少一个维度（就是指定的dim求范数的维度）。
    而keepdim=True时，输出与输入维度相同，仅仅是输出在求范数的维度上元素个数变为1。
    这也是为什么有时我们把参数中的dim称为缩减的维度，因为norm运算之后，此维度或者消失或者元素个数变为1.
    """


def squeeze_unsqueeze():
    """
    变换tensor的维度

    squeeze(): 挤压维度, 只有维度为1时才会去掉
    unsqueeze(): 扩展维度

    维度从0开始计数，超出维度范围会报错 IndexError: Dimension out of range
    不返回新对象，所以变换后需要重新赋值
    """
    x = torch.arange(0, 6)
    x = x.view(2, 3)
    """ 扩展维度
    >>> print(x, x.shape)
    tensor([[0, 1, 2],
        [3, 4, 5]]) torch.Size([2, 3])
    >>> print(x.unsqueeze(0), x.unsqueeze(0).shape)  # 0
    tensor([[[0, 1, 2],
         [3, 4, 5]]]) torch.Size([1, 2, 3])
    >>> print(x.unsqueeze(1), x.unsqueeze(1).shape)  # 1
    tensor([[[0, 1, 2]],
        [[3, 4, 5]]]) torch.Size([2, 1, 3])
    >>> print(x.unsqueeze(2), x.unsqueeze(2).shape)
    tensor([[[0],
         [1],
         [2]],

        [[3],
         [4],
         [5]]]) torch.Size([2, 3, 1])
    >>> print(x.unsqueeze(-1), x.unsqueeze(-1).shape)  # -1
    tensor([[[0],
         [1],
         [2]],

        [[3],
         [4],
         [5]]]) torch.Size([2, 3, 1])
    >>> print(x.unsqueeze(-2), x.unsqueeze(-2).shape)  # -2
    tensor([[[0, 1, 2]],
        [[3, 4, 5]]]) torch.Size([2, 1, 3])
    >>> print(x.unsqueeze(-3), x.unsqueeze(-3).shape)
    tensor([[[0, 1, 2],
         [3, 4, 5]]]) torch.Size([1, 2, 3])
    """

    y = x.unsqueeze(0)
    print(y, y.shape)
    print(y.squeeze(0), y.squeeze(0).shape)
    print(y.squeeze(1), y.squeeze(1).shape)
    """ 挤压维度， 只有维度为1时才会去掉
    >>> print(y, y.shape)
    tensor([[[0, 1, 2],
         [3, 4, 5]]]) torch.Size([1, 2, 3])
    >>> print(y.squeeze(0), y.squeeze(0).shape)
    tensor([[0, 1, 2],
        [3, 4, 5]]) torch.Size([2, 3])
    >>> print(y.squeeze(1), y.squeeze(1).shape)
    tensor([[[0, 1, 2],
         [3, 4, 5]]]) torch.Size([1, 2, 3])
    """


if __name__ == '__main__':
    # randn_rand_normal_linspace()
    # arange_range()
    # new_tensor()
    # size_shape()
    # operations()
    # slice_ops()
    # change_dim()
    # get_val()
    # trans_numpy()
    # cuda_tensor()
    # _norm()
    squeeze_unsqueeze()
