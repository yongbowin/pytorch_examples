# -*- coding:utf-8 -*-
"""
@Time  : 3/12/20 10:14 AM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : tensor.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter2/2.1.1.pytorch-basics-tensor.ipynb
"""

import torch
import numpy as np

# print(torch.__version__)  # 打印版本

# ------------------------------ 1.张量 ------------------------------

"""
张量的英文是Tensor，它是PyTorch里面基础的运算单位，与Numpy的ndarray相同都表示的是一个多维的矩阵。 
与ndarray的最大区别就是，PyTorch的Tensor可以在 GPU 上运行，而 numpy 的 ndarray 只能在 CPU 
上运行，在GPU上运行大大加快了运算速度。

张量（Tensor）是一个定义在一些向量空间和一些对偶空间的笛卡儿积上的多重线性映射，其坐标是|n|维空间内，
有|n|个分量的一种量， 其中每个分量都是坐标的函数， 而在坐标变换时，这些分量也依照某些规则作线性变换。
r称为该张量的秩或阶（与矩阵的秩和阶均无关系）。

在同构的意义下，第零阶张量 （r = 0） 为标量 （Scalar），第一阶张量 （r = 1） 为向量 （Vector）， 
第二阶张量 （r = 2） 则成为矩阵 （Matrix），第三阶以上的统称为多维张量。
"""

scalar = torch.tensor(2.333)  # 定义一个标量，可以直接使用item()取出值
# print(scalar, scalar.size(), scalar.item())  # tensor(2.3330) torch.Size([]) 2.3329999446868896

tensor = torch.tensor([2.333])  # 特别的：如果张量中只有一个元素的tensor也可以调用tensor.item方法
# print(tensor, tensor.size(), tensor.item())  # tensor([2.3330]) torch.Size([1]) 2.3329999446868896

# ------------------------------ 2.基本类型 ---------------------------
"""基本类型
Tensor的基本数据类型有五种：

1. 32位浮点型：torch.FloatTensor。 (默认)
2. 64位浮点型：torch.DoubleTensor。
3. 16位整型：torch.ShortTensor。
4. 32位整型：torch.IntTensor。
5. 64位整型：torch.LongTensor。

除以上数字类型外，还有 byte和char 型
"""

long = tensor.long()  # tensor([2])  long64
half = tensor.half()  # tensor([2.3320], dtype=torch.float16)
int_t = tensor.int()  # tensor([2], dtype=torch.int32)
flo = tensor.float()  # tensor([2.3330])  float32
short = tensor.short()  # tensor([2], dtype=torch.int16)
ch = tensor.char()  # tensor([2], dtype=torch.int8)
bt = tensor.byte()  # tensor([2], dtype=torch.uint8)

# ------------------------------ 3.Numpy转换 -------------------------
"""
Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源。但这也意味着，如果其中一个变了，另外一个也会随之改变。
"""

a = torch.randn(2, 3)
b = a.numpy()
"""tensor转换为numpy
>>> print(a, a.shape, type(a))
tensor([[-2.8169,  0.1127, -0.2912],
        [ 0.3958, -1.2278,  2.0335]]) torch.Size([2, 3]) <class 'torch.Tensor'>

>>> print(b, b.shape, type(b))
[[-2.81688547  0.1126704  -0.29115945]
 [ 0.39579567 -1.22784209  2.03350854]] (2, 3) <class 'numpy.ndarray'>
"""

aa = np.random.randn(2, 3)
bb = torch.from_numpy(aa)
"""numpy转换为tensor
>>> print(aa, aa.shape, type(aa))
[[ 0.12939299  0.24575266  0.27196222]
 [-0.27205402  2.22990158  1.87830408]] (2, 3) <class 'numpy.ndarray'>

>>> print(bb, bb.shape, type(bb))
tensor([[ 0.1294,  0.2458,  0.2720],
        [-0.2721,  2.2299,  1.8783]], dtype=torch.float64) torch.Size([2, 3]) <class 'torch.Tensor'>
"""

# ------------------------------ 4.设备间转换 -------------------------
"""
一般情况下可以使用.cuda方法将tensor移动到gpu，这步操作需要cuda设备支持
使用.cpu方法将tensor移动到cpu(), 使用.cuda()将数据移动到gpu
"""

# cpu_a = torch.rand(4, 3)
# gpu_a = cpu_a.cuda()
# cpu_b = gpu_a.cpu()
"""
>>> print(cpu_a.type())
torch.FloatTensor
>>> print(gpu_a.type())
torch.cuda.FloatTensor
>>> print(cpu_b.type())
torch.FloatTensor
"""

# 如果我们有多GPU的情况，可以使用to方法来确定使用那个设备，这里只做个简单的实例
"""
# 使用torch.cuda.is_available()来确定是否有cuda设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 将tensor传送到设备
gpu_b=cpu_b.to(device)
gpu_b.type()
"""

# ------------------------------ 5.初始化 ----------------------------
"""
Pytorch中有许多默认的初始化方法可以使用
"""
# 使用[0,1]均匀分布随机初始化二维数组
rnd = torch.rand(5, 3)
# 初始化，使用1填充
one = torch.ones(2, 2)
# 初始化，使用0填充
zero = torch.zeros(2, 2)
# 初始化一个单位矩阵，即对角线为1 其他为0
eye = torch.eye(2, 2)
"""
>>> print(rnd)
tensor([[0.8415, 0.1301, 0.8953],
        [0.7142, 0.8432, 0.1240],
        [0.2347, 0.3675, 0.5906],
        [0.9613, 0.0037, 0.5290],
        [0.4023, 0.6651, 0.0144]])

>>> print(one)
tensor([[1., 1.],
        [1., 1.]])

>>> print(zero)
tensor([[0., 0.],
        [0., 0.]])

>>> print(eye)
tensor([[1., 0.],
        [0., 1.]])
"""

# ------------------------------ 6.常用方法 ---------------------------
"""
PyTorch中对张量的操作api 和 NumPy 非常相似，如果熟悉 NumPy 中的操作，那么他们二者基本是一致的
"""
"""
x = torch.randn(3, 3)
print(x)
    tensor([[ 0.2462, -1.1556, -0.1816],
            [-0.6253, -1.7717,  1.7138],
            [-0.3104, -0.2461, -0.3915]])

# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)
print(max_value, max_idx)
    tensor([ 0.2462,  1.7138, -0.2461]) tensor([0, 2, 1])

# 每行 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)
    tensor([-1.0911, -0.6832, -0.9480])

y = torch.randn(3, 3)
z = x + y
print(z)
    tensor([[-0.1790, -1.8821,  0.1292],
            [-0.3455, -2.2804,  1.4619],
            [-0.5108,  0.3400, -1.7599]])

# 正如官方60分钟教程中所说，以_为结尾的，均会改变调用值, add 完成后x的值改变了
x.add_(y)
print(x)
    tensor([[-0.1790, -1.8821,  0.1292],
            [-0.3455, -2.2804,  1.4619],
            [-0.5108,  0.3400, -1.7599]])
"""


