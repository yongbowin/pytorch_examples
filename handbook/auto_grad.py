# -*- coding:utf-8 -*-
"""
@Time  : 3/12/20 3:01 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : auto_grad.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter2/2.1.2-pytorch-basics-autograd.ipynb
"""

import torch
from torch.autograd.function import Function

# ------------------------ 1.使用PyTorch计算梯度数值 ------------------------
"""
PyTorch的Autograd模块实现了深度学习的算法中的反向传播求导数，在张量（Tensor类）上的所有操作，Autograd都能为他们自动提供微分，
简化了手动计算导数的复杂过程。在0.4以前的版本中，Pytorch使用Variable类来自动计算所有的梯度。
Variable类主要包含三个属性： 
    data：保存Variable所包含的Tensor；
    grad：保存data对应的梯度，grad也是个Variable，而不是Tensor，它和data的形状一样；
    grad_fn：指向一个Function对象，这个Function用来反向传播计算输入的梯度。

从0.4起， Variable 正式合并入Tensor类，通过Variable嵌套实现的自动微分功能已经整合进入了Tensor类中。虽然为了代码的兼容性还是
可以使用Variable(tensor)这种方式进行嵌套，但是这个操作其实什么都没做。所以，以后的代码建议直接使用Tensor类进行操作，因为官方文档
中已经将Variable设置成过期模块。要想通过Tensor类本身就支持了使用autograd功能，只需要设置.requries_grad=True

Variable类中的的grad和grad_fn属性已经整合进入了Tensor类中.在张量创建时，通过设置 requires_grad 标识为Ture来告诉Pytorch
需要对该张量进行自动求导，PyTorch会记录该张量的每一步操作历史并自动计算
"""

"""
PyTorch会自动追踪和记录对与张量的所有操作，当计算完成后调用.backward()方法自动计算梯度并且将计算结果保存到grad属性中。
"""
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)

z = torch.sum(x + y)

"""
>>> print(x)
tensor([[0.4726, 0.0475, 0.7037, 0.2869, 0.4234],
        [0.7746, 0.0753, 0.1488, 0.9336, 0.8356],
        [0.3281, 0.0896, 0.5626, 0.7944, 0.5212],
        [0.1592, 0.4565, 0.2649, 0.5818, 0.3899],
        [0.9224, 0.1142, 0.1059, 0.5006, 0.6779]], requires_grad=True)
>>> print(y)
tensor([[0.9963, 0.4550, 0.8037, 0.5922, 0.3045],
        [0.0207, 0.9251, 0.4665, 0.7278, 0.3434],
        [0.9745, 0.3942, 0.2186, 0.2788, 0.7698],
        [0.5739, 0.9537, 0.6821, 0.3083, 0.6439],
        [0.9699, 0.5692, 0.4185, 0.4336, 0.2055]], requires_grad=True)
>>> print(z)  # 所有元素求和
tensor(24.7466, grad_fn=<SumBackward0>)

在张量进行操作后，grad_fn已经被赋予了一个新的函数，这个函数引用了一个创建了这个Tensor类的Function对象。 
Tensor和Function互相连接生成了一个非循环图，它记录并且编码了完整的计算历史。每个张量都有一个.grad_fn属性，
如果这个张量是用户手动创建的那么这个张量的grad_fn是None。
"""
# 下面我们来调用反向传播函数，计算其梯度
z.backward()
"""简单的自动求导
>>> print(x.grad, y.grad)
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]]) 
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])

如果Tensor类表示的是一个标量（即它包含一个元素的张量），则不需要为backward()指定任何参数，但是如果它有更多的
元素，则需要指定一个gradient参数，它是形状匹配的张量。 
以上的 z.backward()相当于是z.backward(torch.tensor(1.))的简写。 这种参数常出现在图像分类中的单标签分类，
输出一个标量代表图像的标签。
"""
z = x ** 2 + y ** 3
"""
>>> print(z, z.shape)
tensor([[0.3010, 0.0380, 0.1961, 0.9003, 0.3687],
        [0.6798, 0.0985, 0.5004, 0.7540, 0.0446],
        [0.9591, 0.0563, 0.6681, 0.3549, 0.9221],
        [1.0690, 0.1520, 0.4603, 0.4100, 1.4041],
        [0.6572, 0.0447, 0.5905, 0.4237, 0.0669]], grad_fn=<AddBackward0>) torch.Size([5, 5])
"""

# 我们的返回值不是一个标量，所以需要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量
z.backward(torch.ones_like(x))
"""复杂的自动求导
>>> z.backward()
print(x.grad)  # RuntimeError: grad can be implicitly created only for scalar outputs
>>> print(z)
tensor([[1.2145e+00, 8.0179e-01, 3.3951e-01, 9.8454e-01, 7.5921e-01],
        [4.0545e-01, 8.8097e-02, 2.6056e-02, 6.9376e-01, 1.1010e+00],
        [5.2763e-01, 5.6900e-01, 4.2468e-02, 5.7891e-01, 3.6468e-01],
        [1.1329e-04, 8.5020e-01, 5.9714e-01, 1.9064e-01, 7.0475e-01],
        [2.1210e-01, 8.1691e-01, 5.9694e-01, 7.4586e-01, 1.3333e+00]],
       grad_fn=<AddBackward0>)
>>> print(x.grad)
tensor([[1.9996, 1.8721, 2.1021, 1.9885, 1.2462],
        [2.5504, 2.3313, 1.1229, 2.1825, 2.8365],
        [2.8997, 1.3244, 2.5240, 2.5269, 1.0563],
        [2.1476, 1.6009, 2.4977, 1.8490, 2.5048],
        [1.2682, 2.3044, 2.6159, 2.6736, 2.4522]])
>>> print(torch.ones_like(z), torch.ones_like(z).shape)
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]]) torch.Size([5, 5])
"""

"""
我们可以使用with torch.no_grad()上下文管理器临时禁止对已设置requires_grad=True的张量进行自动求导。
这个方法在测试集计算准确率的时候会经常用到，例如：
with torch.no_grad():
    print((x + y * 2).requires_grad)  # False
使用.no_grad()进行嵌套后，代码不会跟踪历史记录，也就是说保存的这部分记录会减少内存的使用量并且会加快少许的运算速度。
"""

"""Autograd 过程解析
为了说明Pytorch的自动求导原理，我们来尝试分析一下PyTorch的源代码，虽然Pytorch的 Tensor和 TensorBase都是使用
CPP来实现的，但是可以使用一些Python的一些方法查看这些对象在Python的属性和状态。 Python的 dir() 返回参数的属性、
方法列表。z是一个Tensor变量，看看里面有哪些成员变量。
>>> print(dir(z))
['T', '__abs__', '__add__', '__and__', '__array__', '__array_priority__', '__array_wrap__', '__bool__', '__class__', 
'__contains__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__div__', '__doc__', '__eq__', 
...
'trunc', 'trunc_', 'type', 'type_as', 'unbind', 'unflatten', 'unfold', 'uniform_', 'unique', 'unique_consecutive', 
'unsqueeze', 'unsqueeze_', 'values', 'var', 'view', 'view_as', 'where', 'zero_']

直接看几个比较主要的属性： .is_leaf：记录是否是叶子节点。通过这个属性来确定这个变量的类型 在官方文档中所说的“graph leaves”，“leaf variables”，
都是指像x，y这样的手动创建的、而非运算得到的变量，这些变量成为创建变量。 像z这样的，是通过计算后得到的结果称为结果变量。
一个变量是创建变量还是结果变量是通过.is_leaf来获取的
>>> print("x.is_leaf="+str(x.is_leaf))  # x.is_leaf=True   创建变量
>>> print("z.is_leaf="+str(z.is_leaf))  # z.is_leaf=False  结果变量
x是手动创建的没有通过计算，所以他被认为是一个叶子节点也就是一个创建变量，而z是通过x与y的一系列计算得到的，所以不是叶子结点也就是结果变量。

为什么我们执行z.backward()方法会更新x.grad和y.grad呢？ .grad_fn属性记录的就是这部分的操作，虽然.backward()方法也是CPP实现的，
但是可以通过Python来进行简单的探索。grad_fn：记录并且编码了完整的计算历史
>>> print(z.grad_fn)  # <AddBackward0 object at 0x7f38c21d5390>
grad_fn是一个AddBackward0类型的变量 AddBackward0这个类也是用Cpp来写的，但是我们从名字里就能够大概知道，他是加法(ADD)的反向传播（Backward），

看看里面有些什么东西
>>> print(dir(z.grad_fn))
['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', 
...
'__subclasshook__', '_register_hook_dict', 'metadata', 'name', 'next_functions', 'register_hook', 'requires_grad']

next_functions就是grad_fn的精华

>>> print(z.grad_fn.next_functions)
((<PowBackward0 object at 0x7fb3c7411438>, 0), (<PowBackward0 object at 0x7fb3c7411470>, 0))

next_functions是一个tuple of tuple of PowBackward0 and int。
为什么是2个tuple ？ 因为我们的操作是z= x**2+y**3 刚才的AddBackward0是相加，而前面的操作是乘方 PowBackward0。
tuple第一个元素就是x相关的操作记录
>>> xg = z.grad_fn.next_functions[0][0]
>>> print(dir(xg))
['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', 
...
'__subclasshook__', '_register_hook_dict', 'metadata', 'name', 'next_functions', 'register_hook', 'requires_grad']

继续深挖
xg = z.grad_fn.next_functions[0][0]
x_leaf = xg.next_functions[0][0]
>>> print(x_leaf)  # <AccumulateGrad object at 0x7f54c674de80>
>>> print(dir(x_leaf))
['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
 ...
'_register_hook_dict', 'metadata', 'name', 'next_functions', 'register_hook', 'requires_grad', 'variable']
>>> print(type(x_leaf))  # <class 'AccumulateGrad'>


在PyTorch的反向图计算中，AccumulateGrad类型代表的就是叶子节点类型，也就是计算图终止节点。AccumulateGrad类中有一个.variable属性指向叶子节点。

>>> xg = z.grad_fn.next_functions[0][0]
>>> x_leaf = xg.next_functions[0][0]
>>> print(x_leaf.variable)
tensor([[0.3574, 0.1786, 0.8387, 0.4478, 0.5188],
        [0.3775, 0.2761, 0.9287, 0.3029, 0.2472],
        [0.9633, 0.6428, 0.8715, 0.5042, 0.7870],
        [0.7327, 0.1474, 0.5699, 0.2333, 0.5584],
        [0.9348, 0.3190, 0.8691, 0.2535, 0.2636]], requires_grad=True)

这个.variable的属性就是我们的生成的变量x
>>> print("x_leaf.variable的id:"+str(id(x_leaf.variable)))
x_leaf.variable的id:139637840054384
>>> print("x的id:"+str(id(x)))
x的id:139637840054384

这样整个规程就很清晰了：
    1.当我们执行z.backward()的时候。这个操作将调用z里面的grad_fn这个属性，执行求导的操作。
    2.这个操作将遍历grad_fn的next_functions，然后分别取出里面的Function（AccumulateGrad），
    执行求导操作。这部分是一个递归的过程直到最后类型为叶子节点。
    3.计算出结果以后，将结果保存到他们对应的variable 这个变量所引用的对象（x和y）的 grad这个属性里面。
    4.求导结束。所有的叶节点的grad变量都得到了相应的更新
最终当我们执行完z.backward()之后，x和y里面的grad值就得到了更新。
"""

"""扩展Autograd
如果需要自定义autograd扩展新的功能，就需要扩展Function类。因为Function使用autograd来计算结果和梯度，并对操作历史进行编码。
在Function类中最主要的方法就是forward()和backward()他们分别代表了前向传播和反向传播。

一个自定义的Function需要一下三个方法：
    1. __init__ (optional)：如果这个操作需要额外的参数则需要定义这个Function的构造函数，不需要的话可以忽略。
    2. forward()：执行前向传播的计算代码
    3. backward()：反向传播时梯度计算的代码。 参数的个数和forward返回值的个数一样，每个参数代表传回到此操作的梯度。
"""

"""
# 引入Function便于扩展
from torch.autograd.function import Function
"""


# 定义一个乘以常数的操作(输入参数是张量)
# 方法必须是静态方法，所以要加上@staticmethod 
class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx 用来保存信息这里类似self，并且ctx的属性可以在backward中调用
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # 返回的参数要与输入的参数一样.
        # 第一个输入为3x3的张量，第二个为一个常数
        # 常数的梯度必须是 None.
        return grad_output, None


a = torch.rand(3, 3, requires_grad=True)
b = MulConstant.apply(a, 5)  # 调用forward方法
"""
>>> print("a:" + str(a))
a:tensor([[0.6683, 0.9953, 0.6180],
        [0.4804, 0.5893, 0.8309],
        [0.2677, 0.4305, 0.1125]], requires_grad=True)

>>> print("b:" + str(b))  # b为a的元素乘以5
b:tensor([[3.3415, 4.9766, 3.0900],
        [2.4018, 2.9466, 4.1545],
        [1.3386, 2.1524, 0.5625]], grad_fn=<MulConstantBackward>)
"""
# 反向传播，返回值不是标量，所以backward方法需要参数
b.backward(torch.ones_like(a))
"""
>>> print(a.grad)  # 梯度为1
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
>>> print(b.grad)  # None
"""
