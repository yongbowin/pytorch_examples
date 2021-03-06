# -*- coding:utf-8 -*-
"""
@Time  : 3/10/20 10:46 AM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : classifier.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/4_cifar10_tutorial.ipynb
        raw from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import datetime

import utils

"""
关于数据？
一般情况下处理图像、文本、音频和视频数据时，可以使用标准的Python包来加载数据到一个numpy数组中。 然后把这个数组转换成 torch.*Tensor。

1.图像可以使用 Pillow, OpenCV
2.音频可以使用 scipy, librosa
3.文本可以使用原始Python和Cython来加载，或者使用 NLTK或 SpaCy 处理
4.特别的，对于图像任务，我们创建了一个包 torchvision，它包含了处理一些基本图像数据集的方法。这些数据集包括 Imagenet, CIFAR10, MNIST 等。
除了数据加载以外，torchvision 还包含了图像转换器， torchvision.datasets 和 torch.utils.data.DataLoader。

torchvision包不仅提供了巨大的便利，也避免了代码的重复。

在这个教程中，我们使用CIFAR10数据集，它有如下10个类别 ：‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, 
‘ship’, ‘truck’。CIFAR-10的图像都是 3x32x32大小的，即，3颜色通道，32x32像素。
"""

""" 代码流程
训练一个图像分类器, 依次按照下列顺序进行：
1.使用torchvision加载和归一化CIFAR10训练集和测试集
2.定义一个卷积神经网络
3.定义损失函数
4.在训练集上训练网络
5.在测试集上测试网络
"""

"""
1. 读取和归一化 CIFAR10
使用torchvision可以非常容易地加载CIFAR10。
"""


class Net(nn.Module):
    """
    定义一个卷积神经网络
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    """
    展示一些训练图像。
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_data():
    """
    torchvision的输出是[0,1]的PILImage图像，我们把它转换为归一化范围为[-1, 1]的张量。
    """

    """
    `transforms.Compose()` :  Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    -------------------------
    `transforms.CenterCrop()` : Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    -------------------------
    `transforms.ToTensor()` : Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    -------------------------
    `transforms.Normalize()` : Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 下载数据集
    trainset = torchvision.datasets.CIFAR10(root="/home/wyb/data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="/home/wyb/data", train=False, download=True, transform=transform)

    """数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
    reference from https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/#torchutilsdata
    
    torch.utils.data.DataLoader(dataset,  # 加载数据的数据集。
                                batch_size=1,  # (int, optional) – 每个batch加载多少个样本(默认: 1)。
                                shuffle=False,  # (int, optional) – 每个batch加载多少个样本(默认: 1)。
                                sampler=None,  # (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
                                num_workers=0,  # (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
                                collate_fn= < function default_collate >, 
                                pin_memory=False, 
                                drop_last=False)  # (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除
                                                    最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最
                                                    后一个batch将更小。(默认: False)
    
    --------------- datasets sampler
    torch.utils.data.sampler.SequentialSampler(data_source)  # data_source是采样的数据集, 样本元素顺序排列，始终以相同的顺序。
    torch.utils.data.sampler.RandomSampler(data_source)  # 样本元素随机，没有替换。
    torch.utils.data.sampler.SubsetRandomSampler(indices)  # indices (list) – 索引的列表, 样本元素从指定的索引列表中随机抽取，没有替换
    torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)  # 样本元素来自于[0,..,len(weights)-1]，
                                                                                            给定概率（weights）。
                                                                                            weights (list) – 权重列表。没必要加起来为1。
                                                                                            num_samples (int) – 抽样数量。
    """
    # 数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)
    """假如 num_workers != 0 会出现如下运行时错误：
    
    Exception in thread Thread-1:
    Traceback (most recent call last):
      File "/home/wyb/anaconda3/lib/python3.6/threading.py", line 916, in _bootstrap_inner
        self.run()
      File "/home/wyb/anaconda3/lib/python3.6/threading.py", line 864, in run
        self._target(*self._args, **self._kwargs)
      File "/home/wyb/anaconda3/lib/python3.6/multiprocessing/resource_sharer.py", line 139, in _serve
        signal.pthread_sigmask(signal.SIG_BLOCK, range(1, signal.NSIG))
      File "/home/wyb/anaconda3/lib/python3.6/signal.py", line 60, in pthread_sigmask
        sigs_set = _signal.pthread_sigmask(how, mask)
    ValueError: signal number 32 out of range
    """

    """
    # 10个类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 获取随机数据
    dataiter = iter(trainloader)  # iter()是内置函数
    # print(dataiter)
    images, labels = dataiter.next()  # 逐个遍历图像
    # 展示图像
    imshow(torchvision.utils.make_grid(images))
    # 显示图像标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    """

    return trainloader, testloader


if __name__ == '__main__':
    # ----------------- 1.定义神经网络 -----------------
    net = Net()

    # ----------------- 2.定义损失函数和优化器 -----------
    # 使用交叉熵作为损失函数，使用带动量的随机梯度下降。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # ----------------- 3.加载数据 --------------------
    trainloader, testloader = load_data()

    # ----------------- 4.训练网络 --------------------
    # 只需在数据迭代器上循环，将数据输入给网络，并优化。
    """
    for epoch in range(2):  # 多批次循环
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
    
            # 梯度置0
            optimizer.zero_grad()
    
            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # 打印状态信息
            running_loss += loss.item()
            if i % 2000 == 1999:    # 每2000批次打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')
    """
    start_time, _ = utils.get_time()
    for epoch in range(2):  # 运行两个epoch
        running_loss = 0.0  # 每个epoch都需要重新初始化loss为 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):  # 从0开始迭代
            # 获取输入
            inputs, labels = data
            """
            inputs : torch.Size([4, 3, 32, 32])
            tensor([[[[-0.9843, -0.9843, -0.9922,  ..., -1.0000, -0.9922, -0.9765],
                      ...,
                      [-0.9922, -0.9922, -0.9922,  ..., -0.9137, -0.9216, -0.8902]]],
                      ...,
                    [[[ 0.4196,  0.3882,  0.4039,  ...,  0.6314,  0.6392,  0.6314],
                      ...,
                      [ 0.5765,  0.5686,  0.5765,  ..., -0.0039,  0.0980,  0.0902]]]])
            
            labels : tensor([1, 4, 5, 0])  torch.Size([4])
            """
            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，得到输出
            outputs = net(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 损失（误差）反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()  # tensor中只有一个至时，使用item()进行获取
            if i % 2000 == 1999:  # 每2000步打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    end_time, _ = utils.get_time()

    # 保存训练好的模型
    """
    Example:
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, 'tensor.pt')
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)

    ----------
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    """
    PATH = "./_net.pth"
    torch.save(net.state_dict(), PATH)

    # ----------------- 5.在测试集上测试网络 ------------
    """
    # 第一步，显示测试集中的图片并熟悉图片内容。
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # 10个类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    """

    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    """
    start_time_pred, _ = utils.get_time()
    correct = 0
    total = 0
    correct_num = 0
    nums = 0
    with torch.no_grad():  # 测试时不需要梯度的更新
        for data in tqdm(testloader):
            nums += 1
            images, labels = data
            outputs = net(images)
            # 预测时没有loss
            _, predicted = torch.max(outputs.data, 1)
            """
            images.shape: torch.Size([4, 3, 32, 32])
            labels shape: torch.Size([4])
            predicted shape: torch.Size([4])
            """
            total += labels.size(0)  # 求测试集的样本个数
            correct += (predicted == labels).sum().item()  # 求出的值为相等个数的 len(predicted) 倍, 10标签就是10倍

            if (predicted == labels).sum().item() > 0:  # 预测正确
                correct_num += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print("self computing accuracy: " + str(correct_num / nums) + " correct nums: " + str(correct_num) + " nums: " + str(nums))

    end_time_pred, _ = utils.get_time()

    print("Model training, start at: " + str(start_time) + ", end at: " + str(end_time))
    print("Model predicting, start at: " + str(start_time_pred) + ", end at: " + str(end_time_pred))
    """
    Accuracy of the network on the 10000 test images: 56 %
    self computing accuracy: 0.9636 correct nums: 2409 nums: 2500
    Model training, start at: 2020-03-11 15:44:34, end at: 2020-03-11 15:48:34
    Model predicting, start at: 2020-03-11 15:48:34, end at: 2020-03-11 15:48:37
    """

    # ------ 6.在测试集上检查哪些类预测的好，哪些不好 -------
    """
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    """
    # class_correct = list(0. for i in range(10))  # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # class_total = list(0. for i in range(10))

    # ----------------- 7.在GPU上训练网络 --------------
    """
    把一个神经网络移动到GPU上训练就像把一个Tensor转换GPU上一样简单。并且这个操作会递归遍历有所模块，并将其参数和缓冲区转换为CUDA张量。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)  # cpu

    """
    然后这些方法将递归遍历所有模块并将模块的参数和缓冲区 转换成CUDA张量 ： net.to(device)
    记住：inputs, labels 和 images 也要转换 : inputs, labels = inputs.to(device), labels.to(device)
    
    为什么我们没注意到GPU的速度提升很多？那是因为网络非常的小。
    
    用如下方式把一个模型放到GPU上 : 
    device = torch.device("cuda:0")
    model.to(device)
    
    然后复制所有的张量到GPU上：
    mytensor = my_tensor.to(device)
    
    请注意，只调用my_tensor.to(device)并没有复制张量到GPU上，而是返回了一个copy。所以你需要把它赋值给一个新的张量并在GPU上使用这个张量。
    """
