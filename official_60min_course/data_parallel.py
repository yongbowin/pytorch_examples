# -*- coding:utf-8 -*-
"""
@Time  : 3/11/20 4:41 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : data_parallel.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/5_data_parallel_tutorial.ipynb
        数据并行
        raw from https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import os

# 设置GPU可见
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

"""
在多GPU上执行前向和反向传播是自然而然的事。 但是PyTorch默认将只使用一个GPU。
使用DataParallel可以轻易的让模型并行运行在多个GPU上。
model = nn.DataParallel(model)
"""


class RandomDataset(Dataset):
    """
    制作一个虚拟数据集
    """

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    """简单模型
    作为演示，我们的模型只接受一个输入，执行一个线性操作，然后得到结果。 说明：DataParallel能在任何模型（CNN，RNN，Capsule Net等）上使用。
    我们在模型内部放置了一条打印语句来打印输入和输出向量的大小。注意批次的秩为0时打印的内容。

    input_size = 5
    output_size = 2
    """

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())
        return output


if __name__ == '__main__':
    # Parameters and DataLoaders
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------- 1.随机创建一个数据集 --------------
    # DataLoader的用法，输入的是一个Dataset，是一个tensor, 100×5
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

    # ------------- 2.创建一个模型和数据并行 -----------
    """
    首先，我们需要创建一个模型实例和检测我们是否有多个GPU。 如果有多个GPU，使用nn.DataParallel来包装我们的模型。 
    然后通过model.to(device)把模型放到GPU上。
    """
    model = Model(input_size, output_size)
    if torch.cuda.device_count() > 1:
        print("Current device has " + str(torch.cuda.device_count()) + " GPUs.")
        model = nn.DataParallel(model)  # 使用GPU数据并行
    model.to(device)  # 将模型放入GPU

    # ------------- 3.运行模型 ----------------------
    # 现在可以看到输入和输出张量的大小。
    for data in tqdm(rand_loader):
        input = data.to(device)  # 将数据放入GPU
        output = model(input)
        print("Outside: input size", input.size(), "output_size", output.size())

    """
    DataParallel会自动的划分数据，并将作业发送到多个GPU上的多个模型。 并在每个模型完成作业后，收集合并结果并返回。
    """

    """batch_size=30, forward一次处理一个batch的数据
    
    当有1个GPU或者没有GPU时，显示：
    In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
    
    --------------
    当有2个GPU时，显示：
    Current device has 2 GPUs.
            In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
            In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
            In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
    
    --------------
    当有8个GPU时，显示： (数据会自动分配到各个GPU上，最后进行汇总), 并不是每个卡都占用，会随机分发
    Current device has 8 GPUs.
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
            In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
    """
