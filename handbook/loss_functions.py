# -*- coding:utf-8 -*-
"""
@Time  : 3/9/20 6:40 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : loss_functions.py
@Desc  : reference from https://github.com/zergtant/pytorch-handbook/blob/master/chapter2/2.2-deep-learning-basic-mathematics.ipynb
        损失函数
"""

import torch.nn as nn

"""
损失函数（loss function）是用来估量模型的预测值(我们例子中的output)与真实值（例子中的y_train）的不一致程度，它是一个非负实值函数，
损失函数越小，模型的鲁棒性就越好。 我们训练模型的过程，就是通过不断的迭代计算，使用梯度下降的优化算法，使得损失函数越来越小。损失函数越小
就表示算法达到意义上的最优。
这里有一个重点：因为PyTorch是使用mini-batch来进行计算的，所以损失函数的计算出来的结果已经对mini-batch取了平均
常见（PyTorch内置）的损失函数有以下几个：

1. `nn.L1Loss` : 
    输入x和目标y之间差的绝对值，要求 x 和 y 的维度要一样（可以是向量或者矩阵），得到的 loss 维度也是对应一样的

2. `nn.NLLLoss` : 
    用于多分类的负对数似然损失函数, NLLLoss中如果传递了weights参数，会对损失进行加权

3. `nn.MSELoss` : 
    均方损失函数 ，输入x和目标y之间均方差

4. `nn.CrossEntropyLoss` : 
    多分类用的交叉熵损失函数，LogSoftMax和NLLLoss集成到一个类中，会调用nn.NLLLoss函数，我们可以理解为
    CrossEntropyLoss()=log_softmax() + NLLLoss()
    因为使用了NLLLoss，所以也可以传入weight参数, 所以一般多分类的情况会使用这个损失函数

5. `nn.BCELoss` : 
    计算 x 与 y 之间的二进制交叉熵。与NLLLoss类似，也可以添加权重参数, 用的时候需要在该层前面加上 Sigmoid 函数。

nn.AdaptiveLogSoftmaxWithLoss
nn.BCEWithLogitsLoss
nn.CosineEmbeddingLoss
nn.CTCLoss
nn.HingeEmbeddingLoss
nn.KLDivLoss
nn.MarginRankingLoss
nn.MultiLabelMarginLoss
nn.MultiLabelSoftMarginLoss
nn.MultiMarginLoss
nn.PoissonNLLLoss
nn.SoftMarginLoss
nn.TripletMarginLoss
nn.SmoothL1Loss
nn.NLLLoss2d
"""

