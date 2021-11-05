- [2021-11-1](#2021-11-1)
  - [PLAN](#plan)
  - [进入系统填报信息](#进入系统填报信息)
    - [Penn State](#penn-state)
    - [MADISON](#madison)
  - [微生物结论](#微生物结论)
  - [修改甲基化文件 转成bed](#修改甲基化文件-转成bed)
- [2021-11-2](#2021-11-2)
  - [PLAN](#plan-1)
  - [deep learning approach antibiotics](#deep-learning-approach-antibiotics)
- [2021-11-3](#2021-11-3)
  - [PLAN](#plan-2)
  - [机器学习作业总结](#机器学习作业总结)
    - [loop 固定写法](#loop-固定写法)
    - [全连接模型](#全连接模型)
    - [卷积](#卷积)
    - [卷积实现相关参数](#卷积实现相关参数)
- [2021-11-4](#2021-11-4)
  - [PLAN](#plan-3)
  - [微生物](#微生物)
  - [卷积神经网络](#卷积神经网络)
    - [resnet](#resnet)
    - [卷积变种](#卷积变种)
- [2021-11-5](#2021-11-5)
  - [PLAN](#plan-4)
  - [resnet实现代码](#resnet实现代码)
  - [微生物introduction 修改](#微生物introduction-修改)
    - [原introduction 逻辑](#原introduction-逻辑)
    - [修改逻辑](#修改逻辑)

# 2021-11-1
## PLAN
+ **修改PS**
+ **微生物作图**
+ **进入系统填报信息**
+ **修改甲基化文件 转成bed**

## 进入系统填报信息
    
### Penn State
+ 地址 No.29 Wangjiang Road, Chengdu, Sichuan, China,610064
+ 密码 wzy851234wzy851234
+ 登录号 zxw5399@psu.edu

### MADISON
+ 密码 5#DvMnQwAwHhL+r
+ user name pry0921

## 微生物结论
无植物 各种处理alpha beta多样性都没有区别

## 修改甲基化文件 转成bed
代码
```python
#   API trans_mCfile2bed
#   trans_mCfile2bed(path,input_file_name)
#   path 文件相应路径 file_name 甲基化文件名称

def judge_if_mC(line):
    if str(line).endswith("1"):
        return True
    return False

def trans2bed(line):
    line_list=line.split("\t")
    chr_num="chr"+line_list[0]
    start_position=line_list[1]
    end_position=str(int(start_position)+1)
    name=line_list[3]
    length_of_mC="1"
    strand=line_list[2]
    linr_out="\t".join([chr_num,start_position,end_position,name,length_of_mC,strand])
    return linr_out

def trans_mCfile2bed(path,input_file_name):
    # path= "/home/ubuntu/Arabidopsis/Arabidopsis_sequence/Ag-0/"
    # input_file_name="GSM1085193_mC_calls_Ag_0.tsv"
    f = open(path+input_file_name,"r")
    mc_data=f.read()
    mc_lines=mc_data.split("\n")
    [judge_if_mC(i) for i in mc_lines]
    mc_true_lines=list(filter(judge_if_mC,mc_lines))
    list(mc_true_lines)
    mc_true_lines_out=[trans2bed(line) for line in mc_true_lines]
    output_file_name=input_file_name[20:-4]+".mC.bed"
    f_bed=open(path+output_file_name,"w")
    f_bed.write("\n".join(mc_true_lines_out))
    output_file_name=input_file_name[20:-4]+".mC.bed"
    return output_file_name
```

# 2021-11-2
## PLAN
+ **微生物功能分析图**
+ **GCN启发论文**

## deep learning approach antibiotics
方法 MPNN 消息传递神经网络 GNN的一种

# 2021-11-3
    
## PLAN
+ **微生物热图**
+ **MINSET数据集overvoew pytorch回忆 基于教材 Dive into**

## 机器学习作业总结
    
### loop 固定写法
当定义好训练模型后，不需要修改train和test loop是固定的，如以下框架
```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

### 全连接模型
           
```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

### 卷积
```python
class CoNeuralNetwork(nn.Module):
    def __init__(self):
        super(CoNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack =  nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
```

### 卷积实现相关参数
+ kernel 参数 只涉及卷积核形状
+ stride  步幅

在计算互相关时，卷积窗口从输入张量的左上角开始，向下和向右滑动。 在前面的例子中，我们默认每次滑动一个元素。 但是，有时候为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。

![stride](https://zh-v2.d2l.ai/_images/conv-stride.svg)

例子中 **垂直步幅为  3 ，水平步幅为  2  的二维互相关运算**

pytorch实现

```python
## 右下步幅一致情况
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
## 右下步幅不一致情况
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
```
同时在上面例子中卷积核 padding stride都不同

+ padding 在行和列填充

![padding](https://zh-v2.d2l.ai/_images/conv-pad.svg)

例子中添加了一列一行

pytorch 实现
```python
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
```
+ output 下一层的数量 决定了卷积数量


# 2021-11-4

## PLAN
+ **拉美研究ppt初稿**
+ **百面前向神经网络**
+ **微生物调参**

## 微生物
H1-F-vs-H2-F-vs-H3-F0.8601 0.006 **
H1-NF-vs-H2-NF-vs-H3-NF 0.3333 0.026 *

## 卷积神经网络

### resnet
核心 每－层都不比上一层差 每一层新网络训练残差

### 卷积变种
反卷积	转置卷积
实例 AE  VAE GAN

# 2021-11-5

## PLAN
+ **论文修改introduction设计**
+ 卷积神经网络训练方法 loss计算overview
+ **resnet实现 pytorch**

## resnet实现代码

```python
#model.py

import torch.nn as nn
import torch

#18/34
class BasicBlock(nn.Module):
    expansion = 1 #每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)#BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x #捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

#50,101,152
class Bottleneck(nn.Module):
    expansion = 4#4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,#输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
```

这套代码可以训练MNSIT

## 微生物introduction 修改

### 原introduction 逻辑
从hitchhiking现象说起，影响hitchhiking有物理化学环境 

氮浓度是直接影响微生物群落结构的关键因素 

### 修改逻辑
hitchhiking现象介绍 影响hitchhiking已有的报道 氮可以影响微生物群落 但不知道是否影响hitchhiking 

另一方面 植物分泌物作为土壤细菌的营养来源 是否促进hitchhiking 

植物和氮对hitchhiking的关系 