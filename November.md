- [2021-11-1](#2021-11-1)
  - [PLAN](#plan)
  - [进入系统填报信息](#进入系统填报信息)
    - [Penn State](#penn-state)
    - [MADISON](#madison)
  - [无植物微生物结论](#无植物微生物结论)
    - [微生物不同氮肥浓度](#微生物不同氮肥浓度)
    - [不同氮肥浓度是否引发hitchhiking](#不同氮肥浓度是否引发hitchhiking)
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
  - [在线pdf](#在线pdf)
  - [卷积神经网络训练方法 batch noemalization](#卷积神经网络训练方法-batch-noemalization)
- [2021-11-6](#2021-11-6)
  - [PLAN](#plan-5)
  - [Penn State 问题](#penn-state-问题)
  - [UA账户](#ua账户)
  - [单细胞论文](#单细胞论文)
    - [结论](#结论)
    - [方法](#方法)
  - [卷积神经网络模块细节](#卷积神经网络模块细节)
    - [全局平均池化](#全局平均池化)
    - [瓶颈模型](#瓶颈模型)
  - [进一步关注文章](#进一步关注文章)
- [2021-11-7](#2021-11-7)
  - [PLAN](#plan-6)
  - [Penn State 问题](#penn-state-问题-1)
  - [Graph Embedding](#graph-embedding)
    - [Deep work](#deep-work)
- [2021-11-7](#2021-11-7-1)
  - [PLAN](#plan-7)
  - [Diversity Statement](#diversity-statement)
- [2021-11-8](#2021-11-8)
  - [PLAN](#plan-8)

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

## 无植物微生物结论

### 微生物不同氮肥浓度
H1-F-vs-H2-F-vs-H3-F0.8601 0.006 **
H1-NF-vs-H2-NF-vs-H3-NF 0.3333 0.026 *

### 不同氮肥浓度是否引发hitchhiking
H1-NF H1-F 0.343565
H2-NF H2-F 0.08328
H3-NF H3-F 0.001832 **

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
+ **卷积神经网络训练方法 loss计算overview**
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

## 在线pdf
<iframe src="https://docs.google.com/gview?embedded=true&url=https://github.com/YangyiLab/Ziyuan-Wang-Daily-Log/raw/master/June.pdf" style="width:600px; height:500px;" frameborder="0"></iframe>

https://docs.google.com/gview?embedded=true&url=*https://github.com/YangyiLab/Ziyuan-Wang-Daily-Log/raw/master/June.pdf* 修改斜体部分

## 卷积神经网络训练方法 batch noemalization
加入BN batch Normalization 后 
BN层需要多学习 $\beta$  $\gamma$ 方法 BP

# 2021-11-6

## PLAN
+ **Penn State & UA 材料准备**
+ **阅读单细胞文献1篇**
+ **阅读G4文献1**
+ **卷积神经网络百面**

## Penn State 问题
+ Describe your most meaningful independent research experience. Your answer should include the scientific questions or hypotheses you asked, the experimental design you used to test the hypothesis or question, your results, and the conclusions you drew about the biological processes you studied. Describe your role in the project: Were you part of a research team? What parts of the project were you responsible for? 500 words maximum.

+ Generally, it's considered best practice for a student to move to a new institution for Graduate School. If you are currently a Penn State, University Park campus student, please tell us why staying here is a good choice for you.

+ Briefly describe an academic obstacle you faced or a challenge you have met. What strategies did you use to overcome the challenge? If you faced a future technical problem in your research, how would you overcome this hurdle. 250 words maximum

+ What are your current research interests, why is that field/area attractive to you, and which faculty in the department would allow you to pursue these interests? What do you see yourself doing in 10 years from now? 250 words maximum

+ Socially diverse groups do better science, are more productive and innovative, and make better decisions. Describe any past activities that have supported diversity and inclusion. How would you support or contribute to diversity and inclusion in the Biology program? 250 words maximum

+ Describe something in your academic life that you are most proud of. 250 words maximum.

## UA账户
邮箱 13230859192@163.com
密码 b#UFaXFK千*.7EPc

## 单细胞论文

### 结论
整合 rna-seq atac-seq 等多组学进行降维 节点分类等

我们展示了 SIMBA 提供了一个单一的框架，允许以通用的方式制定不同的单细胞分析问题，从而简化新分析的开发和其他单细胞模式的集成。

### 方法
图嵌入 可以利用 GCN 代替吗？

## 卷积神经网络模块细节

### 全局平均池化

目的：取代全连接层
优势：降低计算量提高可解释性

### 瓶颈模型

利用1*1卷积做升降维	尤其在大卷积（通道较多时）

## 进一步关注文章
+ https://com-mendeley-prod-publicsharing-pdfstore.s3.eu-west-1.amazonaws.com/0524-PUBMED/10.1371/journal.pone.0113955/pone_0113955_pdf.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCWV1LXdlc3QtMSJIMEYCIQDbQrxMJTyRt5vZU30g41wj6QppW5YXAI1S7Sx2aywA5QIhAPqo0ZqnGqWXaynlhHKkznHvtazdOh62jdNOuDTj0VkTKowECIn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAxoMMTA4MTY2MTk0NTA1Igw0TdvfvuOyFdjenc0q4AN8yJVtQgt%2FGWfXBmR7SV8qPjjUM16%2F8KyVxq%2FJsZdaVN8O2AnCzZBb5DnF7SguS2ElEPlAwCS0n%2B2W8vJRUdSi1WdlKo1lz9jZwF8y%2Fk%2BMevWsQAMjWb4Mt5yypKyuSS%2F5NAMSMc5MHGATNczmF7WdHVcV9WYFOIR6SCXuHVGnunhnRZOami6675XVhx3umZcijQeuZuvEgV1TqqaYT9D1fWJjzcMRe4QPW%2BlzyF1gNPIfwg56w1X0Ff8kh2sQ0Xca1j8XTuEK3h4cG1aXlpa%2B3rEdokIv3X61TAZiXhogzE4TQUZmIp5NdVmUd5666J0FwtsE5npqiJfxZgJH9ng2%2BdKDwtmKHUPHHHKaaxMMNZGgyzuf4INA%2Bjvmyz5YK1naQpnVv7x4%2F9N3c5JRGvvsKx43Qs%2FBE7X%2F6eBAX3YOE4BT87nqd1%2F0UUjOM5piqLyo%2FkFIUyKOq5379BKjauczbrWK9bGm95nh8MopRX63o3zpLms8CEOUujwYTGtyAPUHcc4%2Fn7xKPkjyvwmDrrVape5Y1ut%2BtQa490gpCim5jMz3Gvl3yEHBgIAkWkvO57UdUXooS7EIgzOsvGhSOEoxElZsXbm%2Bc3WB%2FPxadCqi0DTwT%2F7qjZU%2FaRUi46bfPN8w7eOYjAY6pAHiVsIWea5nOvbTpLs99Ym7edSQfFGBFOhkUEQvWJXpHaymqKPkRFwRSlDk5pYSwApgqXi4p0mTFIH2QSwRRk0Mtkxt%2BAATxmf%2Bni%2Fi92iLqjysnKK%2FkDgnLpTmyl67J5uX%2F5gxPThSOPUvn9zW6nRkKFQgG64GmLdVO99dQBwauy0XT4I8MjtiZgeOoUkPFEISPVNXJwtEYkEpgw5TcIZ8zA0SvQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211106T102812Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIARSLZVEVEV6ALJXJW%2F20211106%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Signature=d8237e6a8140c4774008c5094ba0cf96e1e7924d37bece29e8ceb654c0216e78

+ https://www.sciencedirect.com/science/article/pii/S0006291X20301935#bib22

+ https://doi.org/10.1016/j.bbagen.2018.06.014

# 2021-11-7

## PLAN
+ **Penn State问题回答**
+ **图嵌入学习**

## Penn State 问题
+ Describe your most meaningful independent research experience. Your answer should include the scientific questions or hypotheses you asked, the experimental design you used to test the hypothesis or question, your results, and the conclusions you drew about the biological processes you studied. Describe your role in the project: Were you part of a research team? What parts of the project were you responsible for? 500 words maximum.

今年九月开始，我们实验室开展了一项新课题，关于G4、甲基化和转座子在拟南芥全基因组的研究。这个课题是第一次我独自设计的课题并实施的，目前依然在进行。一些研究报道了G4结构与LTR末端区域的研究，同时也有报道了G4在CpG island处富集，我猜测G4结构和甲基化有关。同时G4潜在序列是否折叠除了受序列因素影响还会受理化性质的影响，对于不同类型的拟南芥在不同的生态环境类型，相同序列是否折叠会出现不同情况，我们基于这种假设对于Genome1001项目中世界各地的拟南芥基因组，进行了全基因组分析。考虑到TAIR10项目中已经有了1135种拟南芥在世界各地的坐标，甲基化注释图谱，以及全基因组序列。我们首先利用软件Quadron (一种基于机器学习预测pqs的软件)识别全部G4结构。之后通过统计，识别出的潜在G4结构周边的甲基化程度，推断是否该潜在G4进行了折叠。综合多个拟南芥类群的信息，我们可以推断哪些类别潜在G4结构是否折叠受地理因素影响较大。同时还利用LTR识别软件，将不同拟南芥类群的LTR识别出来，并判断不同类别的拟南芥存在的LTR差异。如果差异LTR的末端重复序列G4结构附近的甲基化程度很高，这意味着不同地理位置的拟南芥基因组种转座子差异主要是由地理位置对于G4结构折叠影响，从而导致逆转录转座子LTR在基因组中的转座受到影响，导致拟南芥物种分化。

目前我们的发现不同地区分布的生态型，在G4区域附近的甲基化程度不同。同时，我们还提出假设，不同地区的拟南芥类群在转座子层面的差异，主要的原因就是，G4结构受地理因素影响，导致其抑制甲基化程度降低，使得转座子无法转座，这一假设还需要继续结合地理因素进行探究。

考虑到我此前已经有了一定的科研训练基础，我在这个项目中作为负责人带领整个团队，从选题到基因组数据获取，和最终的项目实施以及最终完成paper都将由我主要负责。在我们的团队中还有一些目前大学三年级的同学，我还与这些同学进行合作，共同完成这个课题。

+ Generally, it's considered best practice for a student to move to a new institution for Graduate School. If you are currently a Penn State, University Park campus student, please tell us why staying here is a good choice for you.

+ Briefly describe an academic obstacle you faced or a challenge you have met. What strategies did you use to overcome the challenge? If you faced a future technical problem in your research, how would you overcome this hurdle. 250 words maximum

+ What are your current research interests, why is that field/area attractive to you, and which faculty in the department would allow you to pursue these interests? What do you see yourself doing in 10 years from now? 250 words maximum

目前我对基因组学很感兴趣，包括人类进化与疾病的关系很感兴趣，同时技术层面上我对于机器学习深度学习结合生物模型很感兴趣。此前，我的关于Genome1001拟南芥群体基因组的项目与群体遗传学相关，我在本科的学习大量的生物学、统计学以及编程知识。因此，我将宾州州立大学生物博士项目视为一个理想的平台，以实现我成为群体遗传学领域领先的研究人员的梦想。此外，目前随着基因组技术的发展，通过群体遗传学手段帮助人类治疗治病越来越成为可能，我在一方面进一步的研究不仅可以为人类疾病治疗做贡献，同时在精准医疗行业发展越来越快的情况下，对自己的职业生涯也十分有益。

Dr. Yifei Huang 的研究我十分感兴趣，不管是研究的科学问题(从群体遗传学角度研究自然选择对于人类进化和疾病的影响)，还是所采用的方法(用计算手段与机器学习还有深度学习结合)都十分吸引我，同时受他研究的启发，我将他研究的一些方法用到了我研究拟南芥群体基因组项目中。

在十年后，我想我可能在高校寻求教职，做科学研究，也有可能在医疗公司或者医院做研究人员，对于人类群体疾病进行研究。在这十年中，不仅我希望自己能够发表高水平论文，也希望自己的学术成果能真正帮助到病人，解决一些实际问题。同时也希望能开发方法或者软件，帮助到整个研究领域。

+ Socially diverse groups do better science, are more productive and innovative, and make better decisions. Describe any past activities that have supported diversity and inclusion. How would you support or contribute to diversity and inclusion in the Biology program? 250 words maximum

+ Describe something in your academic life that you are most proud of. 250 words maximum.

## Graph Embedding

作用 ： 如果没有Graph Embedding 要使用one hot编码 但是可能长度过于长，同时失去了节点之间的信息

### Deep work
https://zhuanlan.zhihu.com/p/45167021
参数为 参数矩阵 每一个节点映射的向量
利用 似然函数 做梯度下降
似然函数为 在出现某一结点$v_i$的条件下，出现某一个序列的概率 其中概率部分的意思是，在一个随机游走中，当给定一个顶点 $v_i$时，出现它的w窗口范围内顶点的概率。
训练样本构成 : 通过随机游走，建立大量的路径

# 2021-11-7

## PLAN
+ **图嵌入其他算法、代码**
+ **diversity statement完成**

## Diversity Statement

+ Socially diverse groups do better science, are more productive and innovative, and make better decisions. Describe any past activities that have supported diversity and inclusion. How would you support or contribute to diversity and inclusion in the Biology program? 250 words maximum

It is well known that the United States is a diverse society, and Penn State argues that socially diverse groups do better science, are more productive and innovative, and make better decisions.
In my opinion, I accept this tradition and I think diversity is the key to our evolution because it brings new tastes to our existing communities.
Living in a pluralistic environment, where different cultures and environments bring with them different personal experiences, values and worldviews, helps people interact effectively with each other and prepares them to participate in an increasingly complex and pluralistic society.
Take me for example.
When I was in late primary school, my mother left home to study for a PhD, so I learned to take care of myself and manage my time.
During this time, I learned how to cook and clean my room.
Although the experience was a bit difficult, it improved my self-reliance.
In addition, this special experience also reminds me to have confidence to overcome the difficulties in university study and research work.
After entering Sichuan University, I participated in many volunteer and community activities, such as serving as the volunteer of Chengdu station of Chinese Society of Cell Biology Conference, caring for the elderly in nursing homes, and accompanying the elderly during festivals.
In addition, when I entered the senior year of university, I also began to guide junior students to do projects.
These experiences are my valuable asset.
On the one hand, these activities give me a sense of achievement.
On the other hand, I enjoyed brainstorming and discussing academic issues with my younger brothers and sisters in the process of instructing them, although they might lack knowledge.
These gave me the idea of applying for a PhD, hoping to make more contributions to the society.

# 2021-11-8

## PLAN
+ **填写 Penn State Madison UArizona网申**
+ **networkx overview**