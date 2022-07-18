- [2022-7-4](#2022-7-4)
  - [PLAN](#plan)
  - [R4 回答问题](#r4-%E5%9B%9E%E7%AD%94%E9%97%AE%E9%A2%98)
  - [few-shot learning without forgetting](#few-shot-learning-without-forgetting)
    - [cosine similarity](#cosine-similarity)
    - [利用base classification weight vector 和$z'$训练novelty 分类器](#%E5%88%A9%E7%94%A8base-classification-weight-vector-%E5%92%8Cz%E8%AE%AD%E7%BB%83novelty-%E5%88%86%E7%B1%BB%E5%99%A8)
- [2022-7-5](#2022-7-5)
  - [PLAN](#plan-1)
  - [文献公式](#%E6%96%87%E7%8C%AE%E5%85%AC%E5%BC%8F)
    - [cosine的好处](#cosine%E7%9A%84%E5%A5%BD%E5%A4%84)
    - [remain问题](#remain%E9%97%AE%E9%A2%98)
  - [转录组normalization (TPM)](#%E8%BD%AC%E5%BD%95%E7%BB%84normalization-tpm)
- [2022-7-6](#2022-7-6)
  - [PLAN](#plan-2)
  - [COP服务器](#cop%E6%9C%8D%E5%8A%A1%E5%99%A8)
  - [文献阅读](#%E6%96%87%E7%8C%AE%E9%98%85%E8%AF%BB)
- [2022-7-7](#2022-7-7)
  - [PLAN](#plan-3)
  - [COP](#cop)
  - [Cover Letter](#cover-letter)
  - [单细胞模型](#%E5%8D%95%E7%BB%86%E8%83%9E%E6%A8%A1%E5%9E%8B)
- [2022-7-8](#2022-7-8)
  - [PLAN](#plan-4)
  - [COP Rserver](#cop-rserver)
  - [Cover Letter](#cover-letter-1)
  - [GCN](#gcn)
  - [Bonito](#bonito)
- [2022-7-9](#2022-7-9)
  - [PLAN](#plan-5)
  - [Cover Letter](#cover-letter-2)
  - [GCN代码](#gcn%E4%BB%A3%E7%A0%81)
  - [JCppt](#jcppt)
    - [引言](#%E5%BC%95%E8%A8%80)
    - [Related Concept and Work](#related-concept-and-work)
- [2022-7-10](#2022-7-10)
  - [PLAN](#plan-6)
  - [GCN 代码](#gcn-%E4%BB%A3%E7%A0%81)
  - [JC](#jc)
    - [Learning to learn by gradient descent by gradient descent](#learning-to-learn-by-gradient-descent-by-gradient-descent)
- [2022-7-11](#2022-7-11)
  - [PLAN](#plan-7)
  - [单细胞读文件](#%E5%8D%95%E7%BB%86%E8%83%9E%E8%AF%BB%E6%96%87%E4%BB%B6)
  - [Bonito](#bonito-1)
  - [UA 服务器](#ua-%E6%9C%8D%E5%8A%A1%E5%99%A8)
- [2022-7-12](#2022-7-12)
  - [PLAN](#plan-8)
  - [Bonito源码](#bonito%E6%BA%90%E7%A0%81)
  - [GCN](#gcn-1)
- [2022-7-13](#2022-7-13)
  - [PLAN](#plan-9)
  - [细胞间配体受体结合预测](#%E7%BB%86%E8%83%9E%E9%97%B4%E9%85%8D%E4%BD%93%E5%8F%97%E4%BD%93%E7%BB%93%E5%90%88%E9%A2%84%E6%B5%8B)
- [2022-7-17](#2022-7-17)
  - [PLAN](#plan-10)
  - [GCN权值](#gcn%E6%9D%83%E5%80%BC)
- [2022-7-18](#2022-7-18)
  - [PLAN](#plan-11)
  - [GCN代码测试](#gcn%E4%BB%A3%E7%A0%81%E6%B5%8B%E8%AF%95)
  - [Bonito](#bonito-2)
    - [Alphabetical](#alphabetical)
  - [contig 说明](#contig-%E8%AF%B4%E6%98%8E)


# 2022-7-4

## PLAN

+ **文献阅读**
+ **Cover Letter**
+ **单细胞数据集**

## R4 回答问题

主要对于研究范围进行了回答

## few-shot learning without forgetting

主题 小样本学习同时不忘记以前的任务

### cosine similarity

最后一层分类器本来为$s_k=z^\bold{T}w_k^*$

利用cosine similarity 可以做到$s_k=a\times \frac{z^\bold{T}}{||z^\bold{T}||}frac{w_k^*}{w_k^*}$ 即做一种normalization消除量纲

### 利用base classification weight vector 和$z'$训练novelty 分类器

$$G(Z',W_{base}|\psi)$$

novel classification 依赖于novel training的freature extraction 和 Base分类器的权重

# 2022-7-5

## PLAN

+ **单细胞结果可视化**
+ **Cover Letter 回复**
+ **文献公式整理**
+ **NGS 回顾及学习**

## 文献公式

### cosine的好处

分类器直接利用novel类和input $z$的相似度进行判别。

### remain问题

+ 为什么要加入fake novel
+ 如何梯度下降 (利用github)

## 转录组normalization (TPM)

常用软件 EdgeR, Deseq2

# 2022-7-6

## PLAN
+ **文献阅读**
+ **服务器配置**

## COP服务器

密码更改为wzy851234,.

## 文献阅读

主题 **基于atac-seq数据推测TRN**

深度学习模型 VAGE

训练方法对于ground-true数据，mask一些TRN edge 通过可见的进行预测不可见的TRN edge

# 2022-7-7

## PLAN

+ **COP服务器探索**
+ **Cover Letter**
+ **单细胞模型检查**

## COP

目前没有sudo权限

## Cover Letter

R4 基本回复完毕，R3需要对照新版本进行修订

## 单细胞模型

通过Z1/Z2 Z1/Z3的作图发现，模型出问题的关键在于多层后会出现一种收敛的情况，这种状态似乎也是与生物状态不符的。

# 2022-7-8

## PLAN

+ **COP 服务器配置**
+ **GCN代码下载**
+ **Bonito代码下载**
+ **Cover Letter第一版**

## COP Rserver

ip http://10.128.207.5:8787/auth-sign-in?appUri=%2F

已可以迁移in silico perturbation的代码

## Cover Letter

目前存在问题

+ Line 526-529 this sentence is very speculative. How do you know transported bacteria are "favourable" and they have plant growth promoting characteristics? 准确细菌类型需要表述
+ Line 402 This a strong argument. You only studied N concentration influence, but many other factors may be implied (you may cite some studies here). 问题没看懂

## GCN

配置了一个conda环境 GCN 导入了pyg库

## Bonito

代码已下载但还不能单步执行跑


# 2022-7-9

## PLAN

+ **Cover Letter**
+ **GCN代码探索**
+ **JC ppt**

## Cover Letter

+ 一个难以回答的问题
+ 方法的问题没有回答

## GCN代码

成功导入pyg库在conda GCN环境中，并成功运行

## JCppt

### 引言

主要讨论这篇文章模型的目的

> **In this context, the goal of our work is to devise a few-shot visual learning system that during test time it will be able to efficiently learn novel categories from only a few training data while at the same time it will not forget the initial categories on which it was trained**

### Related Concept and Work

Concept: Meta-learning/ Incremental Learning

Work: Meta-Learning/ Metric-Learning(包括两个方向，正则化和matching方法)


# 2022-7-10

## PLAN
+ **GCN代码探索**
+ **JC ppt**

## GCN 代码

可以在`GCNconv()`函数里增加edge_attr 来获取边的值，这也就是tf_net中对应的权重。

## JC

### Learning to learn by gradient descent by gradient descent

```python
def learn(optimizee,unroll_train_steps,retain_graph_flag=False,reset_theta = False): 
    """retain_graph_flag=False   默认每次loss_backward后 释放动态图
    #  reset_theta = False     默认每次学习前 不随机初始化参数"""
    
    if reset_theta == True:
        theta_new = torch.empty(DIM)
        torch.nn.init.uniform_(theta_new,a=-1,b=1.0) 
        theta_init_new = torch.tensor(theta,dtype=torch.float32,requires_grad=True)
        x = theta_init_new
    else:
        x = theta_init
        
    global_loss_graph = 0 #这个是为LSTM优化器求所有loss相加产生计算图准备的
    state = None
    x.requires_grad = True
    if optimizee.__name__ !='Adam':
        losses = []
        for i in range(unroll_train_steps):
            
            loss = f(x)
            
            #global_loss_graph += torch.exp(torch.Tensor([-i/20]))*loss
            #global_loss_graph += (0.8*torch.log10(torch.Tensor([i+1]))+1)*loss
            global_loss_graph += loss
            
            
            loss.backward(retain_graph=retain_graph_flag) # 默认为False,当优化LSTM设置为True
            update, state = optimizee(x.grad, state)
            losses.append(loss)
           
            x = x + update
            
            # x = x.detach_()
            #这个操作 直接把x中包含的图给释放了，
            #那传递给下次训练的x从子节点变成了叶节点，那么梯度就不能沿着这个路回传了，        
            #之前写这一步是因为这个子节点在下一次迭代不可以求导，那么应该用x.retain_grad()这个操作，
            #然后不需要每次新的的开始给x.requires_grad = True
            
            x.retain_grad()
            #print(x.retain_grad())
            
            
        #print(x)
        return losses ,global_loss_graph 
    
    else:
        losses = []
        x.requires_grad = True
        optimizee= torch.optim.Adam( [x],lr=0.1 )
        
        for i in range(unroll_train_steps):
            
            optimizee.zero_grad()
            loss = f(x)
            global_loss_graph += loss
            
            loss.backward(retain_graph=retain_graph_flag)
            optimizee.step()
            losses.append(loss.detach_())
        #print(x)
        return losses,global_loss_graph 
    
Global_Train_Steps = 1000

def global_training(optimizee):
    global_loss_list = []    
    adam_global_optimizer = torch.optim.Adam([{'params':optimizee.parameters()},{'params':Linear.parameters()}],lr = 0.0001)
    _,global_loss_1 = learn(LSTM_Optimizee,TRAINING_STEPS,retain_graph_flag =True ,reset_theta = True)
    print(global_loss_1)
    for i in range(Global_Train_Steps):    
        _,global_loss = learn(LSTM_Optimizee,TRAINING_STEPS,retain_graph_flag =True ,reset_theta = False)       
        adam_global_optimizer.zero_grad()
        
        #print(i,global_loss)
        global_loss.backward() #每次都是优化这个固定的图，不可以释放动态图的缓存
        #print('xxx',[(z,z.requires_grad) for z in optimizee.parameters()  ])
        adam_global_optimizer.step()
        #print('xxx',[(z.grad,z.requires_grad) for z in optimizee.parameters()  ])
        global_loss_list.append(global_loss.detach_())
        
    print(global_loss)
    return global_loss_list

# 要把图放进函数体内，直接赋值的话图会丢失
# 优化optimizee
global_loss_list = global_training(lstm)


```

LSTM optimizer
```python
Layers = 2
Hidden_nums = 20
Input_DIM = DIM
Output_DIM = DIM
# "coordinate-wise" RNN 
lstm=torch.nn.LSTM(Input_DIM,Hidden_nums ,Layers)
Linear = torch.nn.Linear(Hidden_nums,Output_DIM)
batchsize = 1

print(lstm)
    
def LSTM_Optimizee(gradients, state):
    #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
    #原gradient.size()=torch.size[5] ->[1,1,5]
    gradients = gradients.unsqueeze(0).unsqueeze(0)   
    if state is None:
        state = (torch.zeros(Layers,batchsize,Hidden_nums),
                 torch.zeros(Layers,batchsize,Hidden_nums))
   
    update, state = lstm(gradients, state) # 用optimizee_lstm代替 lstm
    update = Linear(update)
    # Squeeze to make it a single batch again.[1,1,5]->[5]
    return update.squeeze().squeeze(), state
```

**LSTM优化器的最终优化策略是没有任何人工设计的经验在里面，是自动学习出的一种学习策略**


# 2022-7-11

## PLAN

+ **JC 讲稿**
+ **Nanopore sequencing 摸索**
+ **单细胞数据初探(NGS)**

## 单细胞读文件

数据来源 GSE225978

tpm 已被normalization后的数据
annotation 每个样本的描述(type, name...)

```R
annotation = read.table("GSE115978_cell.annotations.csv.gz",header = TRUE, sep = ",")[,3]
tpm = read.table("GSE115978_tpm.csv.gz",sep = ",",header = TRUE)
rownames(tpm) = tpm$X
tpm$X = NULL
tpm = t(tpm)
genes = colnames(tpm)
```

## Bonito

Input: fast5
Output: fastq,fasta...

+ 其中包含多个model
+ 需要找到合适的fast5文件 fast5文件本质就是数字

## UA 服务器

UA HPC Slurm手册 https://docs.slurm.cn/users/

# 2022-7-12

## PLAN

+ **GCN数据导入**
+ **Cover Letter定稿**
+ **Bonito源码**

## Bonito源码

修改部分包括
+ /home/princezwang/software/bonito/bonito/training.py
+ /home/princezwang/software/bonito/bonito/cli/train.py

## GCN
+ 使用新的训练权值可以导入
+ **GCNConv 出现nan值**

# 2022-7-13

## PLAN

+ **文献阅读**

## 细胞间配体受体结合预测

+ 通过gene profile推断不同细胞间的配体受体结合因素

+ 创建4D 配体受体打分表

+ 进行TCA降维可解释性

# 2022-7-17

## PLAN

+ **GCN权值**

## GCN权值

将每个权值设置为正数,需要挑选权值进行加减

**最大问题，处理负权重**

# 2022-7-18

## PLAN

+ **GCN代码测试**
+ **Bonito 代码查看更新修改**
+ **contig 文档**
+ **甲基化数据处理**

## GCN代码测试

normalize 去除后可以进行传递

## Bonito

```bash
bonito basecaller dna_r10.4_e8.1_sup@v3.4  /home/hongxuding/sunlab --device cpu> /home/princezwang/basecalls.bam
```

```python
def ctc_label_smoothing_loss(self, log_probs, targets, lengths, weights=None):
        T, N, C = log_probs.shape
        weights = weights or torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)])
        log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
        loss = ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean')
        label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
        return {'total_loss': loss + label_smoothing_loss, 'loss': loss, 'label_smooth_loss': label_smoothing_loss}
```

需要一个正则化loss

### Alphabetical

/home/princezwang/software/bonito/bonito/models/configs

```tmol
[labels]
labels = ["N", "A", "C", "G", "T"]
motifications = ["N", "A", "C", "G", "T"]
```

## contig 说明

包括三种物种: 拟南芥 大肠杆菌 桃

目前只保留原始数据: 参考基因组+测序数据

**已有代码 在scripts中**

+ WGS
  + SV (manta)
  + Population Genetics (vcftools/bcftools)
+ RNA-seq (hista/STAR)
+ sc-seq 
+ *bs-seq* (bimarsk)