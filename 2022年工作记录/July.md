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