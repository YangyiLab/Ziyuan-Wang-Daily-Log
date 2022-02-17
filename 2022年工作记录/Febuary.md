---
output : pdf_document
---
- [2022-2-1](#2022-2-1)
  - [PLAN](#plan)
  - [文献地址](#%E6%96%87%E7%8C%AE%E5%9C%B0%E5%9D%80)
- [2022-2-2](#2022-2-2)
  - [PLAN](#plan-1)
- [2022-2-5](#2022-2-5)
  - [PLAN](#plan-2)
  - [微生物文章](#%E5%BE%AE%E7%94%9F%E7%89%A9%E6%96%87%E7%AB%A0)
  - [Field Vector / Single-Cell](#field-vector--single-cell)
    - [Framework](#framework)
    - [Kinetic parametres estimation](#kinetic-parametres-estimation)
- [2022-2-6](#2022-2-6)
  - [PLAN](#plan-3)
  - [文章修改todo list](#%E6%96%87%E7%AB%A0%E4%BF%AE%E6%94%B9todo-list)
  - [Arizona 面试](#arizona-%E9%9D%A2%E8%AF%95)
    - [Introduction](#introduction)
    - [Bio Background](#bio-background)
    - [CS Background](#cs-background)
    - [*in-silico* perturbation](#in-silico-perturbation)
  - [Dynamo](#dynamo)
    - [limitation 改进](#limitation-%E6%94%B9%E8%BF%9B)
    - [相关工作](#%E7%9B%B8%E5%85%B3%E5%B7%A5%E4%BD%9C)
- [2022-2-7](#2022-2-7)
  - [PLAN](#plan-4)
  - [全基因组项目资料](#%E5%85%A8%E5%9F%BA%E5%9B%A0%E7%BB%84%E9%A1%B9%E7%9B%AE%E8%B5%84%E6%96%99)
    - [群体遗传软件](#%E7%BE%A4%E4%BD%93%E9%81%97%E4%BC%A0%E8%BD%AF%E4%BB%B6)
  - [dynamo insilico](#dynamo-insilico)
  - [微生物摘要](#%E5%BE%AE%E7%94%9F%E7%89%A9%E6%91%98%E8%A6%81)
- [2022-2-8](#2022-2-8)
  - [PLAN](#plan-5)
  - [统计学学习](#%E7%BB%9F%E8%AE%A1%E5%AD%A6%E5%AD%A6%E4%B9%A0)
    - [最小二乘法](#%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95)
    - [残差图](#%E6%AE%8B%E5%B7%AE%E5%9B%BE)
- [2022-2-9](#2022-2-9)
  - [PLAN](#plan-6)
  - [拟南芥文章](#%E6%8B%9F%E5%8D%97%E8%8A%A5%E6%96%87%E7%AB%A0)
  - [统计学学习](#%E7%BB%9F%E8%AE%A1%E5%AD%A6%E5%AD%A6%E4%B9%A0-1)
    - [多元线性回归的误差分布](#%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%9A%84%E8%AF%AF%E5%B7%AE%E5%88%86%E5%B8%83)
    - [多元线性回归的统计修正](#%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%9A%84%E7%BB%9F%E8%AE%A1%E4%BF%AE%E6%AD%A3)
- [2022-2-10](#2022-2-10)
  - [PLAN](#plan-7)
  - [会议记录](#%E4%BC%9A%E8%AE%AE%E8%AE%B0%E5%BD%95)
  - [统计学学习](#%E7%BB%9F%E8%AE%A1%E5%AD%A6%E5%AD%A6%E4%B9%A0-2)
  - [拟南芥文献](#%E6%8B%9F%E5%8D%97%E8%8A%A5%E6%96%87%E7%8C%AE)
    - [Key Points](#key-points)
    - [Refs](#refs)
- [2022-2-11](#2022-2-11)
  - [PLAN](#plan-8)
  - [PPT](#ppt)
  - [生存分析](#%E7%94%9F%E5%AD%98%E5%88%86%E6%9E%90)
    - [生存曲线](#%E7%94%9F%E5%AD%98%E6%9B%B2%E7%BA%BF)
    - [概念](#%E6%A6%82%E5%BF%B5)
    - [COX 回归模型](#cox-%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B)
- [2022-2-12](#2022-2-12)
  - [PLAN](#plan-9)
- [2022-2-13](#2022-2-13)
  - [PLAN](#plan-10)
  - [PPT](#ppt-1)
    - [题目](#%E9%A2%98%E7%9B%AE)
  - [拟南芥旁系种](#%E6%8B%9F%E5%8D%97%E8%8A%A5%E6%97%81%E7%B3%BB%E7%A7%8D)
- [2022-2-15](#2022-2-15)
  - [PLAN](#plan-11)
# 2022-2-1

## PLAN

+ **ntu报告代码完成**
+ **文献下载**

## 文献地址

https://sci-hub.mksa.top/10.1126/science.1070919#

# 2022-2-2

## PLAN

+ **NTU完成**
+ **文献阅读**

# 2022-2-5

## PLAN
+ **微生物文章修改**
+ **Qiu 单细胞文章阅读**

## 微生物文章

目前存在无法修改内容

+ 没法确定结果中是否修改逻辑
+ Introduction 叙述what we have done 部分

## Field Vector / Single-Cell

### Framework

输入 细胞状态 $\mathbf{x}$ 各个基因的表达，$\mathbf{f}$ *field vector 瞬时速度* 的到每一时刻状态变化。 

field vector由每一个基因表达量所决定，并建立动力学方程

$$\frac{\partial f_1}{\partial x_1}=\alpha_1 \frac{n*K_1^nx_1^{n-1}}{(K_1^n+x_1^n)^2}-1$$

$$\frac{\partial f_1}{\partial x_2}=b_2 \frac{n*K_2^nx_1^{n-1}}{(K_2^n+x_1^n)^2}-1$$

### Kinetic parametres estimation

**data**

> cscRNA-seq 传统的通过推断 unslinced/sliced

> tscRNA-seq 时间标记

推断参数 csc推断相对参数$\gamma'=\frac{\gamma}{\beta}$,tsc可以推断绝对参数$\alpha,\beta,\gamma$

+ $\alpha$ 转录速率
+ $\beta$ splicing 速率
+ $\gamma$ degration 速率

# 2022-2-6

## PLAN
+ **文章修改**
+ **dynamo**
+ **面试修改**

## 文章修改todo list

+ Introduction 内容
+ 简写梳理/希望用一张表
+ 摘要需修改

## Arizona 面试

### Introduction

My name is Ziyuan Wang, a fourth-year undergraduate from Sichuan University, majoring in computational biology.I have been taking plenty of courses related to biology, statistics, and programming during my three-year undergraduate study. I am from Hebei Province, China. Hebei is near Beijing, far more north compared with Tucson. I went to Sichuan University in Chengdu, which is famous for hotpot and panda. Plus, it rains a lot the whole year.

### Bio Background

During my undergraduate study, I participated in the computational biology program jointly cultivated by The College of Software and The College of Life Science of Sichuan University, so I have a good background in biology. To be honest, my knowledge of biology and medicine may not be as competitive as the other candidates. However, I studied cell biology, molecular biology and biochemistry in undergraduate study, so I have a basic understanding of modern order biology. With this background, I think I can survive in this project, although I am still not familiar with drug development.

At the same time, I've been hearing about a group of ambitious scientists at the University of Arizona College of Pharmacy, who want to make important breakthroughs in science, like Harvard and MIT. At present, deep learning and other technologies are more and more widely used in pharmacy. I believe that my participation will definitely make new breakthroughs in scientific research of your school.

### CS Background

That’s a good question since I have been taking a John program from college of software engineering and school of life sciences train university. I definitely have a lot of experience of programming. 

When we talk about programming we would like to talk about the programming language. In the field of bioinformatics and biostatistics, R and Python are really pop.

Plus, C and Java, 2 of the most widely used languege are also okay for me. I can use them to do algorithm and do tieh optimization of my framework or construct web and visualization.

In addition, I have solid theoretical knowledge of computer, including data structure and algorithm, operating system, discrete mathematics and other courses, which I have learned during my undergraduate study. These have helped me a lot in my scientific research.

### *in-silico* perturbation

Dr. Ding and I have worked together in depth for a period of time to jointly develop a deep learning model that can predict gene profile expression through transcriptome data. For drug development, many drugs can be targeted at specific genes, and screening for specific drugs requires extensive animal and cell testing. So we developed deep learning models to infer single cell cellular state initiates in response to chemical perturbations to simulate this process. As I've mentioned, this is an example of how emerging technologies are evolving in medicine.

The data of the expression of Transcription factors will be input into the latent space and then the whole gene profile will be predicted.

Generative is like we can do the imagination even if we have not seen the whole event have ever happened. We have seen the dogs and cats and players playing soccer, however we never saw dogs and cats playing soccer. In the dream, we can. That is what generative model can do. Even though, we may not now seen the outcome of a specific Transcription factors expression profile, we can predict that outcome.

## Dynamo

易懂版教程 https://dynamo-release.readthedocs.io/en/latest/notebooks/perturbation_introduction_theory.html#perturbation-theory-tutorial

### limitation 改进

![fig 3](https://ars.els-cdn.com/content/image/1-s2.0-S0092867421015774-gr3.jpg)

+ Fig3 B 左工具scVelo(https://www.nature.com/articles/s41587-020-0591-3) 右工具 dynamo


此前工作问题为利用splice/unspliced RNA预测的假设前提下，splice/unspliced RNA含量测定不准确 如果基于cscRNA-seq

### 相关工作

**单细胞轨迹**

+ https://www.nature.com/articles/s41587-019-0071-9 *A comparison of single-cell trajectory inference methods*

**RNA 速度**

+ https://www.nature.com/articles/s41586-018-0414-6 *RNA velocity of single cells*
+ https://www.nature.com/articles/s41587-020-0591-3 *Generalizing RNA velocity to transient cell states through dynamical modeling*

**代谢标记** 通过代谢标记来量化新生的 RNA (**tscRNA-seq**)

+ https://www.nature.com/articles/s41592-020-0935-4 *Generalizing RNA velocity to transient cell states through dynamical modeling*
+ https://www.science.org/doi/10.1126/science.aax3072 *Sequencing metabolically labeled transcripts in single cells reveals mRNA turnover strategies*
+ https://www.nature.com/articles/s41586-019-1369-y *scSLAM-seq reveals core features of transcription dynamics in single cells*

# 2022-2-7

## PLAN
+ **全基因组项目资料查询**
+ **面试准备**
+ **dynamo 收尾**
+ **微生物文章表格设计**

## 全基因组项目资料

### 群体遗传软件

DnaSP V6

https://academic.oup.com/mbe/article/34/12/3299/4161815

Genetic Differentiation Analysis $K_{st},F_{st},\pi$

## dynamo insilico

$\Delta x$ pertrubation 需要到PCA维度转化 再回到 gene-wise

## 微生物摘要

hitchhiking 是什么 生物和环境(植物和氮肥)影响尚不清楚。所以我们怎么设计的实验，为了阐明植物是否影响hitchhiking，加一层膜阻挡hitchhiking 解释这种影响(不用说氮肥浓度). 结果显示...

# 2022-2-8

## PLAN
+ **统计学学习/回归**
+ **面试准备**
+ **微生物文章修改**

## 统计学学习

### 最小二乘法

利用求导和极大似然法求解$\alpha,\beta$

### 残差图

残差应该满足正态分布，通过残差图判断使用的拟合模型是否有问题。


# 2022-2-9

## PLAN
+ **统计学学习/回归**
+ **面试准备**
+ **拟南芥项目文献阅读**
+ **摘要定稿**

## 拟南芥文章

+ Puig Lombardi, E., Holmes, A., Verga, D., Teulade-Fichou, M.-P., Nicolas, A., & Londoño-Vallejo, A. (2019). Thermodynamically stable and genetically unstable G-quadruplexes are depleted in genomes across species. Nucleic Acids Research, 47(12), 6098–6113. https://doi.org/10.1093/nar/gkz463
+ Griffin, B. D., & Bass, H. W. (2018). Review: Plant G-quadruplex (G4) motifs in DNA and RNA; abundant, intriguing sequences of unknown function. Plant Science, 269(January), 143–147. https://doi.org/10.1016/j.plantsci.2018.01.011

## 统计学学习

### 多元线性回归的误差分布

$$f(x_1,x_2,...,x_m)=\frac{1}{\sqrt{2\pi}^{n}}e^{-\frac{\sum_{i=1}^{n}x_i^2}{2}}$$

通过该公式可以通过极大似然法求解参数$\mathbf{\beta},\alpha$
$$E(\epsilon|\mathbf{x})=\begin{bmatrix} 0  \\  0 \\0 \\ ... \\ 0 \end{bmatrix}_{n\times1}$$

$$Var(\epsilon|\mathbf{x}) = \sigma^2I$$

回归系数的均值无偏估计为$\beta$,方差无偏估计为$\sigma^2 /(\mathbf{XX^T})$

### 多元线性回归的统计修正

+ R 修正 $\bar{R}^2 = 1-(1-R^2)\frac{n-1}{n-k-1}$ 防止出现自变量多的拟合越来越优的悖论
+ 多重共线性的解决方法 利用方差膨胀因子VIF ${VIF}_i=\frac{1}{1-R_i^2}$ $R_i$为将$x_i$ 回归到$\mathbf{x}^{-i}$时计算的R值 VIF<10，自变量可以被接受

# 2022-2-10

## PLAN
+ **会议**
+ **统计学学习**
+ **面试演练**
+ **文献阅读**

## 会议记录

software

dataset

remote VPN 才能连接

VPN address https://technology.pharmacy.arizona.edu/get-support/cisco-anyconnect-vpn



## 统计学学习

二值变量学习

Probit/Logistic

区别在于Probit给予的假设为误差服从正态分布，计算较复杂

## 拟南芥文献

### Key Points

+ duplex opening G4的状态可以被称为inreversible or conditional

### Refs

Tang, J., Wu, J., Zhu, R., Wang, Z., Zhao, C., Tang, P., Xie, W., Wang, D., & Liang, L. (2021). Reversible photo-regulation on the folding/unfolding of telomere G-quadruplexes with solid-state nanopores. Analyst, 146(2), 655–663. https://doi.org/10.1039/d0an01930e


# 2022-2-11

## PLAN
+ **面试演练**
+ **文献阅读**
+ **生存分析**
+ **摘要修改**

## PPT

+ Predicting gene expression alterations caused by chemical or genetic perturbations on TFs
+ reconstruction Loss 做一张ppt
+ Version 藏起来

## 生存分析

### 生存曲线

利用估计法 估计每个时刻的存活率


### 概念

$d_i$ 是否为$t_i$时刻事件发生 $w_i$ $t_i$时刻发生删失事件。

### COX 回归模型

比例风险模型 对应的多因素模型则常用Cox回归模型（Cox风险比例模型）

风险函数表示生存时间达到t后瞬时发生失效事件的概率，用$h(t)$表示，$h(t)=f(t)/S(t)$。其中$f(t)$为概率密度函数（Probability Density Function），$f(t)=F'(t)$的导数。$F(t)$为积累分布函数（Cumulative Distribution Function），$F(t)=1-S(t)$，表示生存时间未超过时间点t的概率。累积风险函数$H(t)=-logS(t)$。概率密度和积累分布的关系类似于速度与位移的关系。


Cox比例风险回归模型的基本形式为：将某时点t个体出现失效事件的风险分为两部分：$h_0(t)$和$h(t,X)$。$第i个影响因素X使风险函数从h0(t)$$增加exp（βiXi）$而成为$h_0(t)*exp(βiXi)$。

# 2022-2-12

## PLAN
+ **修改ppt**
+ **更新讲稿**

# 2022-2-13

## PLAN
+ **面试演练**
+ **拟南芥旁系种**
+ **润色查看**

## PPT

### 题目

**using TE landscape to infer geographic evolution of plant**

+ geographical distribution of 拟南芥  地理和进化的关系
+ 我们利用 TE G4 甲基化 关系弄出来 environmental conditions G4 folding 甲基化 
+ 流程图 空 和箭头 中间指示图

## 拟南芥旁系种

物种选择 C. rubella

# 2022-2-15

## PLAN
+ 面试演练

