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
+ 统计学学习/回归
+ 面试准备
+ 拟南芥项目文献阅读
+ 摘要定稿


