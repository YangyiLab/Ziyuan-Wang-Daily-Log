- [2022-4-1](#2022-4-1)
  - [PLAN](#plan)
  - [MR analysis](#mr-analysis)
    - [1-sample MR](#1-sample-mr)
    - [2-sample MR](#2-sample-mr)
- [2022-4-2](#2022-4-2)
  - [PLAN](#plan-1)
  - [类别分类](#%E7%B1%BB%E5%88%AB%E5%88%86%E7%B1%BB)
- [2022-4-3](#2022-4-3)
  - [PLAN](#plan-2)
- [2022-4-4](#2022-4-4)
  - [PLAN](#plan-3)
  - [perturbation 可视化](#perturbation-%E5%8F%AF%E8%A7%86%E5%8C%96)
  - [转座子进化树问题](#%E8%BD%AC%E5%BA%A7%E5%AD%90%E8%BF%9B%E5%8C%96%E6%A0%91%E9%97%AE%E9%A2%98)
- [2022-4-5](#2022-4-5)
  - [PLAN](#plan-4)
  - [单细胞单个TF](#%E5%8D%95%E7%BB%86%E8%83%9E%E5%8D%95%E4%B8%AAtf)
- [2022-4-6](#2022-4-6)
  - [PLAN](#plan-5)
  - [hsc](#hsc)
    - [分化](#%E5%88%86%E5%8C%96)
    - [造血干细胞亚群](#%E9%80%A0%E8%A1%80%E5%B9%B2%E7%BB%86%E8%83%9E%E4%BA%9A%E7%BE%A4)
  - [拟南芥TE的G4含量](#%E6%8B%9F%E5%8D%97%E8%8A%A5te%E7%9A%84g4%E5%90%AB%E9%87%8F)
- [2022-4-7](#2022-4-7)
  - [PLAN](#plan-6)
  - [单细胞路径推断](#%E5%8D%95%E7%BB%86%E8%83%9E%E8%B7%AF%E5%BE%84%E6%8E%A8%E6%96%AD)
  - [拟南芥项目思路整理](#%E6%8B%9F%E5%8D%97%E8%8A%A5%E9%A1%B9%E7%9B%AE%E6%80%9D%E8%B7%AF%E6%95%B4%E7%90%86)
- [2022-4-8](#2022-4-8)
  - [PLAN](#plan-7)
  - [ds160](#ds160)
  - [topology (total)](#topology-total)
    - [0.9 时](#09-%E6%97%B6)
  - [分类别统计G4富集](#%E5%88%86%E7%B1%BB%E5%88%AB%E7%BB%9F%E8%AE%A1g4%E5%AF%8C%E9%9B%86)
  - [进行G4_TE计算](#%E8%BF%9B%E8%A1%8Cg4te%E8%AE%A1%E7%AE%97)
- [2022-4-9](#2022-4-9)
  - [PLAN](#plan-8)
  - [QTL 调控元件](#qtl-%E8%B0%83%E6%8E%A7%E5%85%83%E4%BB%B6)
  - [meta learning和度量学习](#meta-learning%E5%92%8C%E5%BA%A6%E9%87%8F%E5%AD%A6%E4%B9%A0)
  - [图神经网络](#%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
  - [GRN](#grn)
    - [TF2TF 调控网络](#tf2tf-%E8%B0%83%E6%8E%A7%E7%BD%91%E7%BB%9C)
    - [TF2GENES 调控网络](#tf2genes-%E8%B0%83%E6%8E%A7%E7%BD%91%E7%BB%9C)
- [2022-4-11](#2022-4-11)
  - [PLAN](#plan-9)
  - [GRN 影响范围](#grn-%E5%BD%B1%E5%93%8D%E8%8C%83%E5%9B%B4)
    - [Gata1](#gata1)
    - [Spi1](#spi1)
  - [年龄计算](#%E5%B9%B4%E9%BE%84%E8%AE%A1%E7%AE%97)
- [2022-4-12](#2022-4-12)
  - [PLAN](#plan-10)
  - [G4 prediction](#g4-prediction)
  - [TE转座子](#te%E8%BD%AC%E5%BA%A7%E5%AD%90)
    - [graph](#graph)
    - [完成ltr计算代码](#%E5%AE%8C%E6%88%90ltr%E8%AE%A1%E7%AE%97%E4%BB%A3%E7%A0%81)
- [2022-4-13](#2022-4-13)
  - [PLAN](#plan-11)
  - [Visa](#visa)
  - [GRN排查](#grn%E6%8E%92%E6%9F%A5)
- [2022-4-14](#2022-4-14)
  - [PLAN](#plan-12)
  - [journal club](#journal-club)
    - [ouline](#ouline)

# 2022-4-1

## PLAN

+ **MR 学习**

## MR analysis

MR 通过gene作为介质，推断因果关系

+ trait X -> trait Y via gene G
+ X exposure 
+ Y outcome
+ G gene

![MR](https://pic2.zhimg.com/v2-d129171841cfc93d454451d11bf4fb01_1440w.jpg?source=172ae18b)

### 1-sample MR

+ exposure GWAS sig genes
+ outcome GWAS sig genes
+ harmonize
+ MR plot

### 2-sample MR

exposure and outcome using different dataset

# 2022-4-2

## PLAN

+ 写出含有G4的TE类别script

## 类别分类

```py
import os

##### path: the path that contains the number of the sample
##### chr_num: the number of chrom
##### output the ltr number that contains the G4

def annoate_G4_TE(path,chr_num):
    os.chdir(path)
    os.system(" bedtools coverage -a chr"+str(chr_num)+".ltr.bed -b chr"+str(chr_num)+".g4.bed -s -u > tmp.bed")
    tmp = open('tmp.bed')
    tmp_list = tmp.read().split('\n')
    tmp_list.pop()
    yes_list = [idx+1  for idx,i in enumerate(tmp_list) if float(i.split('\t')[-1])!=0 ]
    tmp.close()
    os.system("rm tmp.bed")
    return yes_list
def get_g4te_type(chr_num,path):
    g4_tes = annoate_G4_TE(path,chr_num)
    tes = open(path+"/chr"+str(chr_num)+".ltr.xml")
    tes_list = tes.read().split('\n')
    tes_list.pop()
    # print(tes_list)
    g4_tes_anno = [i for i in tes_list if int(i.split('\t')[1].split('-')[1]) in g4_tes]
    # int(tes_list[58].split('\t')[1].split('-')[1]) in g4_tes
    return g4_tes_anno
```

发现大量G4集中在Gypsy(LTR) 以及 DNA类型的转座子



# 2022-4-3

## PLAN

+ **可视化长时间训练模型**

# 2022-4-4

## PLAN
+ **perturbation 可视化**
+ **myglobal 调整**
+ **COP邮箱账户**
+ **整理G4**

## perturbation 可视化

+ 缺少GO的perturbation
+ 缺少HSC的metadata文件

## 转座子进化树问题

+ 多序列对比的gap过多/在family级别对比可能会效果好



# 2022-4-5

## PLAN
+ **单细胞GO**
+ **单细胞单个TF**


## 单细胞单个TF

测试了gata1,gata2的值，目前迁移结果有限，需要进行进一步测试，结合文献，一个TF作用有限。


# 2022-4-6

## PLAN
+ **拟南芥项目推进**
+ **PU1 perturbation**
+ **文献阅读**


## hsc

### 分化

由转录因子介导的分化即gata1和spi 1的拮抗作用，gata1倾向于mep而spi1倾向于gmp


### 造血干细胞亚群

不同类型细胞，转录因子表达量存在显著性不同


## 拟南芥TE的G4含量

有一定富集效应 有0.8分位数为0.5


# 2022-4-7

## PLAN

+ **修改模型**
+ **文献阅读**
+ **拟南芥项目思路整理**

## 单细胞路径推断

通过不同cluster细胞的关键转录因子进行推断

## 拟南芥项目思路整理

+ 找到G4富集的转座子
+ 计算富集度与活性的相关性
+ 不同群体活性研究，再结合G4

利用clustcal 建树 gap数明显减少

# 2022-4-8

## PLAN
+ **分类别统计G4富集**
+ **模型拓扑结构研究**
+ **进行G4_TE计算**


## ds160
AA00AT0D3D

## topology (total)

### 0.9 时

每一个gene平均受30-40tf调控

一个tf调控300-1000个genes

## 分类别统计G4富集

对于数量较大的TE

+ 有G4的TEs
+ 无G4的TEs
+ rebulla TEs

三个类群进行比较


## 进行G4_TE计算
**结论**
+ G4_TE 和基因组大小没有相关性，TE(total)/去除G4_TE后与基因组大小有相关性
+ TE_G4 和基因组大小没有相关性，G4(total)/去除TE_G4后与基因组大小有相关性

```r
quantile(data_join$TE_G4/data_join$G4_TE)
      0%      25%      50%      75%     100% 
6.445455 6.942857 7.164948 7.431579 8.228916 
```

# 2022-4-9

## PLAN

+ **文献阅读**
+ **meta learning**
+ **图算法**。
+ **tfnet tfgenenet分别分析**
+ **微生物文章方法修改**

## QTL 调控元件

regulatory QTL更好的反应功能

## meta learning和度量学习


我们回顾最基础的元学习模型采用嵌入向量与k近邻算法进行分类，后又可以使用神经网络分类
但k近邻不可导而神经网络需要微调

Attention机制本质是学习嵌入向量相似度，结合两者优点


## 图神经网络

频谱gcn空间域gate 挖掘度，邻接，k邻居之间关系得出每个结点嵌入向量

主流AXW 方法 结合注意力编码器等

## GRN

### TF2TF 调控网络

GATA1 -> Spi1 -1
Spi1 -> Gata1 0 (未复现)

### TF2GENES 调控网络

GATA1 -> Spi1 1 (未复现且冲突)
Spi1 -> Gata1 -1 

# 2022-4-11

## PLAN

+ **GRN网络作图pipeline**
+ **转座子年龄代码**

## GRN 影响范围

### Gata1

+ 0.5 2432
+ 0.7 1586
+ 0.9 456

### Spi1

+ 0.5 2295
+ 0.7 2037
+ 0.9 456

## 年龄计算

+ 每次抽两个
+ 序列比对
+ 计算K值
+ 估计年龄


# 2022-4-12

## PLAN
+ **整理tf GRN结果**
+ **整理拟南芥项目结果**
+ **研究G4 prediction**

## G4 prediction

**In vivo**，意思是在活体中的，实验在完整的活得生物体中进行。 

**In vitro**，意思是在玻璃中的，实验在生物体外，通常是实验玻璃器皿中进行。

+ *In vitro* G4 在$K^+,PDS$环境下有不同折叠

所有的预测都是基于G4-seq数据集，而G4-seq数据集是基于$K^+$进行处理

## TE转座子

### graph

图片路径 /home/ubuntu/data/Arabidopsis_sequence_2/midterm_result

+ 进化树
+ 相关性图

### 完成ltr计算代码

在每一个ltr_classification中

类群的文件下 *_age.txt

# 2022-4-13

## PLAN
+ **签证**
+ **甲基化可视化**
+ **进一步探索GRN**
+ **修改微生物文章**

## Visa
SEVIS N0032960589
school code PHO214F20092000
passport EA5831025

## GRN排查

+ gene/tf construction
+ perturb 之间的散点图

# 2022-4-14

## PLAN
+ **修改并提交微生物文章**
+ **perturbation 纠错可视化**
+ **journal club ppt**

## journal club

### ouline

+ Background Knowledge
  + 图神经网络和化学的应用
  + meta-learning 思想 主要从数据集谈(FSL)
+ Feature
+ Model Overview
+ Thoughts