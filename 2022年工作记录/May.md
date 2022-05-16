- [2022-5-4](#2022-5-4)
  - [PLAN](#plan)
  - [拟南芥年龄分布](#%E6%8B%9F%E5%8D%97%E8%8A%A5%E5%B9%B4%E9%BE%84%E5%88%86%E5%B8%83)
  - [de perturbation](#de-perturbation)
- [2022-5-5](#2022-5-5)
  - [PLAN](#plan-1)
  - [网络拓扑问题](#%E7%BD%91%E7%BB%9C%E6%8B%93%E6%89%91%E9%97%AE%E9%A2%98)
- [2022-5-6](#2022-5-6)
  - [PLAN](#plan-2)
- [2022-5-7](#2022-5-7)
  - [PLAN](#plan-3)
  - [预测数据分析](#%E9%A2%84%E6%B5%8B%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90)
  - [转座子活性](#%E8%BD%AC%E5%BA%A7%E5%AD%90%E6%B4%BB%E6%80%A7)
- [2022-5-8](#2022-5-8)
  - [PLAN](#plan-4)
  - [解决](#%E8%A7%A3%E5%86%B3)
  - [信息](#%E4%BF%A1%E6%81%AF)
- [2022-5-9](#2022-5-9)
  - [PLAN](#plan-5)
  - [空间转录组](#%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84)
    - [技术](#%E6%8A%80%E6%9C%AF)
- [2022-5-10](#2022-5-10)
  - [PLAN](#plan-6)
  - [1135文献](#1135%E6%96%87%E7%8C%AE)
    - [1135 测序](#1135-%E6%B5%8B%E5%BA%8F)
    - [RNA-seq/甲基化](#rna-seq%E7%94%B2%E5%9F%BA%E5%8C%96)
  - [蛋白质生成表示](#%E8%9B%8B%E7%99%BD%E8%B4%A8%E7%94%9F%E6%88%90%E8%A1%A8%E7%A4%BA)
- [2022-5-11](#2022-5-11)
  - [PLAN](#plan-7)
  - [tf2tf 层数修改](#tf2tf-%E5%B1%82%E6%95%B0%E4%BF%AE%E6%94%B9)
- [2022-5-14](#2022-5-14)
  - [PLAN](#plan-8)
  - [网络拓扑](#%E7%BD%91%E7%BB%9C%E6%8B%93%E6%89%91)
    - [此前情况](#%E6%AD%A4%E5%89%8D%E6%83%85%E5%86%B5)
    - [解决](#%E8%A7%A3%E5%86%B3-1)
- [2022-5-15](#2022-5-15)
  - [PLAN](#plan-9)
  - [单细胞训练训练步骤](#%E5%8D%95%E7%BB%86%E8%83%9E%E8%AE%AD%E7%BB%83%E8%AE%AD%E7%BB%83%E6%AD%A5%E9%AA%A4)
  - [minimap2](#minimap2)
- [2022-5-16](#2022-5-16)
  - [PLAN](#plan-10)
  - [拟南芥空间转录组](#%E6%8B%9F%E5%8D%97%E8%8A%A5%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84)
    - [验证](#%E9%AA%8C%E8%AF%81)
    - [应用](#%E5%BA%94%E7%94%A8)
  - [模型修改结果](#%E6%A8%A1%E5%9E%8B%E4%BF%AE%E6%94%B9%E7%BB%93%E6%9E%9C)
  - [NGS 学习](#ngs-%E5%AD%A6%E4%B9%A0)
    - [SRA 命令](#sra-%E5%91%BD%E4%BB%A4)
    - [bwa 命令](#bwa-%E5%91%BD%E4%BB%A4)

# 2022-5-4

## PLAN
+ **研究年龄分布**
+ **单细胞新模型**

## 拟南芥年龄分布

需要利用总长度做矫正,k 为每个核苷酸替换概率 distance/length

## de perturbation

出现了推过量的问题


# 2022-5-5

## PLAN

+ **网络拓扑检测**


## 网络拓扑问题

+ decoder训练没有加两个loss
+ mask rate调整

# 2022-5-6

## PLAN
+ **单细胞结果可视化**
+ **文献阅读**
+ **文件签字**
+ **relict 研究**


# 2022-5-7

## PLAN

+ **整理预测数据**
+ **文献阅读**

## 预测数据分析

v1 v2 push程度较大，但在gene-wise不明显，可以考虑修改umap降维参数

## 转座子活性

利用snp低频偏移以及转座子拷贝数做分析


# 2022-5-8

## PLAN

+ **单细胞原始数据分析**
+ **继续做不同mask rate调优**

## 解决

+ 堆叠层数
+ thredhold 分别不同

## 信息

Hongxu Ding, PhD

Assistant Professor, Translational Pharmacogenomics

Department of Pharmacy Practice and Science, University of Arizona

Drachman Hall B207N

Email: hongxuding@arizona.edu

Tel: 520-626-5764


Email: hongxuding@arizona.edu

Tel: 520-626-5764
Brian Erstad, PharmD

https://www.pharmacy.arizona.edu/

Roy P. Drachmann Hall
Pulido Center – Tucson
1295 N. Martin
PO Box 210202
Tucson, AZ 85721
Phone: 520-626-1427

$5500

19901130

丁鸿绪

SSN 722349266

# 2022-5-9

## PLAN

+ **空间转录组工作**
+ **修改perturbation 函数**

## 空间转录组

### 技术

+ DNB DNA纳米球上CID 进入到chip的每一个孔中，进行测序
+ 再将CID umi与切片结合
+ 二次进行sc-RNA测序时通过CID定位



# 2022-5-10

## PLAN
+ **蛋白生成模型**
+ **SRA fastaq**

## 1135文献

### 1135 测序

pipeline
+ 测fastq
+ QC
+ 生成VCF (没拼接)

同时利用allpath 做assemble了

### RNA-seq/甲基化

甲基化pipeline


+ 测序两次
  + trim
  + 去接头
  + botie

## 蛋白质生成表示

提出了一种变分自编码模型，但其特点为蛋白质已进行one-hot编码，因此其loss有较大不同，同时提出了不同的蛋白距离计算。另一个特点为可以模拟进化树结构

问题 模型十分复杂，不知道是否可以重参数化

# 2022-5-11

## PLAN
+ **毕业设计ppt**
+ **修改tf2tf 层数**

## tf2tf 层数修改

在de perturbation上有很大进展但是没法在tf perturbation上有进展

# 2022-5-14

## PLAN
+ **网络拓扑分析**
+ **进化树文章框架**
+ minimap2

## 网络拓扑

### 此前情况

有一定的推动，但是缺少可视化的结果，是否观察特定基因？

### 解决
问题在于两个网络的权值很小，一般都在0.08左右，有点不够大，因此调低了一些正则化参数

# 2022-5-15

## PLAN
+ **单细胞训练**
+ **minimap2算法**

## 单细胞训练训练步骤
+ Encoder Pretrain
+ Decoder Pretrain 1000 epochwes
+ Decoder mask
+ Decoder 网络中parameters*10
+ 微调
+ VAE微调



## minimap2

minimizer 找到{x,y,w} x 为ref 位置, y为query 位置 w为interval
找到一系列的anchers
最后可以确定位置

+ 首先，将基因组序列的minimizer存储在哈希表中（minimizer指一段序列内最小哈希值的种子）；
+ 然后，对于每一条待比对序列，找到待比对序列所有的minimizer，通过哈希表找出其在基因组中的位置，并利用chaining算法寻找待比对区域；
+ 最后，将非种子区域用动态规划算法进行比对得到比对结果。

# 2022-5-16

## PLAN
+ **单细胞修改模型可视化**
+ **拟南芥空间转录组**
+ **bwa 学习**


## 拟南芥空间转录组


### 验证

有空间数据子群识别以及收集更多转录本，同时找到更多标记鞇

### 应用

+ 细胞分形
+ 不同区域的基因表达关系
+ 不同区域的分化

## 模型修改结果

+ P_MEP,P_GMP(3000/5002) 恢复基因数显著多于 未恢复
+ 但在图上无法可视化

## NGS 学习

### SRA 命令

```bash
fastq-dump --split-3 --defline-qual '+' --defline-seq '@\$ac-\$si/\$ri'  SRR1945478 --gzip
```

下载压缩的双端测序数据

### bwa 命令

```bash
# bwa index -a bwtsw Col-0.fasta
bwa mem /home/ubuntu/data/NGS/Col-0.fasta /home/ubuntu/data/NGS/Col-0.fasta /home/ubuntu/data/NGS/SRR390728.fastq > /home/ubuntu/data/NGS/result/bwa_result.sam
```
+ 建立索引
+ 进行比对

