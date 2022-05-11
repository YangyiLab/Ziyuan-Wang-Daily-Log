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