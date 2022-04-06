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
+ 拟南芥项目推进
+ **PU1 perturbation**
+ 文献阅读
