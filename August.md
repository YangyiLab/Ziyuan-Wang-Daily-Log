- [2021-8-1](#2021-8-1)
  - [PLAN](#plan)
  - [phage 统计](#phage-统计)
  - [UIUC LAB](#uiuc-lab)
- [2021-8-2](#2021-8-2)
  - [PLAN](#plan-1)
  - [NTU LAB](#ntu-lab)
  - [群体遗传-固定系数](#群体遗传-固定系数)
    - [三种衡量](#三种衡量)
    - [Wahlund效应](#wahlund效应)
- [2021-8-3](#2021-8-3)
  - [PLAN](#plan-2)
  - [群体遗传--群体结构理想化模型](#群体遗传--群体结构理想化模型)
    - [大陆海岛模型](#大陆海岛模型)
    - [海岛海岛模型](#海岛海岛模型)
    - [无限海岛模型](#无限海岛模型)
    - [溯祖与亚群、前夕的关系](#溯祖与亚群前夕的关系)
- [2021-8-3](#2021-8-3-1)
  - [PLAN](#plan-3)
  - [Quite good LAB](#quite-good-lab)
  - [统计单链双链phage](#统计单链双链phage)
    - [gc skew](#gc-skew)
  - [the dataset](#the-dataset)
  - [群体遗传-突变](#群体遗传-突变)
    - [突变来源](#突变来源)
    - [基本概念](#基本概念)
  - [Pytorch 线性回归](#pytorch-线性回归)
# 2021-8-1
## PLAN
+ **GRE 填空2阅读2**
+ **收集双链phage序列并初步统计**
+ **修改abstract graphic**
## phage 统计
+ length

**Double Strand linear >>Single Strand circular**
+ gc_content
+ gc_skew
+ rbs_distribution/number

28 features
"GGAGGA" dominant

## UIUC LAB
https://www.sinhalab.net/people

# 2021-8-2
## PLAN
+ **GRE 填空2阅读2**
+ **西瓜书chapter1 2 3**
+ **abstract graphic定稿**
+ **群体遗传3**
+ **Cover Letter**

## NTU LAB
https://www.yqiaolab.com/research
https://gohwils.github.io/biodatascience/
https://dr.ntu.edu.sg/cris/rp/rp00715?ST_EMAILID=ZHAOYAN&CategoryDescription=BiomedicalSciencesLifeSciences
https://www.scelse.sg/People/Detail/5dcdf536-2f65-498b-84bd-c4edb768d659
https://swuertzlab.com/people2/
http://compbio.case.edu/koyuturk/index.html

## 群体遗传-固定系数
[Last section](July.md)
### 三种衡量
+ 观测平均杂合子$H_I$ 每一个亚群观测杂合频率平均
+ 预期平均杂合子$H_S$ 每个亚群杂合子期望再平均
+ 全群体杂合子 $H_T= 2\bar{p}\bar{q}$

### Wahlund效应

Wahlund效应：由于种群结构的存在，造成的分群预期平均杂合子频数比全群体预期杂合子数低的现象，即Ht>Hs。
$e.g.$ 人类很多隐性遗传病往往在孤立群体中发生率较高，而在经历较大基因流的群体中，杂合子频率升高，隐性遗传病发生概率较低。

按照Hardy-Weinberg法则估算的整个群体中纯合体的频率比各地方群体中的纯合体频率的平均值要低。

由于$p_i,q_i$不相等所以会导致这个现象

Fis衡量由于**亚群内**非随机交配造成的杂合子频数观测值和预期值的差异。
Fst衡量由于[种群结构](https://gitee.com/YangyiLab/Daily-Worklog/blob/master/July.md#%E7%BE%A4%E4%BD%93%E9%81%97%E4%BC%A0-%E7%BE%A4%E4%BD%93%E7%BB%93%E6%9E%84%E5%92%8C%E5%9F%BA%E5%9B%A0%E6%B5%81)的存在而造成的杂合子频数的降低。 **由于基因流有限，造成等位基因频率在群体中分布存在异质性**
Fit同时衡量了亚群内非随机交配和种群结构导致的杂合子频数的变异。

$$F_{st}=\frac{H_T-H_S}{H_T}$$
Fst是经常用来衡量种群结构和种群分化的参数
# 2021-8-3
## PLAN
+ **完成sbb投稿**
+ **GRE填空2阅读2**
+ **群体遗传3**
+ **西瓜书chapter 3**

## 群体遗传--群体结构理想化模型
### 大陆海岛模型
![大陆海岛图示](https://pic1.zhimg.com/80/v2-52f578ecd9ff778e0d4fcc2f96a2b0c8_720w.jpg)
假设:
+ 大陆足够大
+ 短期迁徙可以忽略
+ 海岛上每一代有m比例个体来自大陆，1-m比例个体来自海岛本身。
+ 无遗传漂变、无自然选择、无突变、随机交配、迁徙基因型来自随机样本。

**基于等式**
$$p_{t=1}^{island}-p_{t=0}^{island}=-m(p_{t=0}^{island}-p^{continent})$$
**递推公式**
$$p_{t}^{island}=p^{continent}+(1-m)^t(p_{t=0}^{island}-p^{continent})$$
$${\lim_{t \to + \infin}}p_t^{island}=p^{continnent}$$
结论
迁徙率决定了到达平衡状态的时间长短，与初始基因频率差异无关。

### 海岛海岛模型
本模型假设了种群迁徙对两个海岛的等位基因频率都产生影响。此模型下，两个海岛的等位基因频率最终会变得相同，达到平衡状态。

$${\bar{p}}=\frac{p_1m_2+p_2m_1}{m_1+m_2}$$
平衡状态频率更加接近于迁徙（入）率小的海岛的初始频率。

### 无限海岛模型
**当各个亚群之间有足够的基因流时（即各个亚群可以很好的混合），Ht和Hs是相等的，即Fst=0。**
当m=0，即没有基因流时，经过时间t之后，整个大群体的群体分化$F_{st}$指数分布 在遗传漂变的作用下，等位基因频率在各个海岛上要么固定，要么丢失。

单个种群中基因频率不变，杂合子频率逐渐减少$H_T$定值

$H_t= (1-\frac{1}{2N_e})^tH_0$ 表示单个种群杂合子频率降低

$$F_{st}=\frac{H_T-H_S}{H_T}$$
利用$N_e$进行估计，平衡值$F_{st}=\frac{1}{1+4N_em},N_e>>m$
FST取值范围[0,1],最大值为1，表明等位基因在各地方群体中固定，完全分化；
最小值为0，意味着不同地方群体遗传结构完全一致，群体间没有分化。
**上述模型是无限海岛模型，也就是等位基因频率不会在整个大群体得到固定或者丢失。**

Wright建议，实际研究中，FST为0～0.05:群体间遗传分化很小，可以不考虑；
FST为0.05～0.15，群体间存在中等程度的遗传分化；
FST为0.15~0.25，群体间遗传分化较大；
FST为0.25以上，群体间有很大的遗传分化。
### 溯祖与亚群、前夕的关系
当两个亚群之间迁徙率很低时（左图），溯祖往往在各个谱系先发生，然后再迁徙；当迁徙率很高时，分支可能在各个亚群之间频繁移动，一个分支的溯祖可能发生在另一亚群中

![迁徙与溯祖](https://pic1.zhimg.com/80/v2-d2bc90d0e1fbd4b025561e445437dba0_720w.jpg)

回顾溯祖所需时间:
+ 双样本溯祖所需时间为$2N_e$
+ 多个个题素组时间比较短

**含有迁移的溯祖**
来自同一个亚群的两个分支溯祖时间长短不受种群迁徙率的影响，只和大群体种群数量有关，数量越大，溯祖所需时间越长。
当两个分支来自两个不同亚群时，溯祖时间不仅受大群体数量影响，还受迁徙率影响。迁徙率越大，溯祖时间越短。

# 2021-8-3
## PLAN
+ **统计单链双链phage**
+ **GRE填空2阅读2**
+ **群体遗传4**
+ **西瓜书chapter 3代码在pytorch实现**

## Quite good LAB
+ http://blekhmanlab.org/research.html

## 统计单链双链phage
### gc skew
GC skew is when the nucleotides guanine and cytosine are over- or under-abundant in a particular region of DNA or RNA. In equilibrium conditions (without mutational or selective pressure and with nucleotides randomly distributed within the genome) there is an equal frequency of the four DNA bases (adenine, guanine, thymine, and cytosine) on both single strands of a DNA molecule.

## the dataset
Code
```python
import pyfastx
import numpy
import matplotlib.pyplot as mplt
import phanotate_modules.functions as phano
fa_ds = pyfastx.Fasta('ds-linear.fasta')
fa_ss = pyfastx.Fasta('ss-circular.fasta')
dataset=[]
for itm in fa_ds:
  list_itm=[]
  list_itm.append(len(itm.seq))
  list_itm.append(itm.gc_skew)
  list_itm.append(itm.gc_content)
  brbs=phano.get_backgroud_rbs(itm.seq)
  list_itm=list_itm+brbs
  list_itm.append("ds")
  dataset.append(list_itm)

for itm in fa_ss:
  list_itm=[]
  list_itm.append(len(itm.seq))
  list_itm.append(itm.gc_skew)
  list_itm.append(itm.gc_content)
  brbs=phano.get_backgroud_rbs(itm.seq)
  list_itm=list_itm+brbs
  list_itm.append("ss")
  dataset.append(list_itm)
```

Entity = ["length","gc_skew","gc_content","rbs1_content","rbs2_content"..."rbs28_content"]

## 群体遗传-突变
### 突变来源
+ 点突变
+ 插入缺失突变
+ 结构变异

中性突变--受drift 影响
有利有害--受自然选择影响
### 基本概念
+ **mutation fitness spectrum** 遗传适应谱系

一个密度分布函数图，表现了一个突变对于物种的有益或者有害
![mutation fitness spectrum](https://pic4.zhimg.com/80/v2-fb05a9def37b14077cf10b20f91f0187_720w.jpg)
+ mutation-accumulation 突变累积实验

用途:估计突变适应谱。估计多代后做突变处理的组和对照组的适应度变化和基因型的方差确定是否有利突变
## Pytorch 线性回归
步骤 
+ 定义一组输入数据；
+ 定义计算图；
+ 定义损失函数；
+ 优化，拟合参数。