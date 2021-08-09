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
- [2021-8-5](#2021-8-5)
  - [PLAN](#plan-4)
  - [群体遗传4-突变的结局](#群体遗传4-突变的结局)
    - [孟德尔分离](#孟德尔分离)
    - [突变在有限群体模型中的结局](#突变在有限群体模型中的结局)
    - [自然选择下的几何模型](#自然选择下的几何模型)
  - [Demography LAB](#demography-lab)
- [2021-8-6](#2021-8-6)
  - [PLAN](#plan-5)
  - [NUS LAB](#nus-lab)
  - [Boosting](#boosting)
- [2021-8-7](#2021-8-7)
  - [PLAN](#plan-6)
  - [群体遗传--突变模型](#群体遗传--突变模型)
    - [等位基因突变模型](#等位基因突变模型)
    - [DNA序列的突变模型](#dna序列的突变模型)
    - [突变对等位基因频率的影响](#突变对等位基因频率的影响)
- [2021-8-8](#2021-8-8)
  - [PLAN](#plan-7)
  - [决策树python实现](#决策树python实现)
  - [pytorch 流程 用三次方拟合sin](#pytorch-流程-用三次方拟合sin)
- [2021-8-9](#2021-8-9)
  - [PLAN](#plan-8)
  - [群体遗传--自然选择](#群体遗传--自然选择)
    - [适应度](#适应度)
    - [自然选择对不同表达方式基因所产生的作用](#自然选择对不同表达方式基因所产生的作用)
    - [自然选择增加平均适应性(等位基因频率和适应性关系)](#自然选择增加平均适应性等位基因频率和适应性关系)
    - [自然选择的基本定理](#自然选择的基本定理)
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

# 2021-8-5
## PLAN
+ **GRE阅读2填空2**
+ **tensorflow简单做降维**
+ **西瓜书第四章**
+ **群体遗传4**

## 群体遗传4-突变的结局
### 孟德尔分离
在突变二倍体中，有1/2的可能传递给子代，有1/2的可能丢失。如果有k个子代，那么该突变在所有子代中丢失的概率
$$P(mutationlost)=(\frac{1}{2})^k$$
对于一个群体，如果子代数量为Poisson分布，那么该群体经过一代之后，该（中性）突变在群体中丢失的概率是一个常数1/e = 0.3679
推导方法: 对K求全概率，K服从$\lambda=2$的泊松分布，原因，种群数量不变。

### 突变在有限群体模型中的结局
初始频率$p_0=1/N_e$ $p_0\rightarrow `
0$ 由扩散模型，扩散时间$4N_e$ 固定概率$1/2N_e$

### 自然选择下的几何模型
微突变：被自然选择固定的有利突变往往是微小作用的突变，而不是带来很大效应的突变。

**原因如图所示**

![](https://pic3.zhimg.com/80/v2-a667f1b8d85b4d548f2a158c1db38e82_720w.jpg)
靠近圆心则为有利突变

![](https://pic2.zhimg.com/80/v2-50d587bdb7b41075f7afaaa99e7d8e6d_720w.jpg)

m(突变强度，如突变设计的等位基因数)较小时，突变靠近有利变异的可能性较大。

**考虑drift时** m越小，drift的固定作用比较弱而自然选择作用又比较小，有利突变易丢失

## Demography LAB
+ https://www.med.upenn.edu/mathieson-lab/personnel.html
+ https://lohmueller.eeb.ucla.edu/research/
+ http://jjensenlab.org/people
+ https://popgen.gatech.edu/people/

# 2021-8-6
## PLAN
+ **GRE阅读2填空1**
+ 群体遗传4
+ **西瓜书chapter 4 和boosting**
+ **统计phage，给出统计学信息**

## NUS LAB
+ duke-nus.edu.sg/eid/faculty/tenure-track-faculty/tenure-track-faculty-staff-details/Detail/gavin-james-smith
+ https://www.duke-nus.edu.sg/cvmd/the-team/primary-appointment/Detail/owen-john-llewellyn-rackham
+ https://avianevonus.com/lab-members/

## Boosting
Core 给予每一个obs一个权重$w_i$
迭代依据
$$\alpha^{(s)}=log\frac{1-err^{(s)}}{err^{(s)}}$$
$$w_i^{(s)}=w_i^{(s-1)}exp(\alpha)^{(s)}$$

# 2021-8-7
## PLAN
+ **GRE套题1**
+ **群体遗传4**

## 群体遗传--突变模型
### 等位基因突变模型

无限等位基因模型：每次突变都产生一个新的从未有过的基因类型，后代中所有该基因型都是通过该突变得到的，identity by descent。

k 等位基因模型：假设某个位点的等位基因不是无限的，而是k个，每次突变都有k-1种可能。随着k的减小和突变率的增大， 群体中出现的两个相同的等位基因越来越有可能是多次突变事件形成的identity by state，而不是同一个突变事件identity by descent。

逐步突变模型：突变产生的基因型取决于当前基因状态，即突变并不是完全随机的。比如，在不同的DNA序列中，SNP突变的转换(CT/AG)比颠换(CA/CG/GT/AT)更容易发生。
(突变作用于状态，状态的衡量就是微卫星DNA片段的多态性，数量多态性，由整数表示the number of repeats)
**对于IBD的解释，两个具有相同repeats number的基因可能不是IBD e.g. 9+1=10-1**

### DNA序列的突变模型

无限位点模型：假设DNA序列无限长，每一个位点都有发生突变的可能，且每个位点最多只能经历一次突变。
**用途：短时间内近似**

有限位点模型：DNA序列长度有限，每个位点可能经历多次突变。

### 突变对等位基因频率的影响
+ 不可逆

$$p_t=p_0(1-\mu)^t$$
+ 可逆 利用马尔科夫性，之和转移概率有关

$P(a\rightarrow A)= \mu, P(A\rightarrow a)= \lambda$
平衡概率$P_{equilibrium}=\frac{\lambda}{\lambda+\mu}$

+ 考虑遗传漂变 计算杂合率 计算方法和migration gene flow 类似 加入因子

平衡$F=\frac{1}{4N_e\mu+1}$
即在一个无限等位基因模型中，在漂变-突变平衡的群体中，两个随机抽样的等位基因是非同源等位基因allozygous的概率。随着theta的增大，两个等位基因来自非同源的可能性增加，而来自同源等位的概率越来越小。
![allozygous](https://dyerlab.github.io/applied_population_genetics/media/allele_lineages.png)

# 2021-8-8
## PLAN
+ **GRE阅读2填空1**
+ **西瓜书chapter 4 sklearn**
+ **修改abstract定稿**

## 决策树python实现
网址 https://sklearn.apachecn.org/docs/master/11.html
示例代码

```python
from sklearn import tree
X = [[0, 0], [1, 1],[1,2]]
Y = [0, 1, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf.predict_proba([[2., 2.]])
```

## pytorch 流程 用三次方拟合sin
```python
import numpy as np
import math

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
```

# 2021-8-9
## PLAN
+ GRE阅读4
+ 群体遗传5
+ 西瓜书chapter 5
+ **修改研究进展ppt**

## 群体遗传--自然选择
### 适应度
相对适应度(相对于B基因)$w=\frac{\lambda_A}{\lambda_B}$
选择系数$s_{xx}=1-w_{xx}$ 选择系数越高，适应性越差
### 自然选择对不同表达方式基因所产生的作用
**基因的表达方式可以概括为5种：隐性(被选择的基因为隐性，且杂合不被选择)、显性(被选择的基因为显性，且杂合不被选择)、一般显性(被选择的基因为显性，且杂合被选择作用会被中和系数h调整而非只是选择系数s)、杂合劣势(纯合子被选择。杂合子相对来说生存优势比较低)、杂合优势(杂合子生存优势高)。**
自然选择过程的特点--基因频率每一代改变，基因型频率服从HW平衡

Tips:
+ Dominance -- 杂合子的屏蔽作用 sheltering effect
+ 一般象形的共显性数 condominance h 0~1 时图像 0对应隐形拮抗 1对应线性拮抗
+ 杂合劣势 只与其实基因频率有关 **起始等位基因频率高于0.5时，该等位基因会得到固定，而低于0.5时，该等位基因会消失。**
+ 杂合优势 

不管起始频率是高于0.5，还是低于0.5，最后该等位基因频率总是趋近于0.5（假设两个纯合子适应性相同）。杂合子的频率在群体中占有多数。

这和前面提到的几种选择作用不同。前面不管是对显性基因的选择，对隐性基因的选择，还是对杂合子劣势，自然选择的结果都是清除群体中的某一个基因，使得群体多态性降低，也称之为单态平衡monomorphic equilibrium。而杂合优势带来的结果是维持了群体的多态性，没有使得其中一个等位基因固定或者消失，所以杂合优势导致的结果也称之为多态平衡polymorphic equilibrium。

+ 杂合优势的计算方法 基线法，当$\Delta p =0$条件计算

### 自然选择增加平均适应性(等位基因频率和适应性关系)
自然选择的结果是增加的群体的平均适应性

显性、隐形一般线性，自然选择趋向于增加平均适应性

杂合优势和杂合劣势的适应性和等位基因频率

![heter](https://pic2.zhimg.com/80/v2-cee559d46be42ba2fdaa0903ce142451_720w.jpg)

自然选择倾向增大平均适应性
在杂合优势时（左图），当两个等位基因频率相同的时候，群体平均适应性最高。此时等位基因频率最稳定，当一个等位基因低于该频率时，自然选择会驱使它回到该频率。

在杂合劣势时（右图），当两个等位基因频率相同的时候，群体平均适应性最低。此时等位基因频率最不稳定，一旦有微小偏离，自然选择会驱使出现更大的偏离。

### 自然选择的基本定理
Fisher通过公式证明了经过一代自然选择，群体平均适应性的改变等于群体适应性方差（变异）的大小。
$$\Delta \bar{w} =var(w)$$
$$\Delta \bar{w} =\frac{\bar{w}'- \bar{w}}{\bar{w}}$$
$\bar{w}$ denotes the average of fitness after the natural selection.
即一个群体的各个个体的适应性变异很大，那么经过自然选择该群体的平均适应性变化同样会很大。