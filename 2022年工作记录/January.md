- [2021-1-3](#2021-1-3)
  - [PLAN](#plan)
  - [UTHealth 面试准备](#uthealth-面试准备)
    - [讲稿修改](#讲稿修改)
    - [研究兴趣](#研究兴趣)
    - [潜在教授](#潜在教授)
- [2022-1-4](#2022-1-4)
  - [PLAN](#plan-1)
  - [方差分析](#方差分析)
  - [文献阅读](#文献阅读)
    - [结果](#结果)
- [2022-1-5](#2022-1-5)
  - [PLAN](#plan-2)
- [2022-1-6](#2022-1-6)
  - [PLAN](#plan-3)
  - [微生物主旨](#微生物主旨)
- [2022-1-7](#2022-1-7)
  - [PLAN](#plan-4)
  - [单细胞进度](#单细胞进度)
    - [Pre-Training](#pre-training)
  - [微生物](#微生物)
    - [Result](#result)
    - [Discussion](#discussion)
  - [统计学习](#统计学习)
    - [分布复习](#分布复习)
    - [检验](#检验)
- [2022-1-8](#2022-1-8)
  - [PLAN](#plan-5)
  - [单细胞](#单细胞)
    - [单细胞数据检查](#单细胞数据检查)
    - [模型进度](#模型进度)
  - [人类hsc](#人类hsc)
- [2022-1-9](#2022-1-9)
  - [PLAN](#plan-6)
  - [单细胞可视化](#单细胞可视化)
  - [微生物论文](#微生物论文)
    - [Graphical Abstract](#graphical-abstract)
  - [CellOracle](#celloracle)
    - [Summary](#summary)
    - [造血细胞研究](#造血细胞研究)
- [2022-1-10](#2022-1-10)
  - [PLAN](#plan-7)
  - [tmux滚轮](#tmux滚轮)
  - [LTR 学习内容](#ltr-学习内容)
- [2022-1-11](#2022-1-11)
  - [PLAN](#plan-8)
  - [转录因子重建问题](#转录因子重建问题)
- [2022-1-11](#2022-1-11-1)
  - [PLAN](#plan-9)
  - [LTR研究](#ltr研究)
    - [基因破坏和表观遗传控制](#基因破坏和表观遗传控制)
  - [LTR进化分析](#ltr进化分析)

# 2021-1-3

## PLAN
+ **毕业设计修改**
+ **分子生物学复习**
+ **面试准备**

## UTHealth 面试准备

### 讲稿修改

+ 修改下菠菜研究的细节

### 研究兴趣

+ 单细胞、脑胶质瘤的开发 、
+ 基因组学 变异等
+ microbiome
+ VAE

### 潜在教授

+ Akdes Serin-Harmanci
  + https://www.nature.com/articles/ncomms14433
  + 脑胶质瘤和单细胞的兴趣

+ Jacqueline Chyr
  +  1) 癌症基因组中的选择性剪接和多聚腺苷酸化，
  +  2) 影响增强子-启动子相互作用的 3D 基因组组织的大规模和小规模分析
  +  3) 系统生物学和肠道微生物组代谢-代谢物网络建模
  +  4) 开发用于生物医学数据分析的 AI 工具。

# 2022-1-4

## PLAN
+ **分子生物学复习**
+ **ppt修改**
+ **文献阅读**
+ **统计学学习**

## 方差分析

$$F = \frac{组间方差/自由度}{组内方差/自由度}$$

## 文献阅读

*高级别胶质瘤谱系多样性的单细胞转录组分析*

网站 https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-018-0567-9

### 结果

Figure1 

![F1](https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs13073-018-0567-9/MediaObjects/13073_2018_567_Fig1_HTML.png?as=webp)

通过聚类并进行分类

通过染色体异倍型进行PCA降维

通过染色体异倍型对于transformed程度进行打分

e展示异常倍性

**判断转化细胞从marker基因和染色体拷贝数判断**

Figure2

![F2](https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs13073-018-0567-9/MediaObjects/13073_2018_567_Fig2_HTML.png?as=webp)

sox2 是Transformed cells的标志

# 2022-1-5

## PLAN
+ **分子生物学**
+ **面试**


# 2022-1-6

## PLAN
+ **微生物图片定稿**
+ **微生物结果改写**

## 微生物主旨

**氮肥和植物协同促进hitchhiking**

图一 没植物 画上氮肥微生物没有方向性
图二 加植物 画上氮肥微生物有方向性

# 2022-1-7

## PLAN
+ **单细胞进度**
+ **Result**
+ **模式图商议**
+ **整理工作思路**
+ **统计学学习**

## 单细胞进度

> 老鼠HSC数据集

最先计划 新数据集已有pipeline做训练(**在现有的network 正则化**)

### Pre-Training 

在现有的network 正则化 主要做 decoder 


## 微生物

### Result

完成了堆叠图、多样性以及功能、相关性图的描述。

**堆叠图**和**多样性**分析中修改较多

### Discussion

+ Bacillus 没菜没区别 加植物有区别 证明会影响Bacillus 影响细胞运动 影响hitchhiking
+ 通过加了植物后 不同氮肥下 Bacillus 不一样 中等浓度最高 **适量氮肥和植物促进Bacillus** 浓度过高 浓度过低的氮肥影响Bacillus运动
+ 结论 协同共同作用hitchhiking

另一段

+ 氮肥会促进植物生长 我们的结果表明了这个观点 植物生长图
+ 以往认为的原因 1 2 3 
+ 除了这些原因外 通过我们的实验，在加膜后，之间没有差异了（摆结果） 可能通过影响 hitchhiking 
+ 可以得出hitchhiking 促进植物生长

## 统计学习

### 分布复习


均值 正态 z分布 t分布

方差 卡方分布 自由度n-1

比例 t分布

相关系数 正态分布 t分布

### 检验


参数检验 样本量大 总体封据服从正态分布小样本服从分布

非参数 不要求总体分布 品质数据
品质数据往往不是随机变量 由好 很好 差等离散定序数据组成


# 2022-1-8

## PLAN
+ **微生物论文Discussion完成**
+ **单细胞数据检查**
+ **文献阅读**

## 单细胞

### 单细胞数据检查

mouse 数据 /home/ubuntu/data/insilico_pretrubation_data/mouse_epistasis_all.csv

human 数据 /home/ubuntu/data/insilico_pretrubation_data/human_epistasis_all.csv
/home/ubuntu/data/insilico_pretrubation_data/human_epistasis_gene.csv
/home/ubuntu/data/insilico_pretrubation_data/human_epistasis_cell_type.csv

### 模型进度

+ 已修改decoder模型
+ 未添加正则化

## 人类hsc

分化路径 层次


尚在问题 是否可以只调一个转录因子

成熟细胞和祖细胞关系


# 2022-1-9

## PLAN
+ **单细胞可视化**
+ **文献阅读**
+ **微生物论文定稿审查**

## 单细胞可视化

问题 从tfs -> genes 的tsne umap降维效果不好

**double check tf的顺序**

## 微生物论文

### Graphical Abstract
初稿修订完毕


## CellOracle

### Summary
在这里，我们介绍了 CellOracle，这是一种计算工具，它集成了单细胞转录组和表观基因组谱，以推断基因调控网络 (GRN)，这是细胞身份的关键调控因子。利用推断的 GRN，**我们模拟了响应转录因子 (TF) 扰动的基因表达变化**，使网络配置能够在计算机上进行询问，方便他们的解释。**我们验证了 CellOracle 的功效，可以概括造血过程中已知的调节变化，正确预测特征明确的 TF 扰动的结果**。将 CellOracle 分析与直接重编程的谱系追踪相结合，揭示了不同重编程故障模式下的不同网络配置。此外，沿着成功的重编程轨迹分析 GRN 重配置确定了提高靶细胞产量的新因素，揭示了 AP-1 亚基 Fos 与河马信号效应器 Yap1 的作用。总之，这些结果证明了 CellOracle 以高分辨率推断和解释细胞类型特异性 GRN 配置的功效，促进了对细胞身份调节和重编程的新机制见解

### 造血细胞研究

![HSC](https://www.biorxiv.org/content/biorxiv/early/2020/04/21/2020.02.17.947416/F3.large.jpg?width=800&height=600&carousel=1)

# 2022-1-10

## PLAN
+ **微生物论文定稿**

## tmux滚轮

ctrl+B [ 即可正常使用

## LTR 学习内容

+ 3'LTR 5'LTR 末端重复序列 根据这识别
+ 判断失活TE **用进化树聚类分析**

# 2022-1-11

## PLAN
+ 单细胞问题

## 转录因子重建问题

![s](../insilico-purterbation/repots_hsc_human/tfs.png)

蓝色 encoder 后隐藏层
橙色 隐藏层的交互层
绿色 输出层转录因子


# 2022-1-11

## PLAN
+ **LTR 结构研究**
+ **G4 甲基化统计**
+ **LTR初步设计**
+ **微生物文章**

## LTR研究

### 基因破坏和表观遗传控制

**外显子、内含子插入**

**通读转录**

**表观遗传学破坏**

## LTR进化分析

ML树 分析 $K_{ST}$数 进行评估等

详见GBE文献