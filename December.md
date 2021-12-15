- [2021-12-1](#2021-12-1)
  - [PLAN](#plan)
  - [硕士时间](#硕士时间)
  - [机器学习K_Means 作业](#机器学习k_means-作业)
- [2021-12-2](#2021-12-2)
  - [PLAN](#plan-1)
- [2021-12-3](#2021-12-3)
  - [PLAN](#plan-2)
  - [GRU 训练中trick](#gru-训练中trick)
- [2021-12-4](#2021-12-4)
  - [PLAN](#plan-3)
  - [微生物更改](#微生物更改)
- [2021-12-5](#2021-12-5)
  - [PLAN](#plan-4)
- [2021-12-6](#2021-12-6)
  - [PLAN](#plan-5)
  - [甲基化g4 pipeline](#甲基化g4-pipeline)
    - [下载方式](#下载方式)
  - [VAE 回顾](#vae-回顾)
  - [dca](#dca)
  - [mae](#mae)
- [2021-12-7](#2021-12-7)
  - [PLAN](#plan-6)
  - [vae pmbc](#vae-pmbc)
  - [MAE 论文](#mae-论文)
- [2021-12-8](#2021-12-8)
  - [PLAN](#plan-7)
  - [pmbc 结果可视化](#pmbc-结果可视化)
    - [按细胞](#按细胞)
    - [按gene](#按gene)
  - [微生物](#微生物)
- [2021-12-9](#2021-12-9)
  - [PLAN](#plan-8)
  - [MAE 相关研究](#mae-相关研究)
    - [Fine Tuning](#fine-tuning)
- [2021-12-10](#2021-12-10)
  - [PLAN](#plan-9)
  - [Transformer](#transformer)
    - [Self-Attention](#self-attention)
- [2021-12-11](#2021-12-11)
  - [PLAN](#plan-10)
  - [transformer学习](#transformer学习)
    - [attention以及self-attention](#attention以及self-attention)
    - [权重值](#权重值)
  - [文献nature. method](#文献nature-method)
    - [引言](#引言)
    - [建图](#建图)
    - [StructDB](#structdb)
- [2021-12-12](#2021-12-12)
  - [PLAN](#plan-11)
  - [甲基化G4论文](#甲基化g4论文)
- [2021-12-13](#2021-12-13)
  - [PLAN](#plan-12)
  - [文献阅读](#文献阅读)
    - [G4基序与甲基化的关系](#g4基序与甲基化的关系)
- [2021-12-14](#2021-12-14)
  - [PLAN](#plan-13)
  - [Penn state](#penn-state)
    - [PPT overview](#ppt-overview)
    - [问题](#问题)
- [2021-12-14](#2021-12-14-1)
  - [PLAN](#plan-14)
  - [ppt](#ppt)
    - [拟南芥](#拟南芥)


# 2021-12-1

## PLAN
+ **硕士时间收集**
+ **修改penn state ps**
+ **机器学习作业 K-means聚类**

## 硕士时间

+  Yale biostatistics 12.15
+  密歇根 安娜堡 3.1
+  BU 滚动
+  UCLA 滚动 (NG)
+  JHU 滚动
+  Brown 2.1

## 机器学习K_Means 作业
初步代码需要调整数据集

# 2021-12-2

## PLAN
+ **训练模型**
+ **penn_state_ps**

# 2021-12-3

## PLAN
+ **GRU 结果**
+ **机器学习文档**
+ **penn state**

## GRU 训练中trick

+ 学习率必须比较低 0.001
+ Dropout 防止过拟合 设置为0.2

# 2021-12-4

## PLAN
+ **机器学习几次作业完成**
+ **loss plot做出，准确率输出**

## 微生物更改

Bacillus content was 2.63, 7.29 and 2.83(H1, H2 and H3) times higher in the non-membrane treatment group than in the membrane treatment group, respectively.

# 2021-12-5

## PLAN
+ **机器学习PPT 讲稿**
+ **Yale PS**
+ **K MEANS 作业整理**

# 2021-12-6

## PLAN
+ **MAE论文分析**
+ **single cell denoise 分析**
+ **甲基化g4 pipeline**
+ **vae 回顾**

## 甲基化g4 pipeline

### 下载方式

找到网址
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1085222

wget即可下载

下载好tsv.gz文件，直接使用

```python
def trans_mCfile2bed_gz(path,input_file_name):
    '''
    'path' the path to the folder of the the mC tsv.gz file
    'input_file_name' the name of the the mC tsv.gz file
    this will creat a new bed file like xxx.mC.bed file in the
    fold showed in the parametre path
    '''
    f = gzip.open(path+input_file_name, 'rb')
    mc_data=f.read()
    mc_lines=str(mc_data,encoding='utf-8').split("\n")
    # [judge_if_mC(i) for i in mc_lines]
    mc_true_lines=list(filter(judge_if_mC,mc_lines))
    list(mc_true_lines)
    mc_true_lines_out=[trans2bed(line) for line in mc_true_lines]
    output_file_name=input_file_name[20:-4]+".mC.bed"
    f_bed=open(path+output_file_name,"w")
    f_bed.write("\n".join(mc_true_lines_out))
    output_file_name=input_file_name[20:-4]+".mC.bed"
    f_bed.close()
    return output_file_name
```

bedtools coverage 统计G4 含有甲基化位点的比例

```bed
染色体号  g4起始  g4终止  序列  长度  正负链  评分  甲基化位点hit_number  甲基化位点coverage_length 长度again 甲基化位点coverage_ratio
chr1    25335134        25335174        GGGTTTTGTCGGTCCGGGTTAGGGTAAAAGCGGGTCCGGG        40      +       6.68    0       0       40      0.0000000
chr1    25449721        25449752        CCCGCCCTGGTGCTGCATCCCATGCATGCCC 31      -       18.62   3       3       31      0.0967742
chr1    25449865        25449896        CCCCCCCTGGTGCTGCATCCCATGCATGCCC 31      -       22.85   2       2       31      0.0645161
chr1    25449937        25450006        CCCCCCCTGGTGCTGCATCCCGTGCATGCCCTGGTGCTGCATCCCGTGCCCGCCCTGGTGCTGCATCCC   69      -       19.56   9       9       69      0.1304348
chr1    25752671        25752713        CCCTTAATCTTATCCCCAAATTCGAAACCCTAATTAGCTCCC      42      -       3.29    0       0       42      0.0000000
```

bedtools 加 -hist 效果更好

all     0       45550   46089   0.9883052
all     1       539     46089   0.0116948

直接统计好覆盖率

```py
def calculate_coverage_mC_g4(g4_file,mC_file):
    tmp1 = "tmp1.bed"
    tmp2 = "tmp2.bed"
    os.system("bedtools coverage -a "+g4_file+" -b "+mC_file+" -s " + " > "+ tmp1)
    os.system("bedtools coverage -a "+g4_file+" -b "+mC_file+" -s -hist " + " > "+ tmp2)
    [covered,total,rate,hist] = mC_coverage_parser(tmp1,tmp2)
    
    os.system("rm "+tmp1)
    os.system("rm "+tmp2)
```

[covered,total,rate,hist] 有甲基化的G4 全部G4 比例 以及加入hist参数后 all     1       539     46089   0.0116948

## VAE 回顾

分析思路 https://cloud.tencent.com/developer/article/1764757

pmbc数据集降维可以分析 直接利用sc.pl.umap

pmbc 也可以直接训练 但需要转录本做对应

```py
import numpy as np
genes = list(adata.var['gene_symbol'])
f = open('/home/ubuntu/MLPackageStudy/VAE/tf-homo-current-symbol.dat','rb')
tfs = f.read()
tfs = str(tfs,encoding = 'utf-8')
tfs = tfs.split('\r\n')
tfs
tfs_pmbc = set(genes) & set(tfs)
len(tfs_pmbc) , len(tfs)
``` 
重新确定转录本 利用pmbc数据集做
    

## dca

本质：数据清洗针对单细胞数据零值较多进行处理
负二项分布有特点为 方差随均质增大而增大的特点 适合单细胞
该文章问题在于降维无解释性

## mae

图片预训练 找隐变量。按照模块mask 想十分创新

# 2021-12-7

## PLAN

+ **完成BU申请**
+ **VAE 训练代码**
+ **修改RNN和LSTM的训练代码**
+ **甲基化pipeline**

## vae pmbc
将pmbc的tfs 和给出的tfs做了 交集 

```py
adata = sc.read_h5ad("/home/ubuntu/MLPackageStudy/VAE/in-silico/train_pbmc.h5ad")
adata.X = adata.X.A
sc.settings.verbosity = 3 


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

#标准化数据
sc.pp.log1p(adata)
sc.pp.normalize_total(adata)

gene_names = list(adata.var['gene_symbol'])
f = open('/home/ubuntu/MLPackageStudy/VAE/tf-homo-current-symbol.dat','rb')
tfs = f.read()
tfs = str(tfs,encoding='utf-8')
tfs = tfs.split('\r\n')
tfs_pmbc = set(gene_names) & set(tfs)
tfs_pmbc = list(tfs_pmbc)
# data_z = adata.X

batch_size = 128
learning_rate = 1e-5
patience = 20
data_z_genes = adata.X

data_z = adata[:,tfs_pmbc].X
```

## MAE 论文

关注pretraining 后分类问题


# 2021-12-8

## PLAN
+ **pmbc 结果可视化**
+ **微生物讨论初步**
+ **RNN LSTM训练**

## pmbc 结果可视化

### 按细胞
+ 不同类型做平均 
+ plot

### 按gene
同上

## 微生物

Bacilli H1 H3< H2

COR-edges H1 H3< H2

# 2021-12-9

## PLAN
+ **CMU YALE申请结束**
+ **MAE相关研究**
+ **RNN 可视化**

## MAE 相关研究
+ ViT
+ BEIT

### Fine Tuning

由 pretraining 到直接训练

pre-training 是生成模型 得到隐变量后 分类等问题

Patch Embedding Position Embedding https://bbs.cvmart.net/articles/4461

https://arxiv.org/abs/2010.11929

$\mathbf{xp}\in \mathbb{R}^{N\times(P^2\cdot C)}$

$\mathbf{x'{p}}\in \mathbb{R}^{N\times D}$

PATCH 拉成1D特征为$P^2\cdot C$
patches 拉到D维度
$N=HW/P^2$
$\mathbf {x'_{p}}\in \mathbb{R}^{N\times D}$

![ViT数学表达](https://pic1.zhimg.com/v2-cb632e9df1dbc49e379799a0417e9b34_b.jpg)

# 2021-12-10

## PLAN

+ **乔治城 kaust 申请 overview**
+ **RNN LSTM 可视化总结**
+ **Transform 学习**
+ **in-silico 开始训练**

## Transformer

[paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

+ 只用self-Attention 的架构
+ Encoder-Decoder 输出 是auto-regression 每次输出依赖上一次
+ Transformer 使用编码器解码器架构
+ Problem *multi-head Attention 机制*
+ LayerNorm vs BatchNorm 对每一个样本算均值方差

### Self-Attention

一种替代RNN的方案
$$ y_i = f(\mathbf{x},(x_1,x_1),(x_i,x_i),...,) \in \mathbb{R}^d$$ 


$\mathbf{y} $ 为抽取的特征向量 $n\times d$
得到编码矩阵
$$\mathbf{X}\in \mathbb{R}^{n\times d}$$

**位置信息丢失**
*解决方案* Positional Encoding 利用在序列中的位置(0...n)和维度(0...d) 得到编码矩阵
$$\mathbf{P} \in \mathbb{R}^{n\times d}$$

最终结果
$$\mathbf{X}=\mathbf{X}+\mathbf{P}$$


# 2021-12-11

## PLAN
+ **Transform 学习**
+ **nc 论文**
+ **乔治城和kaust ps**


## transformer学习

### attention以及self-attention
https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1703.03130

Self-attention应用于文本情感分析


Attention的用途是在seq2seq中对于解码器给出输入让翻译可以专注于语义词，表示出重要程度

![attention](https://pic3.zhimg.com/80/v2-7113d028d878ebcf8654d3ce6b54fa36_1440w.jpg)



### 权重值

对于attention 权重值由对应位置的编码器embedding和上时刻的解码器的隐状态共同决定

![Image](https://pic4.zhimg.com/80/v2-5561fa61321f31113043fb9711ee3263_1440w.jpg)

![Image](https://pic1.zhimg.com/80/v2-50473aa7b1c20d680abf8ca36d82c9e4_1440w.jpg) 

## 文献nature. method

[URL](https://www.nature.com/articles/s41592-021-01286-1.pdf)

### 引言

线性与非线性降维优缺点
Trajectory estimation ？要查文献

### 建图

knn knn 时间序列等方法

优点：求解xia对轻松比gcn 因为有解析解

### StructDB

**Pseudotime**

源头 monocle2 

先聚类后找路径

找路径
+ 最小生成树
+ 反向嵌入图 则是先对细胞进行聚类，再对细胞群的平均值进行轨迹构建。用**最小生成树**
+ RNA velocity (gene expression trajectory) 从non-splicing 到splicingm 判断发育轨迹，不需要给出发育起点

> 由于RNA velocity分析的前提是要我们从单细胞RNA-seq的数据中区分出未成熟的mRNA(unspliced)和成熟的mRNA(spliced)，所以需要从fastq文件开始，与基因组进行比对后得到sam文件，从sam文件转成bam文件，再从bam文件中提取spliced，unspliced和ambiguous信息。得到.loom为后缀的文件。


# 2021-12-12

## PLAN

+ **KAUST 申请填写**
+ **范德堡账号申请材料overview**
+ **Madison Master Overivew**
+ **甲基化论文初步寻找**

## 甲基化G4论文
+ Spiegel, J., Cuesta, S. M., Adhikari, S., Hänsel-Hertsch, R., Tannahill, D., & Balasubramanian, S. (2021). G-quadruplexes are transcription factor binding hubs in human chromatin. Genome Biology, 22(1), 1–15. https://doi.org/10.1186/s13059-021-02324-z
+ Zhang, S., Li, R., Zhang, L., Chen, S., Xie, M., Yang, L., Xia, Y., Foyer, C. H., Zhao, Z., & Lam, H. M. (2020). New insights into Arabidopsis transcriptome complexity revealed by direct sequencing of native RNAs. Nucleic Acids Research, 48(14), 7700–7711. https://doi.org/10.1093/nar/gkaa588 *CONCERNED WITH WENYAN LEI*
+ Wu, F., Niu, K., Cui, Y., Li, C., Lyu, M., Ren, Y., Chen, Y., Deng, H., Huang, L., Zheng, S., Liu, L., Wang, J., Song, Q., Xiang, H., & Feng, Q. (2021). Genome-wide analysis of DNA G-quadruplex motifs across 37 species provides insights into G4 evolution. Communications Biology, 4(1). https://doi.org/10.1038/s42003-020-01643-4  *CONCERNED WITH WENYAN LEI*
+ Iurlaro, M., McInroy, G. R., Burgess, H. E., Dean, W., Raiber, E. A., Bachman, M., Beraldi, D., Balasubramanian, S., & Reik, W. (2016). In vivo genome-wide profiling reveals a tissue-specific role for 5-formylcytosine. Genome Biology, 17(1), 1–9. https://doi.org/10.1186/s13059-016-1001-5

# 2021-12-13

## PLAN
+ **推荐信**
+ **文献阅读**
+ **地理位置标注overview**
+ **毕业设计overview**
+ **vae修改模型**

## 文献阅读
已作出巨大努力通过计算预测探索 G4 序列、结构和分布在植物19、20、原核生物21、病毒22、23和真核生物24 中的基因组景观和特征，这表明通用生物基因组中存在广泛的模式。以及这些结构在基因25 , 26的启动子和 5' 调节区中的功能意义。最近的两项研究使用高通量结构测序方法绘制了人类和其他生物体的全基因组 G4 结构图6 , 27. 值得注意的是，在这些研究中，通过 G4 结构测序方法6证实了生物信息学分析预测的绝大多数 G4 结构（80-90%）存在于基因组中，这表明基于预测的方法的可行性。

分析表明，随着物种的进化（图 1a），G4 基序密度和长度与整个基因组的比例增加（图 1b、c；补充表 1）)，表明虽然基因组中G4s数量的增加部分与基因组大小的增加有关，但G4s数量和密度的增加是由于物种复杂性的增加。基因组模拟测试进一步支持了这一建议

### G4基序与甲基化的关系

文章 Genome-wide analysis of DNA G-quadruplex motifs across 37 species provides insights into G4 evolution

**DNA 甲基化和 G4 结构都发生在富含 GC 的区域**

使用具有高度甲基化基因组的哺乳动物物种猪和具有相对低甲基化的物种家蚕来检查 G4 基序与上游甲基化之间可能的进化关系 2 kb 基因区域。一般来说，在高度甲基化的猪基因组中，携带 (G/C) 3 L 1-7的基因上游 2 kb 区域的胞嘧啶显示出比整个基因组中的甲基化水平显着较低的甲基化水平（图 6a））。我们进一步绘制了基因上游 2 kb 区域 200 bp 滑动窗口的甲基化水平，发现在每个 200 bp 窗口中，(G/C) 3 L 1-7基因的甲基化水平明显低于在整个基因集背景中（图 6b）。

在稀疏甲基化的家蚕基因组中，(G/C) 3 L 1-7基因的上游区域的甲基化胞嘧啶（mCs）的频率显着低于所有基因（图 6c）。当检查这些 mCs 的甲基化水平时，(G/C) 3 L 1-7基因的高甲基化胞嘧啶的频率显着低于所有基因（图 2d )。 *可能是未折叠*



**这些结果表明 DNA G4s 和甲基化之间的拮抗关系可能存在并且可能是保守的，至少在哺乳动物和昆虫中是这样。**

![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs42003-020-01643-4/MediaObjects/42003_2020_1643_Fig6_HTML.png?as=webp)

**对于G4预测是用软件预测的**

# 2021-12-14

## PLAN
+ **KAUST 提交**
+ **PENN STATE 稿**
+ **PENN STATE PPT**
+ 群体遗传复习
+ **简历复习**

## Penn state

### PPT overview

+ 标题页
+ Background
+ Skills
+ Research Summary: 1. 拟南芥 2. Phage Genome 3. SaAlign
  + 拟南芥1001Genome
    + 背景 
      + 1001Genome distribution 图
      + G4 示意图
      + TE 示意图
      + 甲基化 示意图
    + 猜想 G4 在LTR末端 甲基化 如果不折叠 可能甲基化 导致分化
    + Work
      + 1. G4 Indentification 2. TE Indentification 3. 甲基化示意图
      + G4被甲基化比率 与全体基因组甲基化比率
      + TE差异识别/ TE转座子
      + Because quadruplexes formed on one strand would theoretically leave the other strand in a single- stranded state, it is possible they could hinder methylation of the surrounding sequences, even if they were rich in CpG and other methylable nucleotide pairs. 
    + 未来工作
      + TE 转座子 G4三者关系
  + Phage Genome
    + 背景
      + gene caller 算法回顾 GLIMMER PRODIGAL PHANOTATE
      + PHAGE genome
      + metagenome for phage genome
    + work
      + Random Forest
      + MLP
      + LSTM/RNN/GRU
    + 进一步工作
      + Atthention Transform架构
      + pre-training small samples
  + SaAlign
    + 背景 MSA
    + Work Overview

### 问题

+ Introduction

My name is Ziyuan Wang, a fourth-year undergraduate from Sichuan University, majoring in computational biology.I have been taking plenty of courses related to biology, statistics, and programming during my three-year undergraduate study, during which I got an overall GPA of 3.92/4.00, with the rank in my major as 1/27. My research interests consist of population genetics and deep learning.

+ why school

The reputation of PSU all over the world and in China is great.Penn State’s Department of Biology is one of the top-ranked biology departments in the United States. Penn State has a lot of research funding and can produce good research results.
Dr. Huang's research is attracting. He is good at finding scientific problem and using proper computational and deep learning method to solve them.

+ why phd

First of all, I am very interested in research, and I enjoy the process of solving problems. At the same time, I really want to use computational and statistical methods to solve problems related to human diseases. At the same time, my future goal is to pursue an academic career. Obtaining a doctoral degree can improve my competitiveness, and participating in a doctoral program can enable me to learn more professional knowledge under the guidance of advisor.

+ obstacles

# 2021-12-14

## PLAN
+ **面试稿**
+ sc可视化
+ 群体遗传概念复习

## ppt

### 拟南芥

The main question we want to explore is, what factors influence transposon activity. 

Geographic factors, including altitude and geographical location, are known to affect the Arabidopsis genome. As shown here, despite the Arabidopsises are close to each other, genome variation can be large. The root cause is that the genomes of different arabidopsis species are located different growth conditions.

G4 is a short piece of DNA in the genome that, when folded, blocks methylation on CpG island. The role of transposons in genome changes in Arabidopsis thaliana has been reported in many studies. However, methylation can inactivate transposons leading to genome differences between different species. 

LTR is a retrotransposon that can be copied and pasted.LTRs  are often the target of epigen- etic regulation, whereas retrotransposons are methylated and inactivated by the host. G4s have been observed in unmethylated regions of genomes of different kingdom before. We speculate that the presence of G4s in LTRs may be related to such inactivating mechanism, probably by interfering with the methylation process. Because G4s formed on one strand would theoretically leave the other strand in a single- stranded state, it is possible they could hinder methylation of the surrounding sequences, even if they were rich in CpG.


It has been speculated that the activity of transposable enzymes is affected by different geographical and climatic conditions. But here, we propose that different geographical and climatic environments will affect the folding of G4, and failure of G4 to fold will lead to methylation of CpG islands on both sides of LTR and inactivation of transposons.