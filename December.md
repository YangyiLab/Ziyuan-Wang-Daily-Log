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
+ 乔治城和kaust ps


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


