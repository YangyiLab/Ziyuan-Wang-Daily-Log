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


# 2021-12-1

## PLAN
+ **硕士时间收集**
+ **修改penn state ps**
+ **机器学习作业 K-means聚类**

## 硕士时间

+  Yale biostatistics 12.15
+  密歇根 安娜堡 3.1
+  BU 滚动
+  UCLA 滚动
+  JHU 滚动

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
+ **MAE论文分析**trans_mCfile2bed_gz
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