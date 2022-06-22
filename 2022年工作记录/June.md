- [2022-6-2](#2022-6-2)
  - [PLAN](#plan)
  - [RNA-seq流程](#rna-seq%E6%B5%81%E7%A8%8B)
    - [illumina dragen](#illumina-dragen)
    - [software](#software)
  - [正则化](#%E6%AD%A3%E5%88%99%E5%8C%96)
    - [L1](#l1)
    - [L2](#l2)
    - [conclusion](#conclusion)
- [2022-6-4](#2022-6-4)
  - [PLAN](#plan-1)
  - [RNA-seq](#rna-seq)
    - [histat](#histat)
    - [STAR](#star)
    - [FeatureCounts](#featurecounts)
    - [Result](#result)
  - [L1正则化pytorch](#l1%E6%AD%A3%E5%88%99%E5%8C%96pytorch)
- [2022-6-5](#2022-6-5)
  - [PLAN](#plan-2)
  - [STAR](#star-1)
- [2022-6-6](#2022-6-6)
  - [PLAN](#plan-3)
  - [Single Cell](#single-cell)
    - [Droplet-based RNA-seq](#droplet-based-rna-seq)
    - [UMI](#umi)
  - [范数计算](#%E8%8C%83%E6%95%B0%E8%AE%A1%E7%AE%97)
  - [GWAS](#gwas)
- [2022-6-7](#2022-6-7)
  - [PLAN](#plan-4)
  - [journal club](#journal-club)
  - [转座子$F_st$](#%E8%BD%AC%E5%BA%A7%E5%AD%90fst)
- [2022-6-8](#2022-6-8)
  - [PLAN](#plan-5)
  - [整合单细胞和bulk 转录组](#%E6%95%B4%E5%90%88%E5%8D%95%E7%BB%86%E8%83%9E%E5%92%8Cbulk-%E8%BD%AC%E5%BD%95%E7%BB%84)
  - [单细胞深度学习模型](#%E5%8D%95%E7%BB%86%E8%83%9E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B)
- [2022-6-9](#2022-6-9)
  - [PLAN](#plan-6)
  - [孟德尔随机化](#%E5%AD%9F%E5%BE%B7%E5%B0%94%E9%9A%8F%E6%9C%BA%E5%8C%96)
    - [Example](#example)
  - [TF推测](#tf%E6%8E%A8%E6%B5%8B)
  - [MSA转到VCF](#msa%E8%BD%AC%E5%88%B0vcf)
  - [NGS拼接问题](#ngs%E6%8B%BC%E6%8E%A5%E9%97%AE%E9%A2%98)
  - [单细胞模型训练参数](#%E5%8D%95%E7%BB%86%E8%83%9E%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0)
    - [1/-1](#1-1)
    - [直接mask 不修改权重大小](#%E7%9B%B4%E6%8E%A5mask-%E4%B8%8D%E4%BF%AE%E6%94%B9%E6%9D%83%E9%87%8D%E5%A4%A7%E5%B0%8F)
- [2022-6-11](#2022-6-11)
  - [PLAN](#plan-7)
- [2022-6-20](#2022-6-20)
  - [PLAN](#plan-8)
  - [单细胞网络训练](#%E5%8D%95%E7%BB%86%E8%83%9E%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83)
    - [L1正则训练](#l1%E6%AD%A3%E5%88%99%E8%AE%AD%E7%BB%83)
    - [两层网络](#%E4%B8%A4%E5%B1%82%E7%BD%91%E7%BB%9C)
  - [拟南芥转座子G4思路](#%E6%8B%9F%E5%8D%97%E8%8A%A5%E8%BD%AC%E5%BA%A7%E5%AD%90g4%E6%80%9D%E8%B7%AF)
- [2022-6-21](#2022-6-21)
  - [PLAN](#plan-9)
  - [Cover letter](#cover-letter)
    - [Reviewer #3](#reviewer-3)
    - [Reviewer #4](#reviewer-4)
    - [Reviewer #5](#reviewer-5)
- [2022-6-22](#2022-6-22)
  - [PLAN](#plan-10)
  - [Reviewer #4回答](#reviewer-4%E5%9B%9E%E7%AD%94)
  - [Reviewer #5 回答](#reviewer-5-%E5%9B%9E%E7%AD%94)

# 2022-6-2

## PLAN

+ **RNA-seq pipeline overview**
+ **VCFtools 脚本完善**
+ **正则化学习**

## RNA-seq流程

数据依赖 参考基因组fasta文件 **基因注释gff3**(与WGS不同)

步骤
+ 下载数据利用sratools
+ 质控
+ mapping 可以使用bwa botie或者hisat2
+ 定量->质控->构建表达矩阵

### illumina dragen

![DRAGEN](https://d3f3vsz1y2oz5i.cloudfront.net/9662654/screenshots/443e175f-76dd-4743-8e93-9fd82a6c2db7_800_500.png)

illumina 开发的rna-seq pipeline

集成到测序仪

### software

bwa 因为BW transformation无法做到剪接

+ minimap2 splicing
+ STAR ? 需要去看原始文件
+ kallisto **psudo alignment**

## 正则化

### L1

$$\sum||\mathbf{w_i}||^2$$

网络更加稀疏，但是解不稳定

由于解在坐标轴，因此稀疏。

### L2

$$\sum||\mathbf{w_i}||$$

解更稳定，但0的含量没那么高，而每一个weight的权重都下降了、

### conclusion

训练时试试L1 正则化

# 2022-6-4

## PLAN

+ **完善RNA-seq脚本**
+ **overview L1正则化pytorch写法**
+ **学习STAR软件**

## RNA-seq

### histat

代码
用BWA软件没有考虑内含子剪接问题
```bash

hisat2-build Prunus_persica_v2.0.a1_scaffolds.fasta Prunus_persica_v2.0.a1_scaffolds   

#######################################################################
##########                      Index                        ##########
#######################################################################

hisat2  -p 4 \
        -x /home/ubuntu/data/NGS/Prunus_persica/Prunus_persica_v2.0.a1_scaffolds \
        -1 /home/ubuntu/data/NGS/Prunus_persica/rawdata/SRR8596341_1.clean.fastq.gz \
        -2 /home/ubuntu/data/NGS/Prunus_persica/rawdata/SRR8596341_2.clean.fastq.gz \
        -S /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sam 


samtools sort -o /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sorted.bam /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sam  
samtools view -b -S /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sam -o /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.bam

samtools sort \
-@ 4 \
-m 2G \
/home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.bam   \
/home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sorted

samtools index /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sorted.bam
#######################################################################
##########                    Alignment                      ##########
#######################################################################
 
```

### STAR

+ Index

```bash
STAR --runThreadN 4 --runMode genomeGenerate \
 --genomeDir /home/ubuntu/data/NGS/Prunus_persica \
 --genomeFastaFiles /home/ubuntu/data/NGS/Prunus_persica/Prunus_persica_v2.0.a1_scaffolds.fasta \
 --sjdbGTFfile /home/ubuntu/data/NGS/Prunus_persica/Prunus_persica_v2.0.a1.primaryTrs.gff3 \
 --sjdbOverhang 150 \
 --sjdbGTFfeatureExon  CDS \
 --sjdbGTFtagExonParentGene Name

# --sjdbGTFfeatureExon  CDS --sjdbGTFtagExonParentGene Name 对应type和attribute里面值  --sjdbGTFfile /home/ubuntu/data/NGS/Prunus_persica/Prunus_persica_v2.0.a1.primaryTrs.gff3 需要不压缩的文件
```

+ map
```bash
STAR --runThreadN 4 --genomeDir /home/ubuntu/data/NGS/Prunus_persica \
 --readFilesIn  /home/ubuntu/data/NGS/Prunus_persica/rawdata/SRR8596341_1.clean.fastq.gz \
  /home/ubuntu/data/NGS/Prunus_persica/rawdata/SRR8596341_2.clean.fastq.gz \
 --outSAMtype BAM SortedByCoordinate \
 --readFilesCommand  zcat \
 --outFileNamePrefix /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus_STAR
#  --readFilesIn  /home/ubuntu/data/NGS/Prunus_persica/rawdata/SRR8596341_1.clean.fastq.gz /home/ubuntu/data/NGS/Prunus_persica/rawdata/SRR8596341_2.clean.fastq.gz 如果传入的是gz 需要用--readFilesCommand  zcat 

```

### FeatureCounts

gff 3

```gff
##gff-version 3
##annot-version v2.1
##species Prunus persica
Pp01	JGI_gene	gene	40340	51496	.	+	.	ID=Prupe.1G000100_v2.0.a1;Name=Prupe.1G000100;ancestorIdentifier=ppa023343m.g.v1.0
Pp01	JGI_gene	mRNA	40340	51496	.	+	.	ID=Prupe.1G000100.1_v2.0.a1;Name=Prupe.1G000100.1;longest=1;ancestorIdentifier=ppa023343m.v1.0;Parent=Prupe.1G000100_v2.0.a1
Pp01	JGI_gene	five_prime_UTR	40340	40475	.	+	.	ID=Prupe.1G000100.1_v2.0.a1.five_prime_UTR.1;Parent=Prupe.1G000100.1_v2.0.a1
Pp01	JGI_gene	CDS	40476	40616	.	+	0	ID=Prupe.1G000100.1_v2.0.a1.CDS.1;Parent=Prupe.1G000100.1_v2.0.a1
Pp01	JGI_gene	CDS	40721	40830	.	+	0	ID=Prupe.1G000100.1_v2.0.a1.CDS.2;Parent=Prupe.1G000100.1_v2.0.a1
```

第一列 染色体号 第二列 数据来源 **第三列** type 第四列/第五列 起始终止 第六列 分数 第七列 正负链 第八列 质量 **第九列** attribute

```bash
featureCounts -T 4 \
              -p -t CDS \
              -g ID \
              -a /home/ubuntu/data/NGS/Prunus_persica/Prunus_persica_v2.0.a1.primaryTrs.gff3.gz\
              -o /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/all.id.txt\
            /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sorted.bam
# -g 后面要接attribute 后面的一个选项
# -t 后续要
```

**Output**

```txt
# Program:featureCounts v2.0.0; Command:"featureCounts" "-T" "4" "-p" "-t" "CDS" "-g" "ID" "-a" "/home/ubuntu/data/NGS/Prunus_persica/Prunus_persica_v2.0.a1.primaryTrs.gff3.gz" "-o" "/home/ubuntu/data/NGS/Prunus_persica/RNA_seq/all.id.txt" "/home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sorted.bam" 
Geneid	Chr	Start	End	Strand	Length	/home/ubuntu/data/NGS/Prunus_persica/RNA_seq/Prunus.sorted.bam
Prupe.1G000100.1_v2.0.a1.CDS.1	Pp01	40476	40616	+	141	5
```

最后一列 代表了map上的数量

### Result

STAR 有60%的mapping rate hisat 有58.9%的mapping rate

## L1正则化pytorch

教程 https://blog.csdn.net/guyuealian/article/details/88426648

对于每一层的权重都求1范数

```py
torch.norm(w,p = 1)
```


# 2022-6-5

## PLAN
+ **文献阅读**

## STAR

数学原理 SA (Suffix Array)

**优点** 即通过SA算法实现的 相比于BWT算法

+ 可以进行剪接mRNA 
+ 可以容忍错配

# 2022-6-6

## PLAN
+ **学习pytorch 范数操作**
+ **scRNA-seq 理论学习**

## Single Cell

### Droplet-based RNA-seq

+ 对于tissue，首先对细胞进行消化
+ 消化后每一个细胞和一个磁珠组合成一个drop
+ 每个drop内部对single cell打上DNA barcode(Cell) and UMI(transcript)

Mapping 用STARSolo 可以兼顾DNA barcode 和 UMI

### UMI

Unique molecular identifier (UMI): 转录本来源
The UMI will be used to collapse PCR duplicates

![umi](https://pic4.zhimg.com/80/v2-cf515910e0fcf32bb666c75f6cd9b313_720w.jpg)

UMI是一段12nt的核苷酸序列（序列空间100万），但与Barcode序列不同的是，一个Gel Beads中UMI序列是不同的。UMI序列的空间很大，远多于需要检测的原始细胞的mRNA数量，(即使一种mRNA有多条，也是达不到UMI的序列空间的)。所以每一条mRNA都会带上一个独特的UMI。

UMI的作用是绝对定量，因为每个mRNA的扩增效率是不一样的，即使两个初始的mRNA表达量一致，在多轮扩增之后，我们也会误认为他们差异表达。UMI通过初始的标记，让我们可以统计扩增后UMI的种类就可以知道原始的表达量了。

![10x](https://img-blog.csdnimg.cn/fa6e7d158f454da6ba0584da156884de.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAR2xhbmNlclo=,size_20,color_FFFFFF,t_70,g_se,x_16)

**例子**

![UMI_example](https://pic1.zhimg.com/80/v2-fa134b90d45079eebab1fad18aebd3b8_720w.jpg)

图中的Gene 1定量为2，Gene 2定量为3，Gene 3,000定量为1。

## 范数计算

```py
import torch
# import torch.tensor as tensor
 
a = torch.rand((2,3))  #建立tensor
a2 = torch.norm(a)      #默认求2范数
a1 = torch.norm(torch.abs(a)-1,p=0.5)  #指定求1范数
 
print(a)
print(a2)
print(a1)

# tensor([[0.4072, 0.8210, 0.6538],
#         [0.8089, 0.0198, 0.9845]])
# tensor(1.7004)
# tensor(11.1093)
```  

## GWAS

GWAS 效应值 以及 eQTL效应值

$$ y=ax+ b$$

效应值为a的估计值

# 2022-6-7

## PLAN
+ **西语学习开头**
+ **Journal Club讲稿整理**
+ **探索转座子pipeline**

## journal club

首先介绍技术以及材料背景。技术背景包括单细胞测序以及群体遗传方法。

材料为pbmc由于易制备单细胞因此采用。

介绍全基因组测序方法，群体遗传法，孟德尓随机化。


## 转座子$F_st$

处理方法

+ 利用vcftools 限定区域
+ 利用Dnasp计算 输入fasta

# 2022-6-8

## PLAN
+ **西语语音学习**
+ **文献阅读**
+ **journal club ppt**
+ **单细胞深度学习模型**

## 整合单细胞和bulk 转录组

有单细胞转录组作为参考，利用采样和调整推断各个分布。参考的作用为第一步大致确定组织样本基因分布


## 单细胞深度学习模型

L2正则的weight_decay 改变了，目前修改回1e-5


# 2022-6-9

## PLAN

+ **文章投稿**
+ **单细胞深度学习模型训练**
+ **TF调控XSL课程**
+ **西班牙语语音**

## 孟德尔随机化

工具变量简单来说就是，一个与X相关，但与被忽略的混淆因素以及Y不相关的变量。在经济学研究中工具变量可以是政策改革，自然灾害等等，而在遗传学中，这个变量就是基因。

如果一个基因变异Z 是某个暴露因素X的因果变量，**并且对结果Y没有直接因果关系**(ALDH2变异不会直接导致食道癌)，那么这个基因变异Z与结果Y的关联，只能通过X对Y的因果关系而被观察到 **X->Y**

### Example

在日本人中，ALDH2基因的常见遗传突变影响酒精的加工，导致致癌副产物乙醛的过量产生，以及恶心和头痛。我们可以使用这种遗传变异作为工具变量来评估饮酒与食道癌之间的因果关系。在这里，饮酒是暴露而食道癌是结局。由于吸烟是食道癌的另一个危险因素，所以酒精和吸烟之间的紧密联系使传统流行病学研究得到的因果关系大打折扣。具有两个拷贝的ALDH2多态性的个体由于短期症状的严重性而倾向于避免饮酒，他们患食道癌的风险是没有突变的人的三分之一。该突变单拷贝的携带者仅表现出对酒精的轻度不耐受，他们仍然可以喝酒，但是他们不能有效地加工酒精，并且增加了乙醛的暴露量。与没有突变的等位基因携带者相比，携带突变的等位基因的人患食道癌的风险是未患突变者的三倍，而在酗酒者的研究中则高达十二倍。这是基因与环境相互作用的一个例子（这里是基因型和酒精消耗之间的相互作用），其结论是饮酒会导致食道癌。

## TF推测

主题还是使用Chip-seq

## MSA转到VCF

```bash
snp-sites -v /home/ubuntu/data/ltr_classification/germany/random.ARNOLDY2.aln.fasta -o test
```

## NGS拼接问题

+ 为什么要拼到contig
+ 为什么不拼成全基因组

## 单细胞模型训练参数

### 1/-1

利用1/-1初始化 效果不够好 z2 恢复不好导致后续恢复也不行

+ decoder pretrain 参数 weight_decay = 0 epoch = 2000 lr = 1e-4
+ decoder finetune 参数 lr 前3000 lr=1e-3 后17000 lr=1e-4

### 直接mask 不修改权重大小

+ decoder pretrain 参数 weight_decay = 0 epoch = 2000 lr = 1e-4 （L2正则化）
+ decoder finetune 参数 lr 前3000 lr=1e-3 后17000 lr=1e-4

**结果** 恢复了一些DE 基因和转录因子

# 2022-6-11

## PLAN
+ **单细胞模型**
+ **西班牙语语音**
  
# 2022-6-20

## PLAN
+ **单细胞训练/可视化**
+ **二代测序拼接探究/基于文献(19样本)**

## 单细胞网络训练

### L1正则训练

+ decoder pretrain 参数 weight_decay = 10 epoch = 2000 lr = 1e-4 （L1正则化）
+ decoder finetune 参数 lr 前3000 lr=1e-3 后17000 lr=1e-4

### 两层网络

+ decoder pretrain 参数 weight_decay = 10 epoch = 2000 lr = 1e-4 （L1正则化）(两层TF->TF->TF->GENES) TF->TF的mask相同
+ decoder finetune 参数 lr 前3000 lr=1e-3 后17000 lr=1e-4

de的效果较好，basic效果较好

## 拟南芥转座子G4思路

+ 基因组大小 -> ALLPATHS-LG 拼接得到 目前已下载好软件，需要测试使用场景
+ 转座子 首先拼成contig 再做blast

![pipeline](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/gbe/5/5/10.1093_gbe_evt064/4/m_evt064f1p.jpeg?Expires=1656941469&Signature=STtZQzGp-W3hjbySchNdwm6qpAh6F2FYDozo2VoMVFDycCsBY0T7LumD7hEYaUhFV1j4mbmhq6bqyGCVhph9AT4IISc3QdKA0dr4yNTGBcayAIH5Gnua9XpwFcSd-3HZtZRwrTIzIFWlLunExzTfMtdJTExf1kwSkkcXYIkTvAJIsmKZhcysxjkS2zNb8I-clMJqAHGOd9GX5EpDnP4tWku7Wans1f4gXY1qpidDUdtAmtwshGi5eaJ6yABPsuugAdF66r4dLIg7YzZth83cqf5MavUmB1zGTT3EcPUQqFQwjZKoXbOsHD2FFSshJ6HqtYgDbULP~-WvymMP0EyPzA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

+ G4 利用正则表达式
+ 甲基化 contig拼接直接利用甲基化原始数据

# 2022-6-21

## PLAN
+ **单细胞结果分析**
+ **西班牙语学习**
+ **cover letter overview**

## Cover letter

### Reviewer #3
General comments
This paper studies the influence of nitrogen concentration on microbial communities, focusing on co-mobilization of non-motile bacteria with motile ones. In my opinion, the topic of the paper is very innovative: bacterial hitchhiking is a very interesting and investigation, and at the moment, not many papers are dedicated to study it. Then, the results of the paper are of a good quality to be published in STOTEN, but many details should be improved before publication.
The introduction should be revised. The content is correct but the organisation of the information is poor, leading to the repetition of the same arguments all over the text. The description of the experimental setup in materials and methods is not complete. Only by reading the abstract I could figure out how the experiment was carried out. As probably many readers, I am not familiarized with metabolomic analysis and the statistical analyses carried out, and considering the description included, for me it was too difficult to follow what you have done and how. Materials and methods should be more specific (for example, OTUs are not specified, but only included in Figures). Generally, I found the results section too long and difficult to follow, and the discussion is not concise and sometimes is repeating the results, making the paper very heavy to read.
Therefore, I recommend to reject the paper and resubmit after careful revisions.
Specific comments
Highlights
Highlight 1: soil nitrogen concentrations…
Highlight 2: revise grammar. Delete "Show that the…"
Highlight 3: revise grammar. "the results revealed…". A subject is missing.
Graphical Abstract
Improve image quality
Abstract
Line 6 delete force
Line 7 % of what??
Line 8-9 This sentence is not understandable; what do you mean by ambient soils?
Line 12 correct Bacillus (same in lines 18-19)
Line 30 Hitchhiking is only named here, while you state that plant growth is affected by this mechanism. You should revise the abstract, since it is very difficult to catch what are you investigating. State clearly your objectives, your hypothesis, your experimentation, and what you obtained.
Introduction
Line 60 indicate that N treatments are applied in agricultural practices.
Line 62 which kind of responses? This statement is a bit vague.
Line 64 it is not clear what you mean by "relative functional level of motility"
Line 66 This statement may need a reference.
Line 67 properties of soil
Line 86-90 this paragraph is a bit unclear. Please revise.
Line 93 "inhibition" might not be the correct word here. What about antagonism?
Line 99 Replace "here" by the authors of a paper (I understand you are citing a paper results?)
Line 108 What do you mean by "mechanism behind"?
Line 109/110 delete " 's"
Line 115 groups of what?
Line 108-130 This paragraph is a bit long. You don't have to describe everything you have done so specifically. Be concise. Specify the aim of the investigation, the hypothesis, and what have you done to study it (briefly).
Line 126-130 This is confusing. Terms like "co-ocurrence netwok" on "symbiotic mode" are not clear.

Materials and methods
Line 141 "the soil is yellow" is really vague. Soil is "yellow" because of its inorganic components. Be more specific or just delete it if it not significant for the discussion.
Line 141 Specify the pH if possible. It will be accurate to specify also the taxonomy of the soil used.
Line 148 "of approximately" something is missing.
Line 149 I don't think Figure 1 is necessary if the description of the experimental setup is sufficiently clear in the text
Line 150 This sentence is unclear. Pot or plot?
Lines 150-162 You don't say that the soil where plants were planted was sterile (how have you sterilized it), but this was stated in the abstract. The experimental setup should be more clearly described.
Line 155 Explain how did you do the selection of the seedling
Line 156 specify
Line 157 Still, the text is not clear about if it is an on-site experiment in plots (line 142) or pots. If it is pots?
Line 158/161 what do you mean by region?
Line 159 You should describe better how this membrane was installed.
Line 165 In results you talk about bulk and rhizospheric soil. Specify this in materials and methods, and describe how did you sample both.
Line 195 Define OTU
Line 195 "organic amendment treatment"? The treatments are fertilizers, this is quite confusing.

Results
Line 261 It will be better to cite the Figure at the beginning of its explanation, so that the reader knows which results are you referring to.
Line 265 If here you are explaining the results of the non-filter treatment, you should clarify this.
Line 280 When spinach was planted you have a rhizosphere, so specify you are talking about non-rhizospheric soil.
Lines 275 and so on Revise the names of bacteria/fungi and write them in italics. Revise this throughout the whole text. Also, I am not sure about using the term "group" for the treatments. You may find a more scientific word, such as replicate, plots, pots, etc.
Line 323 What does the "R" in "H1-R-F" stand for?
Line 337 "when the membrane was added"
Line 346 It is not clear for me what function or functional changes mean here?
Line 354 How did you determined cell motility? This is not explained in materials and methods.

Discussion
Line 402 This a strong argument. You only studied N concentration influence, but many other factors may be implied (you may cite some studies here).
Line 509 "which is consistent"
Line 546-529 this sentence is very speculative. How do you know transported bacteria are "favourable" and they have plant growth promoting characteristics?

Conclusions
I found them too short, and missed some sentences about further investigation needed, since co-mobilization of microbes is a very innovative subject.


### Reviewer #4

The topic of the manuscript is interesting and a well-designed study would have contributed to the understanding of how changes in soil chemistry influences microbial mobility through hitchhiking. However, there are some major gaps in the study design, the description of methods and the approach taken:
(i) There are no details of the 0.45 micron filter membrane used. How did the authors ensure that there were no gaps in pieces between the filter which surrounded the soil around the plant roots? Was the filter one sheet or multiple sheets? Was there a pore size distribution? Was this a woven or non-woven filter, and what are the implications on the pore size?
(ii) There is no direct evidence of microbial hitchhiking. All conclusions are based on network analysis of the microbial community data, which is not convincing.
(iii) Because of the use of the soil which has a very complex microbial community, as is subject to changes in composition over time and due to numerous factors, the evidence of the hitchhiking phenomenon is weak. The motility is attributed to Bacillus and Gammaprotobacteriae only - how was this determined? How can one be sure whether other types of motile bacteria were not present? How can be sure that the other bacteria were indeed non-motile?

There are numerous inconsistencies in the writing. A couple of examples are:
1) Lines 56-58. The authors quote 3 studies none of which are relevant to the statement that inorganic ions impact bacterial motility due to increasing temperatures. None of the references discuss impacts of atmospheric temperature increase.
2) In Line 43 the authors state the nutrients used - none of which contain P. Yet in Line 152 they refer to pots containing P.


### Reviewer #5

In their study Liu et al. describe the effects of nitrogen fertilizer addition on plant growth as well as bulk and rhizosphere microbial communities and their (in silico predicted) functions. Two experimental series were performed: one with a membrane between bulk and rhizosphere soil allowing for exchange of dissolved matter (yet presumably not microbes) and a control in the absence of the 0.45 µm membrane. In the controls the authors find clear fertilization effects on microbial communities and functions while corresponding effects on rhizosphere microbial communities were different and reduced in membrane-separated rhizosphere communities. From their data the authors conclude that nitrogen application changed the rhizosphere microbiome and 'thereby affecting plant growth because of the influence on microbial hitchhiking' (in lines 29-30). Such conclusion is based on the assumption that nitrogen addition influences the 'migration of mobile organisms and hitchhiking' (line 112).
While changed microbial communities in different fertilization treatments may not be surprising, the presence of differing rhizosphere communities in membrane and control treatments is interesting. Although we agree with the authors that microbial dispersal (i.e. dispersal ecology) is important and maybe often overlooked in microbial ecology, the data presented do NOT allow for the conclusions of the authors. The mere presence of different communities (based on DNA and after PCR amplification) and a probably higher fraction of bacilli in controls relative to membrane treatment may also be due to other reasons than dispersal (e.g. changed nutrient status in the rhizosphere due to changed water flow regimes in membrane treatment, limited nutrient transport due to the exclusion of fungal mycelial networks (cf. Deveau et al., FEMS Microbiology Reviews, Vol. 42, 2018), limited exchange of protozoa and/or soil mesofauna (earth worms, collembolan etc.), limited access of plant roots to 'bulk soil' due to membranes etc:. Secondly: even if microbial dispersal was the reason, the authors show no evidence of hitchhiking as defined by Muok et al (as cited in the text). As insinuated in the discussion and by the title the community changes would be due to hitchhiking (i.e. co-transport with motile organism or abiotic particles due to flow events). Such assumption is highly speculative.
The technical quality (and the language) of the manuscript further should be improved. From the 'Material and methods' sections e.g. it does not become clear how the experiments were performed: for how long were the plants grown? Dimensions of the rhizosphere compartments vs. bulk soil? Were the experiments done in pots (line 150) placed in differently fertilized plots (line 142)? How and when was the soil fertilized (before or after addition of the membrane)? And importantly: how and to which degree was the soil mixed prior to membrane addition; was the control soil treated exactly the same way.
Next to omissions in the experimental description, the manuscript quality may be improved as several paragraphs seem (i) misplaced (e.g. lines 363ff to M&M; line 392ff to Introduction etc.), (ii) unclear (e.g. lines 370ff or 421ff), (iii) somewhat redundant (e.g. lines 322ff), (iv) in their statements very vague (e.g. line 352 - what does 'weakened' mean?, or simply wrong (lines 536ff - conclusions).
Some other comments:
- Figure 3: are the data presented averages of the three replicates? What about standard deviations? This is in particular important for the motile bacteria.
- Please explain statistics of Fig. 4 better.
- Line 49: as fungal impact on matter and organism transport is important, the reference given seems a bit arbitrary. You may better use the ref. given above (Deveau et al) and the refs. cited therein.
- Line 152: what is meant by 'P treatment'
- Line 155: which was the normal water content in the soil?
- Line 264 and 286: are the %-values given averages? Which are the standard deviations?
   

# 2022-6-22

## PLAN

+ **简要回答审稿意见**
+ **单细胞报告**
+ **西班牙语语音**

## Reviewer #4回答

The topic of the manuscript is interesting and a well-designed study would have contributed to the understanding of how changes in soil chemistry influences microbial mobility through hitchhiking. However, there are some major gaps in the study design, the description of methods and the approach taken:
+ (i) There are no details of the 0.45 micron filter membrane used. How did the authors ensure that there were no gaps in pieces between the filter which surrounded the soil around the plant roots? Was the filter one sheet or multiple sheets? Was there a pore size distribution? Was this a woven or non-woven filter, and what are the implications on the pore size?

新版已回答

+ (ii) There is no direct evidence of microbial hitchhiking. All conclusions are based on network analysis of the microbial community data, which is not convincing.

引用此前的文献，目前Bacillus了，文献已经报道了，**Bacillus差异很大**

+ (iii) Because of the use of the soil which has a very complex microbial community, as is subject to changes in composition over time and due to numerous factors, the evidence of the hitchhiking phenomenon is weak. The motility is attributed to Bacillus and Gammaprotobacteriae only - how was this determined? How can one be sure whether other types of motile bacteria were not present? How can be sure that the other bacteria were indeed non-motile?

文献中主要提到了Bacillus,我们主要关注的是Bacillus介导的 引用文献

There are numerous inconsistencies in the writing. A couple of examples are:
1) Lines 56-58. The authors quote 3 studies none of which are relevant to the statement that inorganic ions impact bacterial motility due to increasing temperatures. None of the references discuss impacts of atmospheric temperature increase.
2) In Line 43 the authors state the nutrients used - none of which contain P. Yet in Line 152 they refer to pots containing P.

## Reviewer #5 回答

In their study Liu et al. describe the effects of nitrogen fertilizer addition on plant growth as well as bulk and rhizosphere microbial communities and their (in silico predicted) functions. Two experimental series were performed: one with a membrane between bulk and rhizosphere soil allowing for exchange of dissolved matter (yet presumably not microbes) and a control in the absence of the 0.45 µm membrane. In the controls the authors find clear fertilization effects on microbial communities and functions while corresponding effects on rhizosphere microbial communities were different and reduced in membrane-separated rhizosphere communities. From their data the authors conclude that nitrogen application changed the rhizosphere microbiome and 'thereby affecting plant growth because of the influence on microbial hitchhiking' (in lines 29-30). Such conclusion is based on the assumption that nitrogen addition influences the 'migration of mobile organisms and hitchhiking' (line 112).


+ While changed microbial communities in different fertilization treatments may not be surprising, the presence of differing rhizosphere communities in membrane and control treatments is interesting. Although we agree with the authors that microbial dispersal (i.e. dispersal ecology) is important and maybe often overlooked in microbial ecology, the data presented do NOT allow for the conclusions of the authors. The mere presence of different communities (based on DNA and after PCR amplification) and a probably higher fraction of bacilli in controls relative to membrane treatment may also be due to other reasons than dispersal (e.g. changed nutrient status in the rhizosphere due to changed water flow regimes in membrane treatment, limited nutrient transport due to the exclusion of fungal mycelial networks (cf. Deveau et al., FEMS Microbiology Reviews, Vol. 42, 2018), limited exchange of protozoa and/or soil mesofauna (earth worms, collembolan etc.), limited access of plant roots to 'bulk soil' due to membranes etc:. Secondly: even if microbial dispersal was the reason, the authors show no evidence of hitchhiking as defined by Muok et al (as cited in the text). As insinuated in the discussion and by the title the community changes would be due to hitchhiking (i.e. co-transport with motile organism or abiotic particles due to flow events). Such assumption is highly speculative.

补充了没有植物的实验组作为对照

+ The technical quality (and the language) of the manuscript further should be improved. From the 'Material and methods' sections e.g. it does not become clear how the experiments were performed: for how long were the plants grown? Dimensions of the rhizosphere compartments vs. bulk soil? Were the experiments done in pots (line 150) placed in differently fertilized plots (line 142)? How and when was the soil fertilized (before or after addition of the membrane)? And importantly: how and to which degree was the soil mixed prior to membrane addition; was the control soil treated exactly the same way.

修改在新版中

Next to omissions in the experimental description, the manuscript quality may be improved as several paragraphs seem (i) misplaced (e.g. lines 363ff to M&M; line 392ff to Introduction etc.), (ii) unclear (e.g. lines 370ff or 421ff), (iii) somewhat redundant (e.g. lines 322ff), (iv) in their statements very vague (e.g. line 352 - what does 'weakened' mean?, or simply wrong (lines 536ff - conclusions).
Some other comments:
- Figure 3: are the data presented averages of the three replicates? What about standard deviations? This is in particular important for the motile bacteria.
- Please explain statistics of Fig. 4 better.
- Line 49: as fungal impact on matter and organism transport is important, the reference given seems a bit arbitrary. You may better use the ref. given above (Deveau et al) and the refs. cited therein.
- Line 152: what is meant by 'P treatment'
- Line 155: which was the normal water content in the soil?
- Line 264 and 286: are the %-values given averages? Which are the standard deviations?