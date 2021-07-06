- [2021-7-1](#2021-7-1)
  - [PLAN](#plan)
  - [VirFinder](#virfinder)
    - [主要功能](#主要功能)
    - [方法](#方法)
    - [KEGG GO NOG](#kegg-go-nog)
    - [Metaboanalyst 代谢通路富集分析](#metaboanalyst-代谢通路富集分析)
- [2021-7-2](#2021-7-2)
  - [PLAN](#plan-1)
  - [病毒基因组介绍](#病毒基因组介绍)
  - [代谢组数据](#代谢组数据)
- [2021-7-3](#2021-7-3)
  - [PLAN](#plan-2)
  - [创建GitHub Gitee Blog](#创建github-gitee-blog)
  - [中科院超算使用记录](#中科院超算使用记录)
    - [登录方法](#登录方法)
    - [使用记录](#使用记录)
  - [eggNog 数据库](#eggnog-数据库)
  - [24samples 16s分析](#24samples-16s分析)
  - [微生物土壤重建文献](#微生物土壤重建文献)
    - [Key point 3](#key-point-3)
- [2021-7-5](#2021-7-5)
  - [PLAN](#plan-3)
- [2021-7-6](#2021-7-6)
  - [PLAN](#plan-4)
  - [微生物土壤重建文献](#微生物土壤重建文献-1)
    - [文章简介](#文章简介)
    - [文章计算解读](#文章计算解读)
    - [NVI 介绍](#nvi-介绍)
  - [VIGOR, an annotation program for small viral genomes](#vigor-an-annotation-program-for-small-viral-genomes)
    - [Feature of viral genes](#feature-of-viral-genes)
    - [key point](#key-point)
# 2021-7-1
## PLAN
+ **VirFinder 文献阅读**
+ **KEGG丰度图**
+ **16S结论总结(桉树)**
+ **Metaboanalyst 代谢通路使用说明总结**
+ **GRE阅读2填空3**

## VirFinder
### 主要功能
将contig or read 宏基因组测序后将病毒和宿主的基因组进行分箱Bin
### 方法
+ Use t statistic to test for each word w if the mean word frequency in viral sequences was significantly different from that in host sequences.
+ Exclude the word whose p value is the highest
+ Use Logistic Regression and Lasso Regression to train and predict

### KEGG GO NOG
可以取平均作图，可以去异常值，注意交换数据，做出热图，并找到显著性检验功能，写到论文中。

### Metaboanalyst 代谢通路富集分析
可以直接输入name并寻求对应和作图

# 2021-7-2
## PLAN
+ **GRE阅读2填空3**
+ **修改根际微生物图**
+ **整理代谢组数据**
+ **搜索病毒基因组数据**

## 病毒基因组介绍
+ 与真核生物相比，基因组size较小
+ 形态多样，单链双链线性
+ 基因重叠现象明显
    + 一个基因在另一个基因中
    + 两个基因相交
    + 两个基因之间只有一个nt相交、即一个基因的start condo和零一个基因的stop condo只有一个密码子相交
+ 大部分区域都编码蛋白质
+ Phage的特点
    + 连续无内含子 而真核生物病毒含有内含子
    + GC content 34%
+ introns Some fragments of introns are introns for some genes, however, when considering the overlaps the regions may become exons

## 代谢组数据
进入 http://42.193.18.116:8080/MetaboAnalyst/
+ 对于通路分析对应网站: Enrichment Analysis
    + 首先需要整理数据，数据表头需要KEGG_ID，每一个sample对应的分组，一个分组至少要有三个重复
    + 对于富集分析只能进行两组间比较
    
# 2021-7-3
## PLAN
+ **整理病毒基因组特征文献并且下载文献管理软件Mendeley**
+ **在github创建个人日志管理系统**
+ **GRE 阅读一套题**
+ **尝试超算使用**
+ 阅读土壤物种重建文章
## 创建GitHub Gitee Blog
+ Github网址 https://github.com/YangyiLab/Daily-Worklog
+ 码云 令牌 4ffb557087446dc944151a59d1beb9ba
  项目地址: https://gitee.com/YangyiLab/Daily-Worklog

## 中科院超算使用记录
### 登录方法
请点击链接 https://cloud.blsc.cn/ 下载客户端,  使用下列北京超级云计算中心账号登陆；
北京超级云计算中心登录账号 | 用户名：princeyuansql@gmail.com 密码：a182d9
### 使用记录
+ 可以使用基本文件操作
+ 可以使用基本账号密码登录(需要单独申请)
+ 跑代码比较合适(但需要单独编写任务提交命令sbatch等)
**需要编写sbatch脚本和.job脚本**

## eggNog 数据库
+ 导入两张表
+ 做自然连接/Where 到处一张表 Orthologous_id	Orthologous_id_description	Class
+ Group by 命令 统计下 Class的数量
+ 导出到excel表中，按照class排序，找到和nitrogen相关的基因

## 24samples 16s分析
+ aplha 多样性
+ beta 多样性 (需要利用commandLine函数辅助)
+ 相对丰度图

## 微生物土壤重建文献
<span id="BS-D1">文献地址 https://doi.org/10.1038/s41467-020-20271-4 </span>
### Key point 3
+ Body size
+ Community assembly processes of soil organisms in paddy system
+ Distance-decay relationship and variation

后续地址 **[section-2](#微生物土壤重建文献-1)**

# 2021-7-5
## PLAN 
+ **修改土壤微生物论文 定稿**
+ **GRE阅读3填空1**
+ 修改简历

# 2021-7-6
## PLAN 
+ **修改土壤微生物论文图片引用**
+ **GRE阅读3填空1**
+ 病毒微生物综述阅读
+ **土壤重建论文继续阅读**

## 微生物土壤重建文献
上文地址: **[section-1](#微生物土壤重建文献)**
### 文章简介
本文主要通过探究体型大小揭示生物体型大小在生物群落构建机制中的作用>
生物生物体型与**丰富度**、**扩散速率**和**生态位宽度**负相关，从而影响了生物群落的周转率。

### 文章计算解读
+ 丰富度 alpha多样性
+ 扩散速率 DDR Distance-decay relationship
  比较方式包括Similarity BC ~ 地理距离Distance
  $\log S =a+b \log D$
  $S_0$与body size关系 以及$b$斜率指标和body size关系

$S_0$计算方法，文章中选用了45个采样点，每个采样点进行分箱，分出不同的class级别物种，此时拥有了class级别物种的body size，在更加细分，可以计算出d距离两地的群落的BC距离，拟合出a和b，当D=1km时计算出的$S_0$即initial similarity,达到$S_0/2$时对应的$d_H$即halving distance,上述计算出$b,S_0,d_H$对应bcd图。a图扩展到了整个群落层面。

+ Assembly 计算 使用三个指标 NVD OMI 扩散指数

### NVI 介绍
介绍论文地址 https://onlinelibrary.wiley.com/doi/full/10.1111/oik.02803?saml_referrer
+ 提出该方法论文的定义为，两个群落获得OTU表后，进行bootstrap抽样并计算BC距离$E(\beta_{null})$与两样本之间的BC距离$\beta_{obs}$,若接近于0，则表示随机性强，若接近于+1或者-1 则证明确定性强
+ 本文使用方法为对于999个随机生成的样本的每个OTU，之后对于28个class level的微生物分类，计算class级NDV 
采样地点20个图新，在实验地块，通过这些数据建库bootstrap
## VIGOR, an annotation program for small viral genomes
### Feature of viral genes
Although most viral genomes are relatively small compared to eukaryotic and prokaryotic genomes, the gene structure of viral genomes can be complex. For example, introns, alternative splicing, overlapping genes, and ribosomal slippage exist in many viral genomes. Thus an all purpose gene finder cannot be easily adapted for gene prediction across all virus families. However, if the genome scaffold and the gene features of a viral genome are well understood, a similarity-based gene prediction approach based on the curated gene repertoire for a specific virus genus with attention to particular recognition features, such as, splice sites and mature peptide cleavage sites can be adapted, and perform better than an ab initio gene finder.
### key point
内含子、可变剪接、重叠基因和核糖体滑移
introns, alternative splicing, overlapping genes, and ribosomal slippage