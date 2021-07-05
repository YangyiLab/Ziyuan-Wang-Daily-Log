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
文献地址 https://doi.org/10.1038/s41467-020-20271-4
### Key point 3
+ Body size
+ Community assembly processes of soil organisms in paddy system
+ Distance-decay relationship and variation

# 2021-7-5
## PLAN 
+ **修改土壤微生物论文 定稿**
+ **GRE阅读3填空1**
+ 修改简历