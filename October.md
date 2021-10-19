- [2021-10-1](#2021-10-1)
  - [PLAN](#plan)
- [2021-10-2](#2021-10-2)
  - [PLAN](#plan-1)
  - [Rafael D'Andrea LAB](#rafael-dandrea-lab)
- [2021-10-3](#2021-10-3)
  - [PLAN](#plan-2)
  - [TE文章](#te文章)
    - [SNP工具](#snp工具)
  - [PLSY假期工作](#plsy假期工作)
    - [处理有问题的bed文件](#处理有问题的bed文件)
    - [已有结果全部整理到相应文件夹](#已有结果全部整理到相应文件夹)
    - [复现其他十八个拟南芥类群](#复现其他十八个拟南芥类群)
- [2021-10-5](#2021-10-5)
  - [PLAN](#plan-3)
  - [UCD lab](#ucd-lab)
- [2021-10-7](#2021-10-7)
  - [PLAN](#plan-4)
- [2021-10-8](#2021-10-8)
  - [处理bed文件更新](#处理bed文件更新)
- [2021-10-10](#2021-10-10)
- [2021-10-11](#2021-10-11)
  - [Quadron 脚本代码](#quadron-脚本代码)
- [2021-10-12](#2021-10-12)
- [2021-10-13](#2021-10-13)
- [2021-10-18 A day back to work](#2021-10-18-a-day-back-to-work)
  - [PLAN](#plan-5)
  - [G4论文关系](#g4论文关系)
    - [Quadruplex-forming sequences occupy discrete regions inside plant LTR retrotransposons](#quadruplex-forming-sequences-occupy-discrete-regions-inside-plant-ltr-retrotransposons)
  - [单细胞文章](#单细胞文章)
    - [VEGA](#vega)
- [2021-10-19](#2021-10-19)
  - [PLAN](#plan-6)
# 2021-10-1
## PLAN
+ **GRE阅读3填空3**
+ **阅读因果推论论文**

# 2021-10-2
## PLAN
+ **GRE套题1**
+ **Rafael D'Andrea LAB 研究总结**
+ **csbj文章投完**

## Rafael D'Andrea LAB
Our model simulates stochastic niche assembly (Tilman, 2004) under external immigration. Coexistence in this context means the sustained presence of multiple species in a focal community for a long period of time (in our case, hundreds of thousands of years), with the forces that tend to reduce species richness—competitive exclusion and demographic stochasticity—being offset by the stabilizing effects of niche differentiation and immigration.

**stochastic cellular automata**

# 2021-10-3
## PLAN
+ GRE阅读3填空3
+ overview 群体遗传
+ TE文章设计任务安排

## TE文章
### SNP工具 
snippy

## PLSY假期工作
### 处理有问题的bed文件
```bash
awk '{$5="";$6="";$7="";print $0}'  chr1.TE.LTR.bed
```
### 已有结果全部整理到相应文件夹
+ An-1
  + chr1
    + TE
      + 全部TE的bed
      + LTR bed
      + TIR bed
      + HEITON bed
    + g4 bed
    + intersect
      + 四种类型
    + 上游bed文件
      + 四种类型
    + 下游bed文件
      + 四种类型
    + fasta文件chr1
  + chr2
  + chr3
  + chr4
  + chr5

### 复现其他十八个拟南芥类群
网站 https://1001genomes.org/data/MPIPZ/MPIPZJiao2020
下载命令
```bash
wget https://1001genomes.org/data/MPIPZ/MPIPZJiao2020/releases/current/strains/C24/C24.chr.all.v2.0.fasta.gz
```

解压缩命令
```
gzip -d C24.gz(gz文件全称)
```
+ 尽量写一个脚本 输入 (chr1.fasta 全部转座子 bed文件) python的
+ 主要关注 18个类群的TE和G4交集
+ 统计出三种转座子的长度 做出直方图
+ 统计出G4的分数，做出直方图 用ggplot2

# 2021-10-5
## PLAN
+ **Gre套题1**
+ **Gre填空5阅读5**
+ Gre数学一套

## UCD lab
https://qtl.rocks/

# 2021-10-7
## PLAN
+ **Gre套题1**
+ **Gre填空5阅读5**
+ Gre数学一套

# 2021-10-8
+ **Gre套题1**
+ **Gre填空5阅读5**

## 处理bed文件更新
```bash
perl -p -i -e "s/shan/hua/g" ./lishan.txt 
# 将当前文件夹下lishan.txt和lishan.txt.bak中的“shan”都替换为“hua”
```
**很多时候生成G4文件时，标注的染色体总会持续为chr1，第一步应该对染色体标注进行改变**

脚本中命令
```python
os.system("perl -p -i -e \"s/chr1/"+chr_fasta[0:4]+"/g\" "+ g4_bed)
```

# 2021-10-10
+ **Gre套题1**
+ **Gre阅读8**


# 2021-10-11
+ **Gre套题1**
+ **Gre阅读7**

## Quadron 脚本代码
```R
#G4-Quadron.R
args <- commandArgs (trailingOnly =TRUE)
print ( "NOTE: Loading Quadron core. . . ", quote=FALSE)
load ( "/home/ubuntu/Arabidopsis/Quadron/Quadron.lib" )
print (args)
Quadron ( FastaFile= args [1],OutFile= args [2],
        nCPU=as.numeric ( args [3] ),
        SeqPartitionBy = 100000 )
```
shell 运行代码
```bash
Rscript G4-Quadron.R /home/ubuntu/Arabidopsis/Arabidopsis_sequence/An-1/chr3.fa /home/ubuntu/Arabidopsis/Arabidopsis_sequence/An-1/chr3.g4.out.txt 2
```
后续有三个参数分别对应arg 1 2 3 即 输入文件 输出文件 cpu数量

# 2021-10-12
+ **Gre阅读7**

# 2021-10-13
+ **Gre数学1**
+ **Gre填空7**


# 2021-10-18 A day back to work
## PLAN
+ **安装论文汇总器**
+ **overview 单细胞测序**
+ **RDA overview以及如何解释结果**
+ **分子生物学作业以及学习**

## G4论文关系
### Quadruplex-forming sequences occupy discrete regions inside plant LTR retrotransposons
主要探究G4与LTR的关系
+ G4在LTR的分布位点/相对和绝对 
![G4在LTR的分布位点/相对和绝对](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/nar/42/2/10.1093_nar_gkt893/3/m_gkt893f1p.jpeg?Expires=1637540582&Signature=QfCBtXTloHlQNUu-QZKD4U~a0QiP7LXu0EE488~Wpn6wZaJfC93kRpudmHvqz75ZpNMgQuf5m~GYWX6SkRfbcLEvs38a~-9nZEkxwEdoAJ9YQogx3~AiMcq5Ad-JdeckuJhsSMcuyxSwpb690-gfFHXh-s0S20f6WhorwfIAmtAtehBUMeKYGQXgJzIpCcd9WbbfjKsE6eM8P6wRetR1a1AOfvGhQaiACkA9Q1JiOD7GeQhhxhQkJB-O44u~~WYy6AtFhrfzg6QHNlzT6D0U~wTuswrTQlIrqm5Vj6mgGnlDRcEoiiE6ovVdeiTfk6B860IcroLJE7qD3drpy0A0jQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

+ G4之间的距离
+ PQS in relation to predicted transcription start site G4与翻译起始位点的距离
![G4在LTR的距离](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/nar/42/2/10.1093_nar_gkt893/3/m_gkt893f4p.jpeg?Expires=1637540582&Signature=Q6lsB4sMKzTO6UlQmBZfKBOnbIxYUvrn-eZiTvZ2PLJhGZdF9Eod8glboIxVEvK~s3H-5ySSWdZlHJtquxhFVP79BPCbNKnSC7bQ7x7ejj5bw5RDiIgMgTDlmcKRDMMj3e4gOCl16tAcAT3QM2ZhngLAhYSICwITMPhoU0V-gVSJ4PLeHsgGsQM86eghuoIBu9LMOVJBZuCGH7OsUcpbBgRsVgS452rM~hLMeey2NbRrf-pkBP5g4ZWO1Yvra5stEx8heGbyF2lzEqu8nsB4JAjcMnB8dR60uKbNpXsZBI-WHjI4PHxUQaPomgpNkz6DnKl9M9ZJDslijHWPpIJlOg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)


+ **讨论**

启动子上游和下游 PQS 的丰度表明四链体可能分别在 RNA 的转录起始和延伸中发挥作用。负链中启动子上游的四链体 DNA 的定位可以通过将转录区域维持在单链构象中来刺激逆转录转座子的转录

我们推测 LTRs 中四链体的存在可能与这种失活机制有关，可能是通过干扰甲基化过程。由于在一条链上形成的四链体理论上会使另一条链处于单链状态，因此它们可能会阻碍周围序列的甲基化，即使它们富含 CpG 和其他可甲基化的核苷酸对。

## 单细胞文章
### VEGA
目的:找到合适特征，从而进行降维后分析，好处可解释性强，相比于传统机器学习。


# 2021-10-19
## PLAN
+ 分子生物学作业+复习
+ VEGA论文阅读
+ 甲基化阅读
+ 统计PHD材料