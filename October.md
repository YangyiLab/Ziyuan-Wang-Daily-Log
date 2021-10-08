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
+ Gre套题1
+ Gre填空5阅读5

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