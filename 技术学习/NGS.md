- [NGS学习](#ngs%E5%AD%A6%E4%B9%A0)
  - [经典算法及其数学分析](#%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E6%95%B0%E5%AD%A6%E5%88%86%E6%9E%90)
    - [Minimap2](#minimap2)
    - [BWT算法](#bwt%E7%AE%97%E6%B3%95)
  - [DNA 重测序技术](#dna-%E9%87%8D%E6%B5%8B%E5%BA%8F%E6%8A%80%E6%9C%AF)
    - [SRA 命令](#sra-%E5%91%BD%E4%BB%A4)
    - [bwa 命令](#bwa-%E5%91%BD%E4%BB%A4)
    - [单端测序&双端测序](#%E5%8D%95%E7%AB%AF%E6%B5%8B%E5%BA%8F%E5%8F%8C%E7%AB%AF%E6%B5%8B%E5%BA%8F)
  - [RNA-Seq 技术](#rna-seq-%E6%8A%80%E6%9C%AF)
  - [单细胞 scRNA-seq 技术](#%E5%8D%95%E7%BB%86%E8%83%9E-scrna-seq-%E6%8A%80%E6%9C%AF)
  - [甲基化技术](#%E7%94%B2%E5%9F%BA%E5%8C%96%E6%8A%80%E6%9C%AF)

# NGS学习

本栏目主要记录学习NGS 原始数据处理技术的相关心得体会

## 经典算法及其数学分析

### Minimap2

minimizer 找到{x,y,w} x 为ref 位置, y为query 位置 w为interval
找到一系列的anchers
最后可以确定位置

+ 首先，将基因组序列的minimizer存储在哈希表中（minimizer指一段序列内最小哈希值的种子）；
+ 然后，对于每一条待比对序列，找到待比对序列所有的minimizer，通过哈希表找出其在基因组中的位置，并利用chaining算法寻找待比对区域；
+ 最后，将非种子区域用动态规划算法进行比对得到比对结果。

### BWT算法

## DNA 重测序技术

### SRA 命令

```bash
fastq-dump --split-3 --defline-qual '+' --defline-seq '@\$ac-\$si/\$ri'  SRR1945478 --gzip
```

下载压缩的双端测序数据

### bwa 命令

```bash
# bwa index -a bwtsw Col-0.fasta
bwa mem /home/ubuntu/data/NGS/Col-0.fasta /home/ubuntu/data/NGS/Col-0.fasta /home/ubuntu/data/NGS/SRR390728.fastq > /home/ubuntu/data/NGS/result/bwa_result.sam
```
+ 建立索引
+ 进行比对

### 单端测序&双端测序

主要原因在于Illumina的二代测序仪的读长短，相对于第一代sanger测序法（约1000bp）或者跟同属于NGS的其他测序仪相比短了许多。因此illumina发展了 Paired-end的建库测序技术。同时这种技术还大大推进了基因组学数据分析的发展。
例如，依赖于Paired-end的技术，假设一个DNA片段刚好跨越了重复序列区域（下图左侧）以及独特序列区域（下图右侧）。

## RNA-Seq 技术

## 单细胞 scRNA-seq 技术

## 甲基化技术