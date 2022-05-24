- [NGS学习](#ngs%E5%AD%A6%E4%B9%A0)
  - [经典算法及其数学分析](#%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E6%95%B0%E5%AD%A6%E5%88%86%E6%9E%90)
    - [Minimap2](#minimap2)
    - [BWT算法](#bwt%E7%AE%97%E6%B3%95)
  - [DNA 重测序技术](#dna-%E9%87%8D%E6%B5%8B%E5%BA%8F%E6%8A%80%E6%9C%AF)
    - [SRA 命令 （数据下载）](#sra-%E5%91%BD%E4%BB%A4-%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)
    - [bwa 命令](#bwa-%E5%91%BD%E4%BB%A4)
    - [单端测序&双端测序](#%E5%8D%95%E7%AB%AF%E6%B5%8B%E5%BA%8F%E5%8F%8C%E7%AB%AF%E6%B5%8B%E5%BA%8F)
    - [质控](#%E8%B4%A8%E6%8E%A7)
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

### SRA 命令 （数据下载）

```bash
## 下载并index参考基因组
curl -C - -O ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz
bgzip -dc GCF_000005845.2_ASM584v2_genomic.fna.gz > E.coli_K12_MG1655.fa
samtools faidx E.coli_K12_MG1655.fa

## 下载WGS文件
/home/ubuntu/data/softwares/sratoolkit.2.11.2-ubuntu64/bin/fastq-dump --split-files SRR1770413

## fastq 文件

/home/ubuntu/data/softwares/sratoolkit.2.11.2-ubuntu64/bin/fastq-dump --split-files SRR1770413
bgzip -f SRR1770413_1.fastq
bgzip -f SRR1770413_2.fastq
```

下载压缩的双端测序数据

### bwa 命令

```bash
# bwa index -a bwtsw Col-0.fasta
bwa mem /home/ubuntu/data/NGS/Col-0.fasta /home/ubuntu/data/NGS/SRR390728.fastq > /home/ubuntu/data/NGS/result/bwa_result.sam
```
+ 建立索引
+ 进行比对

### 单端测序&双端测序

主要原因在于Illumina的二代测序仪的读长短，相对于第一代sanger测序法（约1000bp）或者跟同属于NGS的其他测序仪相比短了许多。因此illumina发展了 Paired-end的建库测序技术。同时这种技术还大大推进了基因组学数据分析的发展。
例如，依赖于Paired-end的技术，假设一个DNA片段刚好跨越了重复序列区域（下图左侧）以及独特序列区域（下图右侧）。

### 质控

**代码**

```bash
fastp \
-i SRR1770413_1.fastq.gz \
-o SRR1770413_1.clean.fastq.gz \
-I SRR1770413_2.fastq.gz \
-O SRR1770413_2.clean.fastq.gz \
-z 4 \
-f 5 -t 5 -F 5 -T 5 \
-5 -W 5 -M 20 \
-Q \
-l 50 \
-c \
-w 4
```

```markdown
# 参数说明
# 1. I/O options   即输入输出文件设置
## -i, --in1    输入read1文件名  
## -o, --out1   输出read1文件名  
## -I, --in2    输入read2文件名  
## -O, --out2   输出read2文件名  
## -z, --compression    输出压缩格式。给定一个数字1-9，调整压缩比率和效率的平衡  

# 2. adapter trimming options   过滤序列接头参数设置  
## -A, --disable_adapter_trimming    默认已开启，设置该参数则关闭  

# 3. global trimming options   剪除序列起始和末端的低质量碱基数量参数  
## -f,--trim_front1  
## -t, --trim_tail1  
## -F, --trim_front2  
## -T, --trim_tail2  

# 4. per read cutting by quality options   划窗裁剪  
## 具体的原理就是通过滑动一定长度的窗口，计算窗口内的碱基平均质量，如果过低，就直接往后全部切除。  
### 不是挖掉read中的这部分低质量序列，像切菜一样，直接从低质量区域开始把这条read后面的所有其它碱基全！部！剁！掉！否则就是在人为改变实际的基因组序列情况。  
## -5, --cut_by_quality5  
## -3, --cut_by_quality3  
## -W, --cut_window_size    设置窗口大小  
## -M, --cut_mean_quality    设置窗口碱基平均质量阈值  

# 5. quality filtering options   根据碱基质量来过滤序列  
## -Q, --disable_quality_filtering    控制是否去除低质量，默认自动去除，设置-Q关闭  
## -q, --qualified_quality_phred    设置低质量的标准，默认Q15  
## -u, --unqualified_percent_limit    低质量碱基所占百分比，默认40代表40%  

# 6. length filtering options   根据序列长度来过滤序列  
## -l, --length_required    设规定read被切除后至少需要保留的长度，如果低于该长度，会被丢掉  

# 7. base correction by overlap analysis options   通过overlap来校正碱基  
## -c, --correction    是对overlap的区域进行纠错，只适用于pairend reads  

# 8. threading options   设置线程数  
## -w, --thread 设置线程数
```

**原理**

质控分析包括

+ read各个位置的碱基质量值分布
好的结果是碱基质量值都基本都在大于30，而且波动很小。
+ 碱基的总体质量值分布
对于二代测序，最好是达到Q20的碱基要在95%以上(最差不低于90%)，Q30要求大于85%(最差也不要低于80%)。
+ read各个位置上碱基分布比例，目的是为了分析碱基的分离程度
最好平均在1%以内。
+ GC含量分布
对于人类来说，我们基因组的GC含量一般在40%左右。
+ read各位置的N含量
N在测序数据中一般是不应该出现的。
+ read是否还包含测序的接头序列
在测序之前需要构建测序文库，测序接头就是在这个时候加上的，其目的一方面是为了能够结合到flowcell上，另一方面是当有多个样本同时测序的时候能够利用接头信息进行区分。一般的WGS测序是不会测到这些接头序列的。
+ read重复率，这个是实验的扩增过程所引入的

**WGS质控与RNA-seq不同**


## RNA-Seq 技术

## 单细胞 scRNA-seq 技术

## 甲基化技术