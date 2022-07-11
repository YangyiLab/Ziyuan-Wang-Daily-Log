- [NGS学习](#ngs%E5%AD%A6%E4%B9%A0)
  - [经典算法及其数学分析](#%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E6%95%B0%E5%AD%A6%E5%88%86%E6%9E%90)
    - [Minimap2](#minimap2)
    - [BWT算法](#bwt%E7%AE%97%E6%B3%95)
  - [DNA 重测序技术](#dna-%E9%87%8D%E6%B5%8B%E5%BA%8F%E6%8A%80%E6%9C%AF)
    - [SRA 命令 （数据下载）](#sra-%E5%91%BD%E4%BB%A4-%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)
    - [bwa 命令](#bwa-%E5%91%BD%E4%BB%A4)
    - [单端测序&双端测序](#%E5%8D%95%E7%AB%AF%E6%B5%8B%E5%BA%8F%E5%8F%8C%E7%AB%AF%E6%B5%8B%E5%BA%8F)
    - [质控](#%E8%B4%A8%E6%8E%A7)
    - [GATK](#gatk)
    - [VCF 文件](#vcf-%E6%96%87%E4%BB%B6)
    - [染色体结构变异](#%E6%9F%93%E8%89%B2%E4%BD%93%E7%BB%93%E6%9E%84%E5%8F%98%E5%BC%82)
    - [$F_{st}$计算(基于vcftools)](#fst%E8%AE%A1%E7%AE%97%E5%9F%BA%E4%BA%8Evcftools)
    - [将·替换为0|0](#%E5%B0%86%E6%9B%BF%E6%8D%A2%E4%B8%BA00)
    - [NUCLEOTIDE DIVERGENCE(基于vcftools)](#nucleotide-divergence%E5%9F%BA%E4%BA%8Evcftools)
  - [RNA-Seq 技术](#rna-seq-%E6%8A%80%E6%9C%AF)
    - [RNA-seq流程](#rna-seq%E6%B5%81%E7%A8%8B)
    - [illumina dragen](#illumina-dragen)
    - [software](#software)
    - [Alignment](#alignment)
    - [**FeatureCounts** 基于gff3 文件统计每一个CDS的hit数量](#featurecounts-%E5%9F%BA%E4%BA%8Egff3-%E6%96%87%E4%BB%B6%E7%BB%9F%E8%AE%A1%E6%AF%8F%E4%B8%80%E4%B8%AAcds%E7%9A%84hit%E6%95%B0%E9%87%8F)
  - [单细胞 scRNA-seq 技术](#%E5%8D%95%E7%BB%86%E8%83%9E-scrna-seq-%E6%8A%80%E6%9C%AF)
    - [读GSE中单细胞数据](#%E8%AF%BBgse%E4%B8%AD%E5%8D%95%E7%BB%86%E8%83%9E%E6%95%B0%E6%8D%AE)
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


**代码**

```bash
# 建立索引
minimap2 -d Col-0.min  Col-0.fasta
# 比对
minimap2 -x sr -a  Col-0.min  SRR390728_1.fastq.gz \
                        SRR390728_2.fastq.gz > test.sam
```

注意的问题是需要调整参数 sr (short read)

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

### GATK

**GATK**全称是The Genome Analysis Toolkit，是Broad Institute（The Broad Institute, formerly the Broad Institute of MIT and Harvard, evolved from a decade of research collaborations among MIT and Harvard scientists.）开发的用于二代重测序数据分析的一款软件，里面包含了很多有用的工具，主要注重与变异的查找，基因分型且对于数据质量保证高度重视。

+ 为参考序列生成一个.dict文件

**代码**
```bash
gatk CreateSequenceDictionary \
-R Col-0.fasta \
-O Col-0.dict
```

+ 生成中间文件gvcf

**代码**
```bash
gatk HaplotypeCaller \
-R Col-0.fasta \
--emit-ref-confidence GVCF \
-I Ath.sorted.markdup.bam \
-O Ath.g.vcf
```

+ 通过gvcf检测变异

**代码**
```bash
gatk HaplotypeCaller \
-R Col-0.fasta \
-I Ath.sorted.markdup.bam \
-O Ath.vcf.gz
```

**通过gatk检测出变异后，可以进行过滤或进一步操作。**

### VCF 文件

*在没检测的用0/0*

**bcftools**
```bash
bcftools merge -0
```
**vcftools**
```bash
vcf-merge -R 0/0 ...
```
*过滤缺失*
```bash
vcftools --max-missing 1 ...
```

### 染色体结构变异

**breakdancer,manta**

```bash
mkdir SV && cd SV
-g -h /home/ubuntu/data/NGS/Athaliana/Ath.sam > 4Ath_requence.cfg
```

### $F_{st}$计算(基于vcftools)

出现$H_t$=0的问题，会计算出nan，可以提前做filter`vcftools --min-allels`?

### 将·替换为0|0

可以进行大量的Fst 计算后0值显著性减少，同时负数可以被看做0值 [如何解读negative
$F_{st}$ value](https://www.researchgate.net/post/What-can-be-interpreted-from-a-negative-Fst-value-005-and-a-high-P-value-06-when-measuring-pairwise-Fst-between-2-populations-mtDNA-cyt-b)

**bcftools**
```bash
bcftools merge -0
```

**vcftools**
```bash
vcf-merge -R 0/0 ...
```

### NUCLEOTIDE DIVERGENCE(基于vcftools)

利用`vcftools --site-pi`计算核酸多样性

## RNA-Seq 技术

### RNA-seq流程

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
+ STAR 基于Suffix Array算法钊MMP (最长前缀) [文章链接](https://academic.oup.com/bioinformatics/article/29/1/15/272537)
+ kallisto **psudo alignment**

### Alignment

代码
**用BWA软件没有考虑内含子剪接问题**
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

**STAR** 包括Index Map

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

### **FeatureCounts** 基于gff3 文件统计每一个CDS的hit数量

**gff3 格式**

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

```bash
cat /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/all.id.txt | cut -f1,7- > /home/ubuntu/data/NGS/Prunus_persica/RNA_seq/count.txt
```

只保留CDS还有count数

## 单细胞 scRNA-seq 技术

### 读GSE中单细胞数据

数据来源 GSE225978

tpm 已被normalization后的数据
annotation 每个样本的描述(type, name...)

```R
annotation = read.table("GSE115978_cell.annotations.csv.gz",header = TRUE, sep = ",")[,3]
tpm = read.table("GSE115978_tpm.csv.gz",sep = ",",header = TRUE)
rownames(tpm) = tpm$X
tpm$X = NULL
tpm = t(tpm)
genes = colnames(tpm)

## 甲基化技术