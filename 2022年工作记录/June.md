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