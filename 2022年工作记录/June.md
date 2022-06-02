# 2022-6-2

## PLAN

+ **RNA-seq pipeline overview**
+ **VCFtools 脚本完善**
+ 正则化学习

## RNA-seq流程

数据依赖 参考基因组fasta文件 **基因注释gff3**(与WGS不同)

步骤
+ 下载数据利用sratools
+ 质控
+ mapping 可以使用bwa botie或者hisat2
+ 定量->质控->构建表达矩阵

### illumina dragen

![GREGEN](https://d3f3vsz1y2oz5i.cloudfront.net/9662654/screenshots/443e175f-76dd-4743-8e93-9fd82a6c2db7_800_500.png)

illumina 开发的rna-seq pipeline

集成到测序仪

### software

bwa 因为BW transformation无法做到剪接

+ minimap2 splicing
+ STAR ? 需要去看原始文件
+ kallisto **psudo alignment**

