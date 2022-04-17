## 4.12
### 计算CpGO/E
+ 复现CpG O/E与基因组大小呈现负相关
+ 方法：The CpG observed/expected ratio was calculated by the CpG density—which is N(CpG)/N, where N(CpG) is the number of CpG dinucleotides and N is the length of the genome—divided by the expected CpG density, N(C) × N(G)/(N × N), where N(C) is the number of cytosines and N(G) is the number of guanines.
+ 代码：/home/ubuntu/Arabidopsis/Scripts/cpG.py
+ 数据集：/home/ubuntu/Arabidopsis/Scripts/CpGOE.csv
+ 疑惑：
    + N（CpG）、N（C）、N(G)：都是全基因组范围，不是转座子内的吧？
    + N（CpG）：CGG、CGA这种的算吗，还是只算CG，但只算CG的话，有些种质里没有，例如/home/ubuntu/data/Arabidopsis_sequence/7000/7000_mC.bed，全是3个碱基的

## 4.13
### 统计种质中反转录性质转座子数量
+ 代码：/home/ubuntu/Arabidopsis/Scripts/statisticLTR.py
+ 数据集：/home/ubuntu/Arabidopsis/Scripts/TE_type.csv

### 每100Kb转座子、全基因组、非转座子序列所具有的甲基化bp数
+ 代码：/home/ubuntu/Arabidopsis/Scripts/deal_TE_mc.py（后面三个函数）
+ 数据集：/home/ubuntu/Arabidopsis/Scripts/mcCompare.csv