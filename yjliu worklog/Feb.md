## 2.7
### 待办
+ 文献阅读
+ 整理思路
### 记录
+ 整理思路：图见笔记本
  + 利用失活的转座子构建进化树：失活的转座子←甲基化程度高←G4折叠←地理环境因素
  + 如何寻找失活的转座子：
    参考论文：volutionary dynamics of retrotransposons assessed by high-throughput sequencing in wild relatives of wheat. Genome Biology and Evolution
    + 目的：区分最近活跃的TE谱系和静止的TE谱系
    + 结论： 通过三个参数：ML 树拓扑结构、遗传分化(由KST衡量)、TE 拷贝之间的错配分布分析
        树 I（也包括树Ⅱ）、高KST和双峰错配分布为最近活跃
        其中高K ST，并且树木具有两个不同的物种特异性进化枝（树Ⅰ），这表明它们是从一个或几个主副本增殖的。

    + 方法：
        + ML树拓扑结构：每个分支若有两色即为混合进化枝，单色即为特异性进化枝
            意义：据此分为四种拓扑结构：全为species-specific clades、大部分为species-specific clades、大部分为mixed-species clades、全为mixed-species clades；species-specific clades越多说明该TE种群在物种间差异越大，即转座发生越多
            测量方法：RAXmlHPC
        + KST：意义：较低的K ST表明 TE 群体大多共享相似的 TE 拷贝，而较高的K ST意味着插入不同基因组中的 TE 拷贝之间的差异更大。(K ST > 0.2)
            测量方法：使用 Arlequin 3.5 版（ Excoffier et al. 2005）通过序列之间的 Tamura 和 Nei 距离估计K ST 
        + TE 拷贝之间的错配分布分析：意义：成对核苷酸差异的比例,这两个物种之一的双峰分布和在特定物种中显著较低的 τ 提供了最近在Ae 中特定家族增殖的证据。
            测量方法：使用 MEGA 版本 5 ( Tamura et al. 2011 )对物种特异性比对进行错配分布，并使用 siZer ( Chaudhui 和 Marron 1999 ) 进行统计可视化，以标记可靠峰周围斜率的显着增加或减少。作为补充，使用 Arlequin 3.5 ( Excoffier et al. 2005 ) 对物种特异性比对和包括两个物种的比对评估参数 τ（即 τ，2.5 和 97.5 分位数），表示自扩张以来的突变单位时间。

## 寻找relict
在另一端，极端的成对分歧（图 3A）出现在 26 个种质中，其中 22 个来自伊比利亚半岛，佛得角群岛、加那利群岛、西西里岛和黎巴嫩各有一个（见数据发布部分）。我们将这些加入物称为“遗物”。22 个伊比利亚遗迹与成对的非遗迹（图 3 C）之间并没有什么不同。其余四个遗物彼此分开并与所有其他加入物分开。
https://1001genomes.org/accessions.html（各分类及经纬度）

只在CG上出现的甲基化 gbM
只有C便能出现 TEM，会沉默
没有甲基化 UM
高甲基化的主要在德国

拟南芥的全基因组大小不同，原因是：与转座子相关
转座子活性和甲基化有关
G4和甲基化抑制的能力有关
环境对G4的影响（折叠）
和数据库类似的东西：全基因组大小，TE序列的数量（不分染色体），TE序列的bp数（不分染色体），活性TE家族数，TE序列/基因组序列（以bp为单位），G4数量，G4被甲基化的数量
bp：矫正G4中甲基化的bp/
1-5号染色体的大小，