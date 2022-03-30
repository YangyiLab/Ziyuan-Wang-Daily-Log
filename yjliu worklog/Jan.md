- [寒假work](#寒假work)
  - [1.15](#115)
    - [理解项目](#理解项目)
    - [待做（bed、pipeline、split、replace、join）](#待做bedpipelinesplitreplacejoin)
    - [记录](#记录)
  - [1.16](#116)
    - [待做](#待做)
    - [记录（bedtools（intersect、coverage））](#记录bedtoolsintersectcoverage)
  - [1.17](#117)
    - [待做](#待做-1)
    - [记录](#记录-1)
  - [1.18](#118)
    - [待做](#待做-2)
    - [记录](#记录-2)
  - [1.20](#120)
    - [待做](#待做-3)
    - [记录](#记录-3)
  - [1.21](#121)
    - [待做](#待做-4)
    - [记录](#记录-4)
  - [1.22](#122)
    - [记录](#记录-5)
  - [1.28、29](#12829)
    - [记录](#记录-6)

# 寒假work

## 1.15
### 理解项目
+ 下载G4序列和已测好的甲基化序列，寻找重叠部分
+ LTR结构中具有CpG，易发生甲基化，但G4会抑制甲基化
+ fasta和bed文件的转化
### 待做（bed、pipeline、split、replace、join）
+ 理解pipeline √
+ 爬虫批量下载文件
+ 文献阅读
  +  NAR Quadruplex-forming sequences occupy discrete regions inside plant LTR retrotransposons

  + NAR Whole genome experimental maps of DNA G-quadruplexes in multiple species

  + Nature structural & molecular biology https://doi.org/10.1038/ s41594-018-0131-8.

  + Cell http://dx.doi.org/10.1016/j.cell.2016.06.044

  + scientific report https://doi.org/10.1038/s41598-017-14017-4

  + gene http://dx.doi.org/10.1016/j.gene.2017.04.051

  + gbe Evolutionary Dynamics of Retrotransposons Assessed by High-Throughput Sequencing in Wild Relatives of Wheat
+ tmux
### 记录
+ bed文件
    + 3个必须的列和9个额外可选的列
        + chrom（染色体名字）、目标区段起止位置
        + strand ：定义链的方向，''+” 或者”-”
    thickStart ：起始位置(例如，基因起始编码位置）
    thickEnd ：终止位置（例如：基因终止编码位置）　
    itemRGB ：是一个RGB值的形式, R, G, B (eg. 255, 0,0), 如果itemRgb设置为'On”, 这个RBG值将决定数据的显示的颜色。
    blockCount ：BED行中的block数目，也就是外显子数目
    blockSize：用逗号分割的外显子的大小, 这个item的数目对应于BlockCount的数目
    blockStarts ：用逗号分割的列表, 所有外显子的起始位置，数目也与blockCount数目对应

+ pipeline：
  + 目的：计算出染色体中G4的甲基化部分
    + Quadron_finder:
        + 输入：该条染色体的序列（fasta）
        + 结果：该条染色体中G4序列（bed）
        + 具体怎么实现的还不大明白？
    + trans_mCfile2bed_gz：
        + 输入：甲基化序列文件（tsv.gz格式）的路径，文件名
        + 结果：甲基化序列文件（bed格式）
        + 过程：
            trans2bed：
            + 输入：甲基化序列文件中的一行
            + 结果：按“/t”分开，分别将信息对应至bed文件中每列
    + calculate_coverage_mC_g4：
        + 输入：G4bed文件和甲基化序列bed文件
        + 输出：即下面函数的输出
        + 过程：
            mC_coverage_parser：
            + 输入：G4bed文件和甲基化序列bed文件合并后的文件（利用bedtools处理，还不太理解？）
            + 输出：重叠的序列，总的序列，重叠序列占比，hist（画直方图）
    + fasta_all_5：
        + 输入：全序列文件和idfile（方便对比）
        + 输出：每个染色体自己的序列文件
        + 过程：
            按行分开，该行若有“>”则说明为新一条染色体的序列的开头，则flag+1，而从它开始到再一次读到“>”,flag均不变，即都写到对应第flag个染色体的文件中
+ python小tips：
    + split（某种符号或就空着）：
        + 将序列以该符号划分，每个部分做为数组的一项
        例如：str = "Line1-abcdef \nLine2-abc \nLine4-abcd";
         print str.split( ); 
         #以空格为分隔符，包含 \n 
         print str.split(' ', 1 ); 
         #以空格为分隔符，分隔成两个
        ['Line1-abcdef', 'Line2-abc', 'Line4-abcd']
        ['Line1-abcdef', '\nLine2-abc \nLine4-abcd']
        + split('\n')[0]是获取第一行的信息
    + replace('符号','')就是把相应符号替换掉，无论是几个符号，只要是符号都会被替换掉
        + replace(' ','')：把空格替换掉
    + join是字符串操作函数，操作的也是字符串，其作用结合字符串使用，常常用于字符连接操作
        + key="\t".join(('a','b','c'))（join括号里的单位只能为一个单位，所以里面那个括号去掉就会报错）
        结果：'a b c'
           result= key.split("\t") 
        结果:[a,b,c]

## 1.16
### 待做
+ bedtools学习
+ 数学建模
+ 文献阅读
### 记录（bedtools（intersect、coverage））
+ bedtools
学习网站：https://bedtools.readthedocs.io/en/latest/content/bedtools-suite.html
  + **intersect**
    + 目的：提取两组基因组特征的重叠（扩展后可以一次识别单个查询 ( -a ) 文件和多个数据库文件 ( -b ) 之间的重叠）
    + 基本操作：$ bedtools intersect -a A.bed -b B.bed
    报告A与B的共享间隔
    如：$ cat A.bed
        chr1  10  20
        chr1  30  40
        $ cat B.bed
        chr1  15   20
        使用如上命令，结果为 chr1 15 20
    
    **扩展功能基本都为在上述原始命令后接“-命令”**

    + -wa:报告原始A特征，即不仅仅是重叠的那部分，而是包含重叠部分的那整段序列
      如上例：$ bedtools intersect -a A.bed -b B.bed -wa
      结果为 chr1  10   20
    + -wb：报告 A 的重叠部分，然后是原始的B特征
      如上例，结果为 chr1  15  20  chr1 15  20
    **同时用-wa和-wb则显示的为A和B各自的包含重叠序列的完整序列**
    + -loj：对每个A特征都进行输出，不管有无重叠
        但若有重叠，将报告A及重叠部分；若无则报告A及NULL B特征
        chr1  10  20  chr1 15  20
        chr1  30  40  . -1  -1
    + -wo：在-wa和-wb同时使用效果的基础上，再加入一列报告重叠量（碱基对的量）
        如上例：$ cat A.bed
        chr1    10    20
        chr1    30    40

        $ cat B.bed
        chr1    15  20
        chr1    18  25

        $ bedtools intersect -a A.bed -b B.bed -wo
        chr1    10    20    chr1    15  20  5
        chr1    10    20    chr1    18  25  2
    + -wao：在-wo的基础上，还报告了A中与B无重叠的的序列
    + -u：如果存在一个或多个重叠，则报告 A 特征。否则，不会报告任何内容；-c扩展-u，在后加一列指示重叠特征的数量（非碱基对的量），另外对于A中无重叠的也会报告；-C则扩展-c在A对于多个文件的比较（分别计数与显示），再加上-filenames，则把对应标号改为文件名
    如下例：$ bedtools intersect -a A.bed -b B.bed -C -names a b
    chr1    10    20    a 2
    chr1    10    20    b 2
    chr1    30    40    a 0
    chr1    30    40    b 0
    + -v：报告A中与B无重叠序列的序列
    + -f：要求最小重叠分数，即要求至少重叠X%，才会报告
    + -s：仅在相同链上找重叠，而-S则限制在相反链上
  + **coverage**
    + 基本操作 $ bedtools coverage -a A.bed -b B.bed
    在 A 中的每个间隔之后，将报告：
      + B 中与 A 区间重叠（至少一个碱基对）的特征数。
      + A 中具有 B 中特征的非零覆盖率的碱基数。
      + A中条目的长度。
      + A 中具有 B 中特征的非零覆盖率的碱基比例。
    + -s：则按链计算覆盖率
    + -hist：为 A 文件中的每个特征创建覆盖率直方图
    如下例：
    $ cat A.bed
    chr1  0   100 b1  1  +
    chr1  100 200 b2  1  -
    chr2  0   100 b3  1  +

    $ cat B.bed
    chr1  10  20  a1  1  -
    chr1  20  30  a2  1  -
    chr1  30  40  a3  1  -
    chr1  100 200 a4  1  +

    $ bedtools coverage -a A.bed -b B.bed
    chr1  0   100 b1  1  +  3  30  100  0.3000000
    chr1  100 200 b2  1  -  1  100 100  1.0000000
    chr2  0   100 b3  1  +  0  0   100  0.0000000

    $ bedtools coverage -a A.bed -b B.bed -s
    chr1  0   100 b1  1  +  0  0   100  0.0000000
    chr1  100 200 b2  1  -  0  0   100  0.0000000
    chr2  0   100 b3  1  +  0  0   100  0.0000000

    $ bedtools coverage  -a A.bed -b B.bed -hist
    chr1  0   100 b1  1  +  0  70  100  0.7000000
    chr1  0   100 b1  1  +  1  30  100  0.3000000
    chr1  100 200 b2  1  -  1  100 100  1.0000000
    chr2  0   100 b3  1  +  0  100 100  1.0000000
    all   0   170 300 0.5666667
    all   1   130 300 0.4333333

    **实质上把B中所有序列都当作一种特征，则A中每段与B重叠的序列都被视为特征1，而无重叠的序列都被视为特征2**

    -s和-hist混用则可以按链来统计
+ 文献阅读：**Quadruplex-forming sequences occupy discrete regions inside plant LTR retrotransposons**
    + 小tips：DNA负链即为模板链，正链：复制中与新链序列相同的原单链，非模板链。
    具mRNA功能、进入宿主细胞后可直接作为模板合成病毒蛋白质的单链RNA病毒，称正链RNA病毒或(+)RNA病毒。

## 1.17
### 待做
+ python批量下载文件（甲基化）√
+ 询问拟南芥序列数据下载方法 √（暂未回）
+ 数学建模学习
### 记录
+ 批量下载甲基化文件：
    + 搜寻给定url中所包含的超链接
      + 问题：使用requests.get(url)时，以“www.baidu.com”没有问题，但用存储甲基化序列的网站就出现异常
    + 判断文件类型：tsv.gz
    + 从链接中提取保存的文件名
    + 已实现：scripts-downloadfile

## 1.18
### 待做
+ 数学建模学习
+ 生信理论学习
### 记录
+ 数学建模：
    + 图与网络模型：
      + 相关经典问题：SPP（从某点到某点的最短路）、公路连接问题（最小生成树）、指派问题、CPP（中国邮递员问题）、TSP（旅行商问题，哈密顿图）、运输问题
      + 数据结构：
        + 邻接矩阵（两点存在边即为1）：稀疏网络则浪费空间
        + 关联矩阵（行为点，列为边）：稀疏网络则浪费空间
        + 弧表示法：只保存弧，即每条弧的起点终点和权，适合稀疏网络
        + 邻接表表示法：对图的每个节点，用一个单向链表列出从该节点出发的所有 弧，链表中每个单元对应于一条出弧。
        + 星形表示法：前向星形表示法：首先存放从节点 1 出发的所有弧，然后接着存放从节点 2 出发的所有孤，依此类推，最后存放从节点n 出发的所有孤。（查入弧方便）
        反向星形表示法：首先存放进入节点 1 的所有孤，然后接着存放进入节点 2 的所有弧，依此类推， 最后存放进入节点n 的所有孤。(查出弧方便)
        **① 星形表示法和邻接表表示法在实际算法实现中都是经常采用的。星形表示法的优点是占用的存储空间较少，并且对那些不提供指针类型的语言（如 FORTRAN 语言 等）也容易实现。邻接表表示法对那些提供指针类型的语言（如 C 语言等）是方便的， 且增加或删除一条弧所需的计算工作量很少，而这一操作在星形表示法中所需的计算工作量较大（需要花费O(m) 的计算时间）② 当网络不是简单图，而是具有平行弧（即多重弧）时，显然此时邻接矩阵表示法是不能采用的。其他方法则可以很方便地推广到可以处理平行弧的情形。**
      + 若道路W 的边互不相同，则W 称为迹(trail)。若道路W 的顶点互不相同，则W 称 为轨(path)。起点和终点重合的轨叫做 圈(cycle)。 
        **(i) 图 P 是一条轨的充要条件是 P 是连通的，且有两个一度的顶点，其余顶点的度 为 2；(ii) 图C 是一个圈的充要条件是C 是各顶点的度均为 2 的连通图。**

## 1.20
### 待做
+ 把数据计入表中
### 记录
+ Python小tips：
    filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
+ 数据计入表中：
  + mC_coverage_parser(file1,file2)输出：
    covered：g4序列中有甲基化的序列数
    total：g4序列总数
    covered/total
    hist：无重叠的部分

'G4/MC':被甲基化的G4（个数）,'G4_total'：全部G4个数,'MC_rate'：第一个除第二个
'G4_MC_bp':bp形式的如上
'G4_all_bp'
'MC_frequncy'

## 1.21
### 待做
+ 下载
+ vcf.gz转为fasta
### 记录
+ 安装 bcftools 1.2,   htslib-1.2.1
  网址：http://blog.sina.com.cn/s/blog_e94982960102yzy4.html
+ vcf的解压方法：
  bgzip -d view.vcf.gz
  gunzip view.vcf.gz
+ vcf.gz转为fasta的方法：
+ https://samtools.github.io/bcftools/howtos/consensus-sequence.html（调用，规范化indels并进行过滤，但报错，怀疑与“chr”有关）
+ https://www.jianshu.com/p/a23d6f1226a1?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation（最终选择方法）
+ https://blog.csdn.net/Cassiel60/article/details/88998254(关于1和chr1)
  https://www.cnblogs.com/zhanmaomao/p/12570237.html（tabix用法）
https://www.biostars.org/p/204875/(关于Fasta 序列与 REF 不匹配错误，并以samtools faidx 检查 )

## 1.22
### 记录
  + 整理了pipeline(包括操作和输入表格)，只需调用deal_package(i)一步
  + 下载则package(i)，vcfgzToFasta(i,"/home/ubuntu/data/Arabidopsis_sequence/"+i)
  + tmux：新建会话tmux new -s my_session。
    在 Tmux 窗口运行所需的程序。
    按下快捷键Ctrl+b d将会话分离。
    下次使用时，重新连接到会话tmux attach-session -t my_session。

  ## 1.28、29
  ### 记录
  + 文献 GBE
      + 区分活跃与静止TE：LTR 逆转录转座子的单个活性拷贝的复制增殖产生了密切相关的序列群，其中大多数拷贝在基因组内具有高度的遗传相似性（Casacuberta 等人，1997）。相比之下，源自较老增殖事件的 TE 种群由于突变的积累而具有遗传异质性。从基因组序列中检测到的给定 TE 家族的大多数拷贝确实是有缺陷的，呈现出过早的终止密码子、插入缺失或进一步的重排，例如截断或嵌套插入 ( SanMiguel et al. 1996）。
      + 目的：1) 在全基因组范围内识别和量化不同 TE 家族在基因组中所占的比例；2) 评估具有不同基因组内容但大小相近的物种中主要 LTR 反转录转座子的进化动态
      + 基因组快照？
      + phred得分：通过自动DNA测序产生的核碱基进行鉴定的质量的度量。
        Phred质量得分与错误概率对数关联 
        Phred质量得分 错误的碱基检出的可能性（检出的碱基里有多少是错的） 碱基检出准确度（正确的有多少被检出来了） 
        10 十分之一 90% 20 十分之一 99% 30 1000分之一 99.9% 40 十分之一 99.99% 50 十万分之一 99.999% 60 1,000,000分之一 99.9999% 
      + 反转录转座子可以分成两大类：
      一类是LTR反转录转座子，包括Tyl-copia类和Ty3-gypsy类转座子，是具有长末端重复序列(long terminal repeats，LTR)的转座子，这也是反转录病毒基因组的特征性结构，这类反转录转座子可以编码反转录酶(Reverse transcriptases)或整合酶(integrases)，**自主地进行转录**，其转座机制同反转录病毒相似，但不能像反转录病毒那样以自由感染的方式进行传播，高等植物中的反转录转座子主要属于Tyl-copia类，分布十分广泛，几乎覆盖了所有高等植物种类。
      另一类是非LTR反转录转座子，包括LINE(long interspersed nuclear elements，长散布核元件)类、SINE(Short interspersed nuclear elements，短散布核元件)类、复合SINE转座子类，没有长末端重复序列(non-long terminal repeats，non-LTR)，自身也没有转座酶或整合酶的编码能力，需要在细胞内已有的酶系统作用下进行转座。
      + **共有位点**（Consensus site）指的是蛋白质上总是被某一种特定的方式修饰的位点，修饰方式如 N- 或 O-糖基化、磷酸化等等。**共有位点的多寡可以用来评估序列的同源性**。与之相关的概念还有共有序列（英语：Consensus sequence）。
      + tblastn：给定蛋白质查找对应的碱基序列（在碱基数据库中）
        blastx：给定碱基序列查找对应的蛋白质（在蛋白质数据库中）
      + 方法：1）454个片段通过BLASTN被分类为TE、细胞器、编码和未分类序列
              2）TE在数据库TREP中比对，被依次归为类、目、超家族和家族；其中有罕见或先验未知的家族，通过**聚类**补充它
              + 评估采样对估计TE家族分布所占比例的影响：重新采样 999 次，并使用 R Cran 中的样本函数估计比例分布
              3）由于整体基因组庞大且测序的低覆盖率，选取的拷贝很难重复，可以视为单独的一种TE（这里对于平均长度为380bp的每个拷贝读取了300bp去数据库中比对）。再用ClustalW合并和对齐每个物种中这些拷贝
              **LTR 区域的 5' 端是反转录转座子的可变和诊断部分**
              4）TE 种群（即宿主基因组内）的个体拷贝之间的遗传分化通过固定指数 (KST) 进行评估，它代表了在种群之间观察到的遗传多样性在总多样性中的比例
              + 在 TE 家族中，通过分析序列间距离的错配分布（即成对核苷酸差异的比例），可以突出表明最近扩增的相似 TE 拷贝组的存在
        **问题、结论（哪三个参数）及其中的图，每个结论由何方法做出**

  ## 1.30
  ### 待办
  + 背单词
  + 数模
  + 读书
  + 机器学习
  + 文献