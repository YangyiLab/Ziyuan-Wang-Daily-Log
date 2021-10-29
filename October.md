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
  - [PHD材料](#phd材料)
    - [list](#list)
    - [Madison *December 1*](#madison-december-1)
    - [Penn State *ddl December 15th*](#penn-state-ddl-december-15th)
    - [UArizona *ddl December 1st*](#uarizona-ddl-december-1st)
    - [Sinai *ddl December 1*](#sinai-ddl-december-1)
    - [Hawaii *ddl January 15*](#hawaii-ddl-january-15)
    - [NTU *31 January*](#ntu-31-january)
    - [NUS *ddl 1.15 or 15 Nov*](#nus-ddl-115-or-15-nov)
    - [UThealth *ddl December 1*](#uthealth-ddl-december-1)
  - [VEGA](#vega-1)
    - [problems](#problems)
- [2021-10-20](#2021-10-20)
  - [PLAN](#plan-7)
  - [VAE & VI](#vae--vi)
    - [VAE](#vae)
    - [VI 与EM区别](#vi-与em区别)
    - [Nature method 文章](#nature-method-文章)
    - [VEGA scVI 异同点总结](#vega-scvi-异同点总结)
  - [TE G4分析思路](#te-g4分析思路)
- [2021-10-25](#2021-10-25)
  - [PLAN](#plan-8)
  - [PS结构](#ps结构)
    - [开头](#开头)
    - [教育经历](#教育经历)
- [2021-10-26](#2021-10-26)
  - [PLAN](#plan-9)
  - [DT](#dt)
  - [PS 第一段科研经历](#ps-第一段科研经历)
    - [汉语架构](#汉语架构)
  - [甲基化文件](#甲基化文件)
- [2021-10-27](#2021-10-27)
  - [PLAN](#plan-10)
  - [PS 第二段科研经历](#ps-第二段科研经历)
    - [汉语架构](#汉语架构-1)
  - [Stefan论文](#stefan论文)
    - [The Protein Interactome of Streptococcus pneumoniae and Bacterial Meta-interactomes Improve Function Predictions](#the-protein-interactome-of-streptococcus-pneumoniae-and-bacterial-meta-interactomes-improve-function-predictions)
- [2021-10-28](#2021-10-28)
  - [PLAN](#plan-11)
  - [PS结尾](#ps结尾)
  - [Generative Model](#generative-model)
    - [波兹曼机](#波兹曼机)
    - [深度信念网络](#深度信念网络)
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
+ **分子生物学作业+复习**
+ **VEGA论文阅读**
+ **甲基化阅读**
+ **统计PHD材料**

## PHD材料
### list
1. Madison
2. Penn State
3. UArizona
4. Sinai
5. Hawaii
6. NTU
7. NUS
8. UThealth
9. **KAUST** MSC

### Madison *December 1*
website https://microbiology.wisc.edu/how-to-apply/


+ **Personal statement, also known as “Statement of Purpose”.** The applicant’s personal statement is important. It should be written thoughtfully to reveal one’s scientific and career interests, research experiences, and abilities that otherwise may not be obvious from the academic record.  See Personal Statement Guidelines and Evaluation Criteria below for detailed information on what our faculty and student reviewers are looking for in a personal statement. *一份PS*
+ **A statement of applicant’s experiences and ambitions** that will contribute to MDTP’s commitment to diversity and inclusion.  See Diversity Statement Guidelines below for more detailed information on this diversity statement. *一份RP*
+ **An official or unofficial copy of transcripts** from each college or university attended. The transcripts should be uploaded to the application website. *成绩单*
+ Three or more **letters of reference from individuals** (faculty, staff, supervisor, mentor) who can comment on the applicant’s qualifications.  This should include scholarly and academic qualifications, and can also include experiences in teaching, outreach, and community service.  Refer to the Evaluation Criteria below to note what the Admissions Committee will be examining.  These letters must be sent electronically. Directions for submission will be provided once you have initiated your application. *三封推荐信*
+ **A brief resume/CV** listing academic awards, scholarships, location and length of research experiences, co-authorship on any publications or presentations at scientific conferences. The resume/CV should be uploaded to the application website under the “Statements and CV” tab. *CV一份*

### Penn State *ddl December 15th*
website https://science.psu.edu/bio/grad/how-to-apply
A. Complete a formal online application through the Penn State Graduate School (select Degree Admission). There is an application fee payable to the Graduate School. The Department of Biology uses five essay prompts in lieu of a personal statement.

  + **Provide the names and contact information for three references**. We realize that in writing recommendation letters, recommenders often write a generic letter for most programs. If their letter does not already do so, the Department of Biology would like for them to address in a paragraph their thoughts about the potential for the applicant to be a future scientist. In particular, we value their evaluation of the future ability of the applicant to ask strong biological questions, design critical experiments, collect and analyze data, and draw biological conclusions from the data. Do you think that the applicant will be resilient when research difficulties arise? How do you think that the applicant will work as a member of a diverse team to help solve problems? A letter with superlatives is less helpful than a letter that provides facts about the applicant's academic and research experiences. (三封推荐信)
  + Upload a resume or curriculum vitae. (简历)
  + Upload a list of the Graduate Faculty with whom you are interested in working. (HYF)
  + Upload an official copy of transcripts (university records) from each institution of higher education attended as an undergraduate or graduate student. International transcripts must be submitted in both the original language and in English. Examples of documents that are not acceptable are: advising transcripts or a degree audit from a self-service website.
  The Department of Biology no longer evaluates GRE scores for admission to our graduate program.(成绩单)

B. International applicants must also submit results of the Test of English as a Foreign Language (TOEFL) electronically to Penn State. Institution code: 2660. (雅思)
Scores from the International English Language Testing System (IELTS) can be submitted in lieu of the TOEFL. Hard copy scores should be sent directly from the IELTS testing agency to Penn State's Graduate School at the following address:

### UArizona *ddl December 1st*
https://www.pharmacy.arizona.edu/academics/graduate-programs/application-requirements

+ Prerequisite Degrees (学历)
Applicants should have earned a RECOGNIZED BACHELOR'S DEGREE appropriate for their desired program track

+ Graduate Record Examination (Optional) (GRE)
The GRE will no longer be required as part of the application packet.  Students have the option of reporting scores but can submit an application without them.  The institution GRE code is 4832. The average scores in the COP graduate programs in previous semesters are:
Verbal: 154 (61%)
Quantitative: 152 (52%)
Analytical Writing: 4.5 (82%)

+ Transcripts (成绩单)
You are required to submit official transcripts from each college or university you have previously attended or currently attending. Transfer credits indicated on another school's transcript are not accepted in lieu of submitting the original institution's record. The transcript must contain the official school seal or stamp and the signature of the Registrar. 

+ Letters of Recommendation (推荐信)
Applicants are required to submit three letters of recommendation, preferably addressed specifically to our program. Letters should be from people with direct knowledge of your academic and/or professional performance and potential. Ideally, at least one or two would come from a former professor - who has had you in class and can speak to your educational skills and performance. Letters from employers and supervisors are also acceptable.

+ Personal Statement (PS)
Applicants must submit a personal statement that is 1-2 pages and single spaced. It should include the following:
Your career goals and how graduate study would contribute to their realization
How professional, educational and life experiences have influenced your desire to pursue graduate study
Your goals and objects for graduate student

+ Résumé(简历)
Applicants must submit a resume that presents your background and skill set. Be sure to include a summary of relevant job experience and education.

+ Research Experience (研究经历)

### Sinai *ddl December 1*
https://icahn.mssm.edu/education/admissions/graduate-education/phd-biomedical-sciences
+ 原始正式成绩单。成绩单应放在密封的机构信封中。或者，注册商可以将成绩单直接发送至：
+ 支付申请费（80 美元）是在线申请的一部分。系统将提示您如何通过信用卡、支票或汇票付款。
+ 提供推荐信的三个人的联系信息。我们特别有兴趣听取有机会在研究环境中与您互动的个人的意见。请提供他们的电子邮件地址，以便您可以在出现提示时提交信息。这些推荐人将收到一封来自“admissions@mssm.edu”的电子邮件，其中包含指导他们完成整个过程的信息。
+ 我们的目标是评估您成功完成所需研究生课程的潜力。如果英语是您的第二语言， 则需要英语作为外语考试 (TOEFL)。 whether IETLS OK?

### Hawaii *ddl January 15*
Tropical Medicine https://jabsom.hawaii.edu/ed-programs/masters-phd/admissions/

Applicants to all programs must submit the following to the Graduate Admissions Office:
+ Completed Graduate Admissions Application 申请表
+ 推荐信两份
+ Application fee 申请费
+ One Official Transcript for each post-secondary institution attended 成绩单
+ In addition, certain applicants may be required to submit the following:
  + Official Standardized Exam Scores
  + Residency Declaration Form
  + Official TOEFL or IELTS Exam Scores IETLS > 7.0 助教
(See International Students < English Proficiency.)
  + Proof of Sufficient Funding 财产证明
(See International Students < Financial Statement.)

### NTU *31 January*
+ Passport/ Identification card (NRIC)

+ One recent passport-sized colour photograph

+ Valid TOEFL/IELTS/GRE/GMAT/GATE Scores

+ Degree Certification and Official Transcripts (Bachelor and/ or Masters)

  + Degree Certificate

    + Original Language
    + Official translation in English (if original is not in English)

  + Official Transcripts
    a. Original Language
    b. Official translation in English (if original is not in English)
    c. Grading or marking scale of the Transcript (Interpretation of Grades/Marks)
+ Two Academic Referees' Reports
+ Other Supporting documents (if applicable)
Research proposal
Resume
Research publications

### NUS *ddl 1.15 or 15 Nov*
杜克
Depending on your intended concentration area,
+ Biostatistics and Health Data Science: a master’s degree or a bachelor’s degree in a quantitative discipline (Statistics/Biostatistics/Math/Computer Science/Epidemiology).
Computational Biology: a bachelor’s degree in biological, computational, or quantitative discipline.
All applicants must have completed, or be in the final year of, a bachelor or honours degree
+ Graduate Record Examination (GRE) General Test results
+ 3 – 5 references, typically from professors, mentors and/or employers.

生命学院
+ Singapore NRIC (for Singapore Citizens or Singapore PRs); Passport information page (for International applicants)
+ Certified true copy of official Bachelor’s or Master’s transcript
+ Certified true copy of official Bachelor’s or Master’s Degree certificate
Other academic certificates (if applicable)
+ Valid TOEFL/IELTS score
Applicants whose native tongue and medium of university instruction is not completely in English should upload the official score sheet of TOEFL (≥85) or IELTS (≥ 6.0) as evidence of their proficiency in the English language.
TOEFL/ IELTS scores are valid for 2 years from the test date and should not have expired at point of application. Expired scores will not be considered for the application.
+ Certified true copy of GRE or GATE scoresheet
+ Two academic referee reports
+ CV 
+ A summary of your education, work experience, co-curricular activities, community service, etc.
Personal Statement of around 2 pages
You may include your broad research interests, previous research accomplishments and personal vision of future career.
+ Financial Statement or Sponsorship Letter (if you have opted for ‘Self-Finance’ at point of application)
+ Passport size photograph following the specifications listed here


### UThealth *ddl December 1*
+ ✓ 申请表
+ ✓ 60 美元的申请费
+ ✓学士学位或更高学历
+ ✓所有就读学院和大学的正式成绩单（国际申请者请参阅下表的其他要求）
+ ✓平均绩点 (GPA)：没有最低 GPA 要求。大多数成功申请者的 GPA 为 3.0 或更高
+ ✓ 研究生入学考试 (GRE)
✓我们博士课程的大多数成功申请者的个人 GRE 分数（定量、口头和分析）在 50% 或更高。
+ ✓简历和/或履历
+ ✓三份来自教育工作者和雇主的推荐信
+ ✓目标声明
+ ✓请注意：强烈建议目前申请博士课程的 SBMI 学生提交至少一封 SBMI 教员的推荐信。
+ ✓仅适用于博士申请者： 如果您就读于美国以外的学院或大学，您的申请可以使用该机构的成绩单进行审核。但是，在提交 WES 或 ECE 的逐门课程教育评估之前，不能延长 SBMI 的录取通知。
+ ✓仅适用于博士申请者：如果您提交的申请没有 WES 或 ECE 要求的逐门课程教育评估，您将需要使用iGPA 计算器将您的 GPA 转换为美国量表。您需要上传转换后的 GPA 的 PDF 副本以及成绩单。请确保您准确报告成绩单上列出的所有课程。不这样做可能会取消您的申请资格


## VEGA
### problems
+ tSNE UMAP 降维方法
+ VAE Variable-AE 变分

# 2021-10-20
## PLAN
+ **选校**
+ **决策树代码完毕**
+ **甲基化论文标注**
+ **变分总结**


## VAE & VI
### VAE
全连接(encoder) -> $\mu,\Sigma$ 向量 -> latent 向量 -> 全连接(decoder)

不同的类别的latent向量为不同的正态分布，相同的趋向相同的正态分布，同时可以有效降维，和计算每一个latent变量，vega中记为GMV与gene的关系，即gene GMV的enrichment

### VI 与EM区别
*EM 如GGM 混合高斯分布， E步根据t时间的均值方差算后验 
VI中由于后验分布算不出* not sure
### Nature method 文章
VI算出隐藏变量 文章中利用KL作为loss函数变体，算出$Z_n$,后续算出

### VEGA scVI 异同点总结
异
+ VEGA尽可能恢复到原来的值 scVI恢复的值服从ZINB [ZINB](https://www.jianshu.com/p/aa948e1be21c) loss 接近原来的值

同
+ 都考虑到了batch effect
+ 都得出latent $Z$

## TE G4分析思路
+ 甲基化在全基因组的破坏性和甲基化在LTR区域破坏性
  + LTR全部识别后做出进化树 海明距离聚类 按照2017Nature structure上方法做
  + G4在LTR上 聚类 参照JGG
  + 甲基化聚类到的G4 再次聚类到TE(LTR) 功能聚类


# 2021-10-25
## PLAN
+ **毕业设计选题确定**
+ **ML决策树作业完成**
+ **PS overview**
+ **phyml绘图**

## PS结构
### 开头
Frankly, when first exposed to microbiology in the second semester of my junior year, I found myself quite ignorant of this academic discipline. To be sure, as a student major in computational biology from the Department of Biology & the Department of Software Engineering, Sichuan University, I had sound foundation in programming and biology; but it required an equally sound knowledge base in microbiology, genetics and sequencing technique in order to perform well in a host of seminars on bioinformatics. To develop my interdisciplinary knowledge, I tried desperately to be educated in microbiology through self-study, consulting a large amount of relevant technical literature, attending lectures, and discussing with classmates.
Indeed, that was a period of overwhelming fatigue and excitement. Nevertheless, the feeling of learning something new and making important progress was itself a source of constant pride. Looking back, this experience has proved decisive in determining my academic interest in biostatistics and in the related bioinformatics. It has further shaped my research interest in the Ph.D. program at your esteemed university.
### 教育经历
I am currently completing my degree in computational biology at Sichuan University. I believe that my exposure to the multi-disciplinary approach of Bioinformatics makes me a good candidate for your program. I have a solid foundation in biology and have gotten straight. As in all courses including cell biology, genetics and biochemistry. Likewise, I score similarly across my computer courses including Data Structures and Algorithms, and Databases and Computational Logic. In addition, I completed courses in mathematical modeling, discrete mathematics, and stochastic processes among others with an A average. My education in stochastic processes was a great help in my subsequent research, reflecting the value of interdisciplinary studies. Conducting a study on the identification of identical-by-decent (IBD) gene fragments in population flow, I applied continuous-time Markov chain to construct a mathematical model and abstracted the complex gene exchange problem into a birth and death chain to solve the problem.


# 2021-10-26
## PLAN
+ **ML决策树作业完成**
+ **PS 第一段**
+ **微生物功能分析**
+ **阅读Stefen论文**

## DT
```python
for value in uniqueVals:
      if splitDataSet(dataSet, bestFeat, value) == []:
          # 返回多数表决
          return majorityCnt(classList)
      subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
      del(subLabels[bestFeat])
      myTree[bestFeatLabel][value] = treeGenerate(splitDataSet(dataSet, bestFeat, value),subLabels,original_dataSet)
  return myTree
```
uniqueVals 使用原始数据集中所有可能取值，如果该类取值未考虑，则直接返回父节点的多数表决值


## PS 第一段科研经历
### 汉语架构
+ 科研主题/title是什么？

hitchhiking研究  Soil Nitrogen Impacts Rhizosphere Microbial Community Due to Microbial Hitchhiking
+ 如何完成科研工作，你在里面扮演了什么样的角色？
处理数据，写论文
详细版本：首先查看测序数据进行处理，之后进行16S扩增子分析，包括群落多样性，功能丰度，已经细菌间相互作用分析。同时在形成手稿时，我整个承担了作图以及论文写作

+ 科研过程中遇到了什么困难？是如何去解决的，例如用了什么理论/模型/办法，有什么创新or改进？
困难: 
如果做常规分析，没有任何创新性，在对数据进行挖掘时，我们最后结合了最新报道的试验性文章的结论，通过本实验的生物信息学验证，印证了此前的报道。
在写introduction时遇到了较大的问题，不知道如何介绍整个领域的研究进展，不知道如何挑选文献等。解决方式为和导师详细交流，共同讨论，学到了如何讲好故事，如何整理逻辑。

+ 科研最终得到了什么研究结论/给出了什么policy建议。得到了导师的褒奖、获什么奖了吗？
最终成果为，完成一篇文章，并投稿。文章的意义在于对于一个大家日常接受的事实 适量施肥利于植物生长 的根本机理从全新的方面做出了解释，即从微生物方面对于进行了解释。

+ 自己的收获和体会，该经历给自己带来了什么（有什么意义）？例如，能力的提升、兴趣的拓展、发现自己需要更深入的学习以达成未来目标云云。
能力的提升 包括如何处理测序数据，如何对微生物组数据进行深入探究，如何写学术论文读文献等等。


## 甲基化文件
including chrom pos strand Mc_class methylated_bases total_bases methylation_call.
这几项的说名


# 2021-10-27
## PLAN
+ **PS 第二段**
+ **决策树再次复习**
+ **Stefan论文精读**

## PS 第二段科研经历
### 汉语架构
+ 科研主题/title是什么？

Gene Identification and Classification In the Viral Genome
+ 如何完成科研工作，你在里面扮演了什么样的角色？

阅读文献，构建算法
详细版本：我们要处理的问题是识别噬菌体基因组的编码区。我主要负责首先查询已有的算法，找到已有算法的问题，对于噬菌体宏基因组采样的噬菌体基因片段进行分类，分类后开发噬菌体基因组基因识别算法。

+ 科研过程中遇到了什么困难？是如何去解决的，例如用了什么理论/模型/办法，有什么创新or改进？
**困难:** 
  + 对于噬菌体基因组，目前测序数据比较少，没有可靠的数据集,现有的基因预测都是利用Prodigal和Phanotate进行预测的，需要开发新算法进行编码区预测。
  + 噬菌体DNA类型很多包括了单链双链，环状和线状
  + 对于宏基因组测序，获得的序列都是片段的而非全基因组拼接后的序列

解决方案
查阅资料，使用最新的技术进行解决。在基因注释时，不使用训练集进行训练，开发generative model直接基于序列进行学习。
在进行基因编码区之前，首先基于片段序列进行预测，预测噬菌体DNA类型。
因为算法需要全基因组的数据，遇到片段时难以处理，我们最终采用，学习每一类噬菌体全基因组的近似数据，当遇到片段时使用全基因组近似数据进行处理，并预测基因。

+ 科研最终得到了什么研究结论/给出了什么policy建议。得到了导师的褒奖、获什么奖了吗？

由于时间较短，目前只完成了噬菌体DNA分类器，但取得了不错的进展，威斯康星麦迪逊的导师评价我的工作很不错，在完成第一部分后我们进行了下一部分的大量文献阅读以及其他软件的测试，并提出了初步想法，这个项目将在我的毕业论文中实现。

+ 自己的收获和体会，该经历给自己带来了什么（有什么意义）？例如，能力的提升、兴趣的拓展、发现自己需要更深入的学习以达成未来目标云云。

能力的提升 包括如何正确使用公共数据集，如何对病毒组数据进行深入探究，如何展开和其他机构合作。


## Stefan论文
### The Protein Interactome of Streptococcus pneumoniae and Bacterial Meta-interactomes Improve Function Predictions
主题 利用蛋白相互功能由已知功能推测未知功能
算法 **模拟退火算法**
同时也要计算不同功能类别之间的联系

# 2021-10-28
## PLAN
+ PS 全部完成
+ 审稿意见回复
+ GAN overview

## PS结尾
+ 第一部分阐述自己未来的职业规划，常见按照长短期分别说明，毕业后短期（1-3年 ）想做什么工作，长期中（5-10年）想达到什么职业目标。职业规划通常需要和学员所申请专业方向、过去实习经历以及自己的远期目标/愿景联系起来。
Considering that I am applying for a PhD program, my short-term career plan is to find a scientist position in a technology company or enter a university after graduation. However, in the long run, my future is uncertain and there are many possibilities. I may consider starting a business centering on biotechnology.


+ 陈述择校原因，夸对方学校。常见原因包括：学校名声、项目设计、课程设置、知名教授、求职机会、校友资源、地理位置、个人情节等等。
                                                    The University of Wisconsin-Madison is a prestigious university known as the public Ivy.
Your school has made outstanding progress in many fields. The Microbiology field I applied for has a large number of achievements, and has published a large number of meaningful papers in Nature Microbiology, Microbiome, ISME and other magazines. I particularly appreciate the work of Professor Karthik Anantharaman, whose work covers both bioinformatics and experiment-based microbiology.
My undergraduate scientific research experience is mainly related to bioinformatics and macro omics data processing. If I study in University of Wisconsin-Madison, I can obtain more results by using high-quality laboratory resources.
Meanwhile, SINCE June 2021, I have had an internship in Professor Karthik Anantharaman's research group for four months. I have a preliminary understanding of his research and have completed part of the work and made preliminary progress.           

## Generative Model
**源头为波兹曼机**
目的：找到一层或者多层隐藏变量
### 波兹曼机
RBM 能量函数与波兹曼机有区别 求解为最大似然法
训练方法 
### 深度信念网络
与rbm区别 只有最后一层hidden layer是RBM