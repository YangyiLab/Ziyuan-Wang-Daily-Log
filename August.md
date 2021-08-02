- [2021-8-1](#2021-8-1)
  - [PLAN](#plan)
  - [phage 统计](#phage-统计)
  - [UIUC LAB](#uiuc-lab)
- [2021-8-2](#2021-8-2)
  - [PLAN](#plan-1)
  - [NTU LAB](#ntu-lab)
  - [群体遗传-固定系数](#群体遗传-固定系数)
    - [三种衡量](#三种衡量)
# 2021-8-1
## PLAN
+ **GRE 填空2阅读2**
+ **收集双链phage序列并初步统计**
+ **修改abstract graphic**
## phage 统计
+ length

**Double Strand linear >>Single Strand circular**
+ gc_content
+ gc_skew
+ rbs_distribution/number

28 features
"GGAGGA" dominant

## UIUC LAB
https://www.sinhalab.net/people

# 2021-8-2
## PLAN
+ **GRE 填空2阅读2**
+ **西瓜书chapter1 2 3**
+ **abstract graphic定稿**
+ **群体遗传3**
+ **Cover Letter**

## NTU LAB
https://www.yqiaolab.com/research
https://gohwils.github.io/biodatascience/
https://dr.ntu.edu.sg/cris/rp/rp00715?ST_EMAILID=ZHAOYAN&CategoryDescription=BiomedicalSciencesLifeSciences
https://www.scelse.sg/People/Detail/5dcdf536-2f65-498b-84bd-c4edb768d659
https://swuertzlab.com/people2/
http://compbio.case.edu/koyuturk/index.html

## 群体遗传-固定系数
[Last section](July.md)
### 三种衡量
+ 观测平均杂合子$H_I$ 每一个亚群观测杂合频率平均
+ 预期平均杂合子$H_S$ 每个亚群杂合子期望再平均
+ 全群体杂合子 $H_T= 2\bar{p}\bar{q}$

Fis衡量由于**亚群内**非随机交配造成的杂合子频数观测值和预期值的差异。
Fst衡量由于[种群结构](https://gitee.com/YangyiLab/Daily-Worklog/blob/master/July.md#%E7%BE%A4%E4%BD%93%E9%81%97%E4%BC%A0-%E7%BE%A4%E4%BD%93%E7%BB%93%E6%9E%84%E5%92%8C%E5%9F%BA%E5%9B%A0%E6%B5%81)的存在而造成的杂合子频数的降低。 **由于基因流有限，造成等位基因频率在群体中分布存在异质性**
Fit同时衡量了亚群内非随机交配和种群结构导致的杂合子频数的变异。

$$F_{st}=\frac{H_T-H_I}{H_T}$$
Fst是经常用来衡量种群结构和种群分化的参数