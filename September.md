- [2021-9-1](#2021-9-1)
  - [PLAN](#plan)
  - [群体遗传-自然选择模型复习](#群体遗传-自然选择模型复习)
  - [花书](#花书)
- [2021-9-2](#2021-9-2)
  - [PLAN](#plan-1)
  - [神经网络四分类输出出现问题](#神经网络四分类输出出现问题)
  - [RNN学习](#rnn学习)
    - [基本模型](#基本模型)
    - [更新函数](#更新函数)
    - [梯度](#梯度)
  - [论文阅读](#论文阅读)
- [2021-9-3](#2021-9-3)
  - [PLAN](#plan-2)
  - [GLMM数学推理](#glmm数学推理)
    - [R语言编程](#r语言编程)
    - [GWAS 应用](#gwas-应用)
  - [dataset skew](#dataset-skew)
  - [群体遗传-中性](#群体遗传-中性)
    - [多样性原因](#多样性原因)
- [2021-9-4](#2021-9-4)
  - [PLAN](#plan-3)
  - [Pairwise Alignment](#pairwise-alignment)
    - [Smith-Waterman 扩展](#smith-waterman-扩展)
  - [深度学习](#深度学习)
    - [文本表示模型](#文本表示模型)
  - [RNN-seq2seq](#rnn-seq2seq)
    - [Concept](#concept)
    - [features](#features)
    - [模型组成](#模型组成)


# 2021-9-1
## PLAN
+ **Gre阅读2填空2**
+ **群体遗传复习**
+ **QTL复习**
+ **深度学习花书overview**

## 群体遗传-自然选择模型复习
+ 单基因3等位基因模型
+ 双位点模型 

$T=t\quad$ $AA:w_{AA}p^2\quad$  $Aa:w_{Aa}pq \quad$  $AA:w_{aa}q^2$
$$p_{t+1}=\frac{w_{AA}p^2+w_{Aa}pq}{\bar{w}}$$
通过此式推导出双位点下一状态配子基因频率
+ $N_e$和$s$对比

## 花书
PCA降维的证明，通过降维编码恢复，的Frobenius范数在正交条件下的约束求解等价于通过特征值分解

$$\mathop{\argmin}_{d}  Tr(d^TX^TXd) \mathop{subject\quad to d^Td=1}$$

等价于协方差矩阵特征值

# 2021-9-2
## PLAN
+ **Gre阅读2填空2**
+ **4-分类问题调试神经网络**
+ **RNN overview**
+ **Katie 论文复习**

## 神经网络四分类输出出现问题
参考下已普遍应用的框架

## RNN学习
### 基本模型
+ 经典模型
+ 导师驱动预测
### 更新函数
从$t= 1$到$t = \tau$的每个时间步，我们应用以下更新方程：
$$a^{(t)} = b + \textbf{W} h^{(t-1)} + \textbf{U} x^{(t)}\\
  h^{(t)} = \tanh(a^{(t)} ), \\
  o^{(t)} = c + \textbf{V}h^{(t)}, \\
  \hat y^{(t)} = {softmax}(o^{(t)}) $$

### 梯度
同全连接网络
估计U,V,W,b,c 对每个变量求偏导

## 论文阅读
GLMM doi:10.1111/evo.13476

# 2021-9-3
## PLAN
+ **Gre阅读2 填空2**
+ **机器学习面经overview**
+ 深度学习RNN
+ **GLMM整理+论文整理(Ferris)**
+ **群体遗传**
+ **探究解决类别不平衡方法**

## GLMM数学推理

$$Y=\mathbf{\beta}^T\mathbf{X}+e$$

一般线性模型 简单线性模型其实是一个仅仅包含了固定效应的模型

**GLMM**
我们用误差项e来表示了其他一个不可控因素的影响。但是， 实际上，在误差项e中仍有一些因素是需要我们考虑的。我们可以把误差项e中的一部分拿出来，作为随机效应部分添加到模型中，也就形成了既包含“固定效应”，又包含“随机效应”的混合效应模型。

### R语言编程
```R
mod= lmer(data = , formula = y ~ Fixed_Factor + (Random_intercept + Random_Slope | Random_Factor))
```

随机因子可以体现在截距也可以体现在斜率上，随机因子体现的是**方差** 从而减小误差项e的值 **我们无法把所有的随机因素都考虑在内，我们只是尽可能的在误差项中剥离出一些随机因素的影响。**

数学模型
$$Y=(\mathbf{\beta}+\sum Condition^1_i \mathbf{\beta_i})\mathbf{X}+(\alpha+\sum Condition^2_i\alpha)+e'$$

### GWAS 应用
link https://gwaslab.com/2021/04/09/gwas%E7%9A%84%E7%BA%BF%E6%80%A7%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8Blmm-linear-mixed-model/

背景：解决早期的GWAS研究使用的模型为固定效应的线性模型，但通常会受两方面混淆因素的影响:
+ 群体结构/分层 （population structure/ stratification）：研究群体中存在有不同祖先（ancestry）的亚群体（subgroup）
+ 隐性关联（cryptic relatedness）：研究样本之间存在未知的亲缘关系

数学模型
$$\mathbf{Y}=\mathbf{W}\mathbf{\beta}+\mathbf{G_s}\lambda+e$$

Fixed effect: $W$ 和 $\beta$ $G_s$ $\lambda$
Random effect: g 
$$g\sim N(0,\sigma^2_g\mathbf{\phi})$$
error: e

## dataset skew
After I used Random Forest to train the classifier. When I am handling the test dataset, I will use predict the probability and filter those whose max prob lower than a thredhold in order to maintain as much as possible data and have a good accuracy.

when the thredhold is 0.45
I can maintain 72% data and have a 80% accuracy when I am dealing with a 4-class classification.

数据集不平衡解决方案:

+ 不对数据进行过采样和欠采样，但使用现有的集成学习模型，如随机森林
+ 输出随机森林的预测概率，调整阈值得到最终结果
+ 选择合适的评估标准，如precision@n

## 群体遗传-中性
### 多样性原因
尽管drift的固定或loss 很强，但由于$N_e$较大，平衡杂合度由有效群体和变异度觉得，这两项越大杂合度越高，即使drift作用大，由于群体较大，基因的多样性一样大。

# 2021-9-4
## PLAN
+ **Gre 套题1**
+ 深度学习RNN
+ **百面chapter1**
+ **生物序列overview**
+ phage训练

## Pairwise Alignment
### Smith-Waterman 扩展
寻找相似区域
+ 扩展包括寻找多处相似区域并进行连配 (如果序列一条或两条都比较长,通过设置阈值T决定分段数，和可识别的同源序列数量)

+ 交叠匹配(一条序列包含另一条的情况) 实现方法匹配可以起始于左端和顶端,匹配结束于右侧和底端，同时可以做重复序列查找实现。

## 深度学习
### 文本表示模型
Key Words: Bag of words, TF-IDF, topic model
$$TF\text{-}IDF(d,t)=TF(d,t)\times IDF(t)$$ 
TF:在d中t词频
$IDF(T)=log\frac{papers_{total}}{papers_{contains(t)}+1}$

N-gram N连单词划分

## RNN-seq2seq
### Concept
输入$\mathbf{(x^{(1)},x^{(2)},\text{...} ,x^{(\tau)})}$
通过第一层RNN训练结果为$\mathbf{C}$定长特征向量
再利用$\mathbf{C}$进行RNN训练，输出定长的$\mathbf{(y^{(1)},y^{(2)},\text{...},y^{(ny)})}$

### features
输入不定长，输入输出不等长。

### 模型组成
包括两个RNN
+ 第一个RNN输入为不定长序列，输出单一状态特征$h_{\tau}(x)$
+ 第二个RNN为通过$h_{\tau}(x)$ 训练输出等长度$\mathbf{(y^{(1)},y^{(2)},\text{...},y^{(ny)})}$