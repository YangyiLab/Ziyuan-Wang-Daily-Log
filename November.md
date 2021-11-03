- [2021-11-1](#2021-11-1)
  - [PLAN](#plan)
  - [进入系统填报信息](#进入系统填报信息)
    - [Penn State](#penn-state)
  - [MADISON](#madison)
  - [微生物结论](#微生物结论)
  - [修改甲基化文件 转成bed](#修改甲基化文件-转成bed)
- [2021-11-2](#2021-11-2)
  - [PLAN](#plan-1)
  - [deep learning approach antibiotics](#deep-learning-approach-antibiotics)
- [2021-11-3](#2021-11-3)
  - [PLAN](#plan-2)

# 2021-11-1
## PLAN
+ **修改PS**
+ **微生物作图**
+ **进入系统填报信息**
+ **修改甲基化文件 转成bed**

## 进入系统填报信息
### Penn State
+ 地址 No.29 Wangjiang Road, Chengdu, Sichuan, China,610064
+ 密码 wzy851234wzy851234
+ 登录号 zxw5399@psu.edu

## MADISON
+ 密码 5#DvMnQwAwHhL+r
+ user name pry0921

## 微生物结论
无植物 各种处理alpha beta多样性都没有区别

## 修改甲基化文件 转成bed
代码
```python
#   API trans_mCfile2bed
#   trans_mCfile2bed(path,input_file_name)
#   path 文件相应路径 file_name 甲基化文件名称

def judge_if_mC(line):
    if str(line).endswith("1"):
        return True
    return False

def trans2bed(line):
    line_list=line.split("\t")
    chr_num="chr"+line_list[0]
    start_position=line_list[1]
    end_position=str(int(start_position)+1)
    name=line_list[3]
    length_of_mC="1"
    strand=line_list[2]
    linr_out="\t".join([chr_num,start_position,end_position,name,length_of_mC,strand])
    return linr_out

def trans_mCfile2bed(path,input_file_name):
    # path= "/home/ubuntu/Arabidopsis/Arabidopsis_sequence/Ag-0/"
    # input_file_name="GSM1085193_mC_calls_Ag_0.tsv"
    f = open(path+input_file_name,"r")
    mc_data=f.read()
    mc_lines=mc_data.split("\n")
    [judge_if_mC(i) for i in mc_lines]
    mc_true_lines=list(filter(judge_if_mC,mc_lines))
    list(mc_true_lines)
    mc_true_lines_out=[trans2bed(line) for line in mc_true_lines]
    output_file_name=input_file_name[20:-4]+".mC.bed"
    f_bed=open(path+output_file_name,"w")
    f_bed.write("\n".join(mc_true_lines_out))
    output_file_name=input_file_name[20:-4]+".mC.bed"
    return output_file_name
```

# 2021-11-2
## PLAN
+ **微生物功能分析图**
+ **GCN启发论文**

## deep learning approach antibiotics
方法 MPNN 消息传递神经网络 GNN的一种

# 2021-11-3
## PLAN
+ **微生物热图**
+ MINSET数据集overvoew pytorch回忆 基于教材 Dive into