## 3.30
### /home/ubuntu/Arabidopsis/Scripts/deal_TEg4.py
直接运行deal_TE_g4_mc_all(filePath,number)即可

step 1:分染色体求TE序列中有G4结构的部分
os.system("bedtools coverage -a "+"chr"+str(i)+".ltr.bed"+" -b "+"chr"+str(i)+".g4.bed"+" -s "+" > "+ tmp1)
os.system("bedtools coverage -a "+"chr"+str(i)+".ltr.bed"+" -b "+"chr"+str(i)+".g4.bed"+" -s -hist "+" > "+ tmp2)

step 2:对以上获得的tmp1、tmp2中数据进行处理
deal_TE_g4()
TE(bp):即TE的总数量（bp数），总数量：读取tmp1，每一行对应一个TE+1，bp则在-hist可直接读
G4（bp):即G4的总数量（bp数）,从“number.csv”中读取
TE_G4(bp):即与位于TE中G4结构的数量，倒数第四列即指明了该TE中与几个G4结构重叠，累加即可
                    TE_G4+=int(array[-4])
如：
    chr4	1314356	1314448	*   92	+	17.85	0	0	92	0.0000000
    chr4	3032654	3032678	*	24	-	28.15	2	2	24	0.0833333
TE_G4_rate（bp）：G4中位于TE的比例
            TE_G4_rate=float(TE_G4)/float(G4)
G4_TE(bp):包含G4结构的TE数目（bp数）
            G4_TE+=1
G4_TE_rate（bp）：TE中包含G4结构的比例
            G4_TE_rate=float(G4_TE)/float(TE)

return TE,TE_G4,TE_G4_rate,G4_TE,G4_TE_rate,TE_bp,G4_TE_bp，G4_TE_bp_rate,TE_G4_bp_rate

step 3:make_data():将以上数据写入表格
make_data_all():总合五条染色体的数据，进行整个个体为单位的运算，并写入表格

step 4：createallDataset()：将数据汇总至datasetTE中
先检查有无该数据，若无则将TE_G4.csv中最后一行数据写入

### /home/ubuntu/Arabidopsis/Scripts/deal_TEg4_mc.py
直接运行deal_TE_g4_mc_all(filePath,number)即可

step 1:
求G4与TE的重叠区域，生成bed文件tmp1
os.system("bedtools intersect -a "+"chr"+str(i)+".g4.bed"+" -b "+"chr"+str(i)+".ltr.bed"+" -s "+" > "+ tmp1)

求tmp1（G4与TE的重叠区域）中的甲基化区域
os.system("bedtools coverage -a "+"tmp1.bed"+" -b "+str(number)+"_mC.bed"+" -s "+" > "+ tmp2)
os.system("bedtools coverage -a "+"tmp1.bed"+" -b "+str(number)+"_mC.bed"+" -s -hist "+" > "+ tmp3)

出于bp数据矫正的需要，求TE的甲基化区域
os.system("bedtools coverage -a "+"chr"+str(i)+".ltr.bed"+" -b "+str(number)+"_mC.bed"+" -s -hist "+" > "+ tmp4)

step 2:数据处理
deal_TE_g4_mc():
TE_G4(bp):即与位于TE中G4结构的数量（bp数）,从TE_G4.csv中获取
TE_G4_mc（bp）：位于TE中G4结构的甲基化数量（bp数）
                if(float(line[-1])>0):
                    TE_G4_mc+=1
TE_G4_mc_rate:位于TE中G4结构的甲基化程度
            TE_G4_mc_rate=float(TE_G4_mc)/float(TE_G4)
TE_G4_mc_bp_rate：先计算a=float(TE_G4_mc_bp)/float(TE_G4_bp)
        再求b=float(TE_mc_bp)/float(TE_bp)
        TE_G4_mc_bp_rate=a/b
TE_mc_bp：该染色体中TE被甲基化的bp数           
return TE_G4,TE_G4_mc,TE_G4_mc_rate,TE_G4_bp,TE_G4_mc_bp,TE_G4_mc_bp_rate,TE_mc_bp

step 3:make_data():将以上数据写入表格
make_data_all():总合五条染色体的数据，进行整个个体为单位的运算，并写入表格
其中TE_G4_mc_bp_rate：先计算a=float(TE_G4_mc_bp)/float(TE_G4_bp)
        TE_mc_all_bp由每次染色体计算返回的TE_mc_bp累加所得
        再求b=float(TE_mc_all_bp)/float(TE_bp)
        TE_G4_mc_bp_rate=a/b

step 4：createallDataset()：将数据汇总至datasetTE中
先检查有无该数据，若无则将TE_G4.csv中最后一行数据写入