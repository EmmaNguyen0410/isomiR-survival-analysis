# Create data
import scipy.stats as stats
import pandas as pd 
 
###### Demographic vs Demographic & isomiRs
# ### Cox ####
# demographic_cox = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic/cox.csv")['score'])
# demographic_isomiRs_cox = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_isomiRs/cox.csv")['score'])
# print(stats.wilcoxon(demographic_isomiRs_cox, demographic_cox, alternative="greater")) # WilcoxonResult(statistic=15.0, pvalue=0.21875)
# ### RSF ####
# demographic_rsf = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic/rsf.csv")['score'])
# demographic_isomiRs_rsf = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_isomiRs/rsf.csv")['score'])
# print(stats.wilcoxon(demographic_isomiRs_rsf, demographic_rsf, alternative="greater")) # WilcoxonResult(statistic=13.0, pvalue=0.34375)
# #### SVM ####
# demographic_svm = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic/svm.csv")['score'])
# demographic_isomiRs_svm = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_isomiRs/svm.csv")['score'])
# print(stats.wilcoxon(demographic_isomiRs_svm, demographic_svm, alternative="greater")) # WilcoxonResult(statistic=1.0, pvalue=0.984375)

###### Demographic vs Demographic & miRNAs 
### Cox ####
demographic_cox = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic/cox.csv")['score'])
demographic_miRNAs_cox = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_miRNAs/cox.csv")['score'])
print(stats.wilcoxon(demographic_miRNAs_cox, demographic_cox, alternative="greater")) # WilcoxonResult(statistic=3.0, pvalue=0.953125)
### RSF ####
demographic_rsf = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic/rsf.csv")['score'])
demographic_miRNAs_rsf = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_miRNAs/rsf.csv")['score'])
print(stats.wilcoxon(demographic_miRNAs_rsf, demographic_rsf, alternative="greater")) # WilcoxonResult(statistic=12.0, pvalue=0.421875)
### SVM ####
demographic_svm = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic/svm.csv")['score'])
demographic_miRNAs_svm = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_miRNAs/svm.csv")['score'])
print(stats.wilcoxon(demographic_miRNAs_svm, demographic_svm, alternative="greater")) # WilcoxonResult(statistic=1.0, pvalue=0.984375)

###### Demographic vs Demographic & miRNAs 
# ### Cox ####
# demographic_isomiRs_cox = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_isomiRs/cox.csv")['score'])
# demographic_miRNAs_cox = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_miRNAs/cox.csv")['score'])
# print(stats.wilcoxon(demographic_isomiRs_cox, demographic_miRNAs_cox, alternative="greater")) # WilcoxonResult(statistic=19.0, pvalue=0.046875)
# ### RSF ####
# demographic_isomiRs_rsf = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_isomiRs/rsf.csv")['score'])
# demographic_miRNAs_rsf = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_miRNAs/rsf.csv")['score'])
# print(stats.wilcoxon(demographic_isomiRs_rsf, demographic_miRNAs_rsf, alternative="greater")) # WilcoxonResult(statistic=8.0, pvalue=0.71875)
# ### SVM ####
# demographic_isomiRs_svm = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_isomiRs/svm.csv")['score'])
# demographic_miRNAs_svm = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_miRNAs/svm.csv")['score'])
# print(stats.wilcoxon(demographic_isomiRs_svm, demographic_miRNAs_svm, alternative="greater")) # WilcoxonResult(statistic=12.0, pvalue=0.421875)
