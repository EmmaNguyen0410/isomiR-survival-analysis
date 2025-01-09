import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival 
from scipy import stats

# Get list of selected isomiRs
selected_isomiRs = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/selected_features/isomiRs.csv")['feature'])
# read isomiR_profile 
isomiR_profiles = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data.csv")
isomiR_profiles = isomiR_profiles[isomiR_profiles['survival_in_days'] != "'--"]

def set_expression_level(row, upper, lower):
    """
    """
    if (row['log_read_count'] >= upper):
        return 'high'
    elif (row['log_read_count'] <= lower):
        return 'low'
    else:
        return ''
    
def plot_kp(high, low):
    ax = plt.subplot()
    ecdf_x = stats.ecdf(high)
    ecdf_x.sf.plot(ax, label='High')
    ecdf_y = stats.ecdf(low)
    ecdf_y.sf.plot(ax, label='Low')
    ax.set_xlabel('Time to death (days)')
    ax.set_ylabel('Empirical SF')
    plt.legend()
    plt.show()
    
def logrank_test(high, low):
    res = stats.logrank(x=high, y=low)
    print(res.pvalue)

for isomiR in selected_isomiRs:
    # Counts of the isomiR across all patients 
    read_counts = isomiR_profiles[str(isomiR)]
    read_counts = np.log1p(read_counts)
    # Get upper threshold (0.9 quantile)
    upper = read_counts.quantile(0.7)
    # Get lower threshold (0.1 quantile)
    lower = read_counts.quantile(0.3)
    # Create dataframe with that isomiR_ID, read_count, log_read_count, status, survival_in_days
    df = isomiR_profiles[[str(isomiR), 'status', 'survival_in_days']]
    df['log_read_count'] = np.log1p(df[str(isomiR)])
    # Mark log_read_count >= upper threshold as high-expressed, <= lower threshold as low-expressed 
    df['expression'] = df.apply(lambda r: set_expression_level(r, upper, lower), axis = 1)
    # only retain expression high or low 
    df = df[df['expression'] != ""]
    # Create censored/uncensored for high expressed
    mask_high = df['expression'] == 'high'
    ttf_high = [float(v) for v in list(df["survival_in_days"][mask_high])]
    high = stats.CensoredData.right_censored(ttf_high, list(df["status"][mask_high]))
    # Create censored/uncensored for low expressed
    mask_low = df['expression'] == 'low'
    ttf_low = [float(v) for v in list(df["survival_in_days"][mask_low])]
    low = stats.CensoredData.right_censored(ttf_low, list(df["status"][mask_low]))
    print(isomiR)
    logrank_test(high, low)
    plot_kp(high, low)
    


        

