import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV, StratifiedKFold
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM, HingeLossSurvivalSVM, FastKernelSurvivalSVM, MinlipSurvivalAnalysis, NaiveSurvivalSVM
from sklearn import set_config
set_config(display="text")
from sklearn.model_selection import train_test_split

# Read data
raw_df = pd.read_csv("../../data/raw_data2.csv", index_col=False)
raw_df = raw_df[raw_df['survival_in_days'] != "'--"]
raw_df['survival_in_days'] = raw_df['survival_in_days'].astype(float)

# Preprocess raw data
random_state = 8
y_status = raw_df.pop('status').to_list()
y_survival_in_days = raw_df.pop('survival_in_days').to_list()
y = [(y_status[i], y_survival_in_days[i]) for i in range(len(raw_df))]
raw_df = raw_df.astype(float)
X_train, X_test, y_train, y_test = train_test_split(raw_df, y, test_size=0.2, random_state=random_state)

structured_y_train = np.array(y_train, dtype=[('status', '?'), ('survival_in_days', '<f8')])
structured_y_test = np.array(y_test, dtype=[('status', '?'), ('survival_in_days', '<f8')])

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["status"], y["survival_in_days"], prediction)
    return result[0]

##### Train model ######
param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'solver':['ecos', 'osqp'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']}
skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = random_state)
cv = list(skf.split(X_train, structured_y_train))
kssvm = MinlipSurvivalAnalysis(max_iter=10000)
kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)
kgcv = kgcv.fit(X_train, structured_y_train)
print(kgcv.best_score_, kgcv.best_params_)
estimator = MinlipSurvivalAnalysis(max_iter=10000)
estimator.set_params(**kgcv.best_params_)
estimator.fit(X_train, structured_y_train)
print(score_survival_model(estimator, X_train, structured_y_train)) # 0.6025063983761363
print(score_survival_model(estimator, X_test, structured_y_test)) # 0.6088082901554405

#### Feature importance ######
# 330       1.537873e-01         0.028565
# 1645      1.273131e-01         0.025109
# 398       1.018258e-01         0.032573
# 912       9.301752e-02         0.021329
# 1230      9.264742e-02         0.029616
# 941       8.734271e-02         0.035373
# 153       7.688132e-02         0.033810
# 652       7.377251e-02         0.031812
# 465       5.933876e-02         0.030369
# 1168      5.771034e-02         0.018846
# 380       5.692080e-02         0.036582
# 703       4.411547e-02         0.031016
# 369       4.209228e-02         0.015987
# 1036      3.972366e-02         0.027442
# 613       3.427091e-02         0.023055
# 1458      3.320997e-02         0.019404
# 1739      3.054528e-02         0.023023
# 1563      2.736245e-02         0.015009
# 761       2.674562e-02         0.013031
# 894       2.405625e-02         0.015040
# 1697      2.383420e-02         0.009039
# 320       2.264989e-02         0.021869
# 1522      2.257587e-02         0.010070
# 1702      2.040464e-02         0.005531
# 697       1.988650e-02         0.016523
# 1775      1.976314e-02         0.016891
# 698       1.773995e-02         0.026623
# 681       1.736985e-02         0.008745
# 1199      1.736985e-02         0.012107
# 965       1.719714e-02         0.027716
# 1648      1.680237e-02         0.009835
# 1075      1.665433e-02         0.005509
# 1166      1.658031e-02         0.016872
# 1618      1.618554e-02         0.020357
# 1354      1.598816e-02         0.009892
# 672       1.559339e-02         0.006773
# 338       1.352085e-02         0.006598
# 834       1.179373e-02         0.010612
# 1546      1.137429e-02         0.030294
# 1752      1.134962e-02         0.009790
# 1385      1.133728e-02         0.019400
# 1306      1.080681e-02         0.022253
# 378       1.060943e-02         0.008984
# 635       9.770540e-03         0.012737
# 391       9.696521e-03         0.006723
# 244       9.671848e-03         0.013330
# 139       9.301752e-03         0.008964
# 1693      8.388848e-03         0.009170
# 1034      8.142117e-03         0.007336
# 1196      8.117444e-03         0.007693
# 1131      8.092771e-03         0.006608
# 418       7.920059e-03         0.004471
# 647       7.920059e-03         0.007547
# 207       7.772021e-03         0.012733
# 1782      7.747348e-03         0.002113
# 643       7.549963e-03         0.008317
# 1706      7.549963e-03         0.006608
# 1293      7.303232e-03         0.003041
# 271       7.031828e-03         0.014845
# 999       6.686405e-03         0.004249
# 817       6.538367e-03         0.004946
# 1369      6.513694e-03         0.005425
# 920       6.365655e-03         0.006026
# 1785      5.995559e-03         0.003504
# 1002      5.773501e-03         0.004275
# 1400      5.773501e-03         0.007160
# 1180      5.576116e-03         0.005247
# 1591      5.428078e-03         0.003340
# 516       5.378732e-03         0.013544
# 1781      5.033309e-03         0.006401
# 1100      4.885270e-03         0.004201
# 1266      4.539847e-03         0.002319
# 1384      4.367135e-03         0.004771
# 440       4.367135e-03         0.003336
# 1664      4.293116e-03         0.002651
# 663       4.169751e-03         0.008771
# 455       4.095732e-03         0.001805
# 1447      4.071058e-03         0.004484
# 1043      3.700962e-03         0.008267
# 1393      3.552924e-03         0.002174
# 215       3.528251e-03         0.002517
# 560       3.404885e-03         0.009979
# 1398      3.034789e-03         0.002943
# 1626      3.034789e-03         0.003422
# 141       2.911424e-03         0.002957
# 172       2.837404e-03         0.003969
# 1090      2.763385e-03         0.007958
# 742       2.738712e-03         0.009405
# 1388      2.714039e-03         0.003445
# 753       2.615347e-03         0.004507
# 839       2.491981e-03         0.002434
# 217       2.417962e-03         0.001154
# 464       2.368616e-03         0.003197
# 1452      2.368616e-03         0.006840
# 720       2.343943e-03         0.001144
# 103       2.245250e-03         0.003044
# 1020      2.245250e-03         0.003536
# 299       2.195904e-03         0.006567
# 1413      2.195904e-03         0.002279
# 1292      1.825808e-03         0.004080
# 1322      1.727116e-03         0.001786
# 1404      1.702443e-03         0.001766
# 951       1.677770e-03         0.003774
# 1409      1.653096e-03         0.001698
# 171       1.603750e-03         0.002292
# 137       1.579077e-03         0.001845
# 1587      1.579077e-03         0.000931
# 1124      1.480385e-03         0.001911
# 127       1.480385e-03         0.001633
# 222       1.455712e-03         0.001126
# 56        1.406366e-03         0.002590
# 1605      1.307673e-03         0.001130
# 435       1.258327e-03         0.000915
# 678       1.208981e-03         0.001093
# 54        1.208981e-03         0.003486
# 102       1.159635e-03         0.004202
# 1793      1.134962e-03         0.001320
# 1295      1.085616e-03         0.008536
# 1258      1.085616e-03         0.012609
# 812       9.129040e-04         0.000992
# 1642      8.882309e-04         0.007398
# 795       8.882309e-04         0.004893
# 1317      8.635579e-04         0.003323
# 57        8.142117e-04         0.000767
# 1198      8.142117e-04         0.001819
# 1174      7.895386e-04         0.000726
# 538       7.648655e-04         0.004395
# 282       7.401925e-04         0.001808
# 1621      7.401925e-04         0.001725
# 642       7.155194e-04         0.001770
# 1183      6.661732e-04         0.002727
# 1254      6.415001e-04         0.001452
# 414       6.168270e-04         0.000657
# 605       6.168270e-04         0.001472
# 1071      5.921540e-04         0.002853
# 1195      5.674809e-04         0.001072
# 345       5.674809e-04         0.001281
# 1243      5.428078e-04         0.002982
# 702       5.428078e-04         0.001177
# 1271      3.947693e-04         0.000793
# 800       3.947693e-04         0.001477
# 809       3.700962e-04         0.000799
# 1474      3.454231e-04         0.001134
# 106       3.454231e-04         0.000838
# 1394      3.207501e-04         0.000445
# 86        2.467308e-04         0.002888
# 6         2.467308e-04         0.000398
# 571       2.467308e-04         0.002947
# 659       1.973847e-04         0.000503
# 1778      1.727116e-04         0.000503
# 1058      9.869233e-05         0.000316
# 1210      7.401925e-05         0.005689
# 1449      7.401925e-05         0.001269
# 518       7.401925e-05         0.000200
# 1686      1.480297e-17         0.000270
# 523       1.480297e-17         0.000191
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn.inspection import permutation_importance
result = permutation_importance(estimator, X_test, structured_y_test, n_repeats=15, random_state=random_state)
print(pd.DataFrame(
    {
        k: result[k]
        for k in (
            "importances_mean",
            "importances_std",
        )
    },
    index=X_test.columns,
).sort_values(by="importances_mean", ascending=False))