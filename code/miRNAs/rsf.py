from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

###### Data processing ########
# Read data
raw_df = pd.read_csv("../../data/ml_inputs/raw_data2.csv", index_col=False)
raw_df = raw_df[raw_df['survival_in_days'] != "'--"]
raw_df['survival_in_days'] = raw_df['survival_in_days'].astype(float)

# Preprocess raw data
random_state = 8
y_status = raw_df.pop('status').to_list()
y_survival_in_days = raw_df.pop('survival_in_days').to_list()
y = [(y_status[i], y_survival_in_days[i]) for i in range(len(raw_df))]
X_train, X_test, y_train, y_test = train_test_split(raw_df, y, test_size=0.2, random_state=random_state)

structured_y_train = np.array(y_train, dtype=[('status', '?'), ('survival_in_days', '<f8')])
structured_y_test = np.array(y_test, dtype=[('status', '?'), ('survival_in_days', '<f8')])

########## Grid Search ############
skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = random_state)
cv = list(skf.split(X_train, structured_y_train))
hyperparams_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': [25, 50, 100],
    'min_samples_leaf': [3, 10, 20]
}
grid_search_tree = GridSearchCV(RandomSurvivalForest(random_state=random_state), hyperparams_grid,cv=cv, verbose=1)
grid_search_tree.fit(X_train, structured_y_train)
best_params = grid_search_tree.best_params_
# Best params: {'max_features': 50, 'min_samples_leaf': 20, 'n_estimators': 50}
print(best_params)
# Best score: 0.6179907220541011
print(grid_search_tree.best_score_)

#### Train model using best params #####
rsf = RandomSurvivalForest(random_state=random_state)
rsf.set_params(**best_params)
rsf.fit(X_train, structured_y_train)
# Score on train set: 0.8387609213661636 
print(rsf.score(X_train, structured_y_train))
# Score on test set: 0.5965951147298297
print(rsf.score(X_test, structured_y_test))

###### Predict survival function #####
# surv = rsf.predict_survival_function(X_test, return_array = True)
# for i, s in enumerate(surv):
#     print(i, '===', s)
#     plt.step(rsf.unique_times_, s, where='post', label=str(i))
# plt.ylabel("Survival probability")
# plt.xlabel("Time in days")
# plt.legend()
# plt.grid(True)


###### Predict cumulative hazard function #####
# surv = rsf.predict_cumulative_hazard_function(X_test, return_array=True)

# for i, s in enumerate(surv):
#     plt.step(rsf.unique_times_, s, where="post", label=str(i))
# plt.ylabel("Cumulative hazard")
# plt.xlabel("Time in days")
# plt.legend()
# plt.grid(True)

###### Permutation to find important features #######
# 1168          0.022847         0.022215
# 686           0.013694         0.010711
# 207           0.009302         0.010261
# 1322          0.008043         0.010572
# 222           0.007525         0.004471
# 1618          0.006119         0.004913
# 369           0.005255         0.003943
# 1400          0.005083         0.004044
# 1645          0.004910         0.003222
# 513           0.004836         0.001889
# 681           0.004639         0.001782
# 398           0.004565         0.011742
# 244           0.004540         0.003816
# 1271          0.004491         0.001522
# 1775          0.004416         0.004321
# 912           0.003972         0.001669
# 141           0.003701         0.002659
# 643           0.003676         0.004911
# 1019          0.003627         0.004028
# 1075          0.003578         0.002752
# 299           0.003528         0.003534
# 776           0.003430         0.002214
# 1297          0.003306         0.003441
# 231           0.003183         0.001528
# 652           0.003133         0.001792
# 338           0.003010         0.002620
# 613           0.002961         0.002905
# 331           0.002911         0.001177
# 1781          0.002837         0.002096
# 806           0.002640         0.002495
# 1210          0.002615         0.002475
# 440           0.002541         0.005025
# 1034          0.002517         0.002274
# 1306          0.002344         0.003559
# 1200          0.002245         0.003366
# 1293          0.002196         0.001717
# 1043          0.002171         0.001404
# 455           0.002171         0.001676
# 783           0.002073         0.000944
# 742           0.002023         0.004618
# 1361          0.001850         0.001767
# 1740          0.001776         0.001431
# 1546          0.001776         0.001291
# 697           0.001776         0.002826
# 345           0.001776         0.000730
# 605           0.001727         0.002292
# 1626          0.001702         0.000934
# 102           0.001678         0.001448
# 1600          0.001653         0.002128
# 663           0.001604         0.005606
# 172           0.001480         0.000752
# 698           0.001431         0.000739
# 1550          0.001357         0.003570
# 834           0.001332         0.001410
# 414           0.001283         0.002075
# 405           0.001283         0.000646
# 1568          0.001234         0.001811
# 941           0.001209         0.002605
# 920           0.001110         0.000740
# 894           0.001086         0.005816
# 809           0.001086         0.001285
# 468           0.001061         0.000843
# 1563          0.001012         0.001033
# 678           0.000987         0.001428
# 215           0.000962         0.002305
# 1802          0.000962         0.000809
# 1199          0.000938         0.001972
# 1077          0.000938         0.001146
# 378           0.000888         0.003668
# 1369          0.000864         0.003122
# 192           0.000864         0.000481
# 162           0.000814         0.001066
# 1323          0.000790         0.000895
# 1292          0.000765         0.001641
# 54            0.000740         0.000689
# 560           0.000716         0.002275
# 80            0.000716         0.000496
# 1335          0.000666         0.001339
# 36            0.000642         0.001024
# 816           0.000617         0.000259
# 1067          0.000592         0.000645
# 1624          0.000592         0.001192
# 379           0.000567         0.000485
# 1277          0.000543         0.001528
# 1407          0.000518         0.000423
# 1225          0.000518         0.003177
# 842           0.000518         0.000401
# 800           0.000493         0.000913
# 286           0.000469         0.001414
# 795           0.000469         0.003973
# 1100          0.000469         0.002139
# 1196          0.000469         0.001158
# 1090          0.000444         0.001589
# 1702          0.000444         0.000878
# 1605          0.000419         0.001990
# 703           0.000395         0.001060
# 642           0.000395         0.000931
# 465           0.000370         0.000331
# 1108          0.000345         0.002239
# 1301          0.000321         0.000503
# 1522          0.000321         0.002592
# 522           0.000296         0.000336
# 1458          0.000271         0.001647
# 761           0.000247         0.001460
# 561           0.000247         0.001087
# 70            0.000173         0.001238
# 127           0.000123         0.000174
# 965           0.000123         0.000933
# 1621          0.000099         0.000458
# 762           0.000099         0.000164
# 727           0.000074         0.001004
# 380           0.000049         0.001953
# 1591          0.000049         0.001442
# 464           0.000049         0.000895
from sklearn.inspection import permutation_importance
result = permutation_importance(rsf, X_test, structured_y_test, n_repeats=15, random_state=random_state)
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
