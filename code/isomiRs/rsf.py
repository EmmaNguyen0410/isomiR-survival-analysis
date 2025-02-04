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
raw_df = pd.read_csv("../../data/ml_inputs/raw_data.csv", index_col=False)
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
# skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 8)
# cv = list(skf.split(X_train, structured_y_train))
# hyperparams_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_features': [25, 50, 100],
#     'min_samples_leaf': [3, 10, 20]
# }
# grid_search_tree = GridSearchCV(RandomSurvivalForest(random_state=random_state), hyperparams_grid,cv=cv, verbose=1)
# grid_search_tree.fit(X_train, structured_y_train)
# best_params = grid_search_tree.best_params_
# # Best params 
# print(best_params)
# # Best score
# print(grid_search_tree.best_score_)


# {'max_features': 50, 'min_samples_leaf': 50, 'n_estimators': 200}
# 0.6065947270081234

#### Train model using best params #####
# n_estimators=200, max_features = 25, min_samples_leaf = 20
# 0.866665139611059, 0.6221864951768489
rsf = RandomSurvivalForest(
    n_estimators=200, max_features = 25, min_samples_leaf = 20, random_state=random_state
)
rsf.fit(X_train, structured_y_train)
# Score on train set 
print(rsf.score(X_train, structured_y_train))
# Score on test set 
print(rsf.score(X_test, structured_y_test))

###### Permutation to find important features #######
# 38676      5.016077e-03         0.003181
# 7293       2.829582e-03         0.002680
# 45746      2.722401e-03         0.001668
# 3978       2.658092e-03         0.001117
# 44531      2.508039e-03         0.001084
# 12494      2.400857e-03         0.001094
# 13929      2.357985e-03         0.001565
# 41064      2.336549e-03         0.001367
# 25638      2.165059e-03         0.001599
# 14493      2.143623e-03         0.001284
# 38197      1.993569e-03         0.001665
# 2481       1.886388e-03         0.000549
# 49606      1.886388e-03         0.001456
# 27294      1.843516e-03         0.000792
# 38299      1.800643e-03         0.001029
# 1505       1.800643e-03         0.000561
# 46965      1.800643e-03         0.002751
# 16862      1.779207e-03         0.000722
# 30542      1.779207e-03         0.001253
# 36446      1.693462e-03         0.001483
# 33925      1.564845e-03         0.000722
# 15112      1.500536e-03         0.000691
# 42056      1.500536e-03         0.000640
# 3523       1.500536e-03         0.000959
# 41285      1.479100e-03         0.001393
# 49775      1.457663e-03         0.001989
# 36160      1.457663e-03         0.000786
# 5009       1.436227e-03         0.001016
# 28204      1.414791e-03         0.000481
# 1890       1.393355e-03         0.000618
# 31524      1.371919e-03         0.001098
# 39805      1.329046e-03         0.000713
# 42360      1.286174e-03         0.000704
# 38546      1.264737e-03         0.001331
# 21477      1.264737e-03         0.000890
# 32441      1.221865e-03         0.000677
# 45654      1.200429e-03         0.000699
# 20865      1.157556e-03         0.001106
# 32136      1.114684e-03         0.001173
# 26111      1.093248e-03         0.000812
# 42070      1.071811e-03         0.000730
# 1736       1.071811e-03         0.000418
# 20818      1.071811e-03         0.000701
# 18831      1.071811e-03         0.001397
# 42067      1.007503e-03         0.000995
# 41569      9.646302e-04         0.000389
# 4411       9.646302e-04         0.000389
# 22705      9.431940e-04         0.000738
# 40510      9.431940e-04         0.000993
# 8753       9.431940e-04         0.000809
# 23452      9.431940e-04         0.000491
# 46735      9.217578e-04         0.000786
# 28564      9.217578e-04         0.003137
# 2984       9.217578e-04         0.000631
# 48801      9.003215e-04         0.000578
# 45922      8.788853e-04         0.000800
# 5304       8.788853e-04         0.000626
# 42807      8.574491e-04         0.000629
# 36849      8.574491e-04         0.000595
# 20462      8.360129e-04         0.001179
# 31461      8.360129e-04         0.001943
# 14112      8.360129e-04         0.000535
# 10451      8.360129e-04         0.000795
# 7286       8.145766e-04         0.000713
# 18259      8.145766e-04         0.001043
# 2421       8.145766e-04         0.000786
# 37443      8.145766e-04         0.002224
# 3638       8.145766e-04         0.000713
# 23612      8.145766e-04         0.000369
# 25167      7.931404e-04         0.001197
# 38041      7.931404e-04         0.000620
# 49138      7.717042e-04         0.000893
# 15226      7.717042e-04         0.000522
# 30968      7.288317e-04         0.000557
# 1583       7.073955e-04         0.000666
# 29192      7.073955e-04         0.000984
# 673        6.859593e-04         0.000453
# 718        6.859593e-04         0.000878
# 35595      6.859593e-04         0.002664
# 34715      6.859593e-04         0.000732
# 7704       6.859593e-04         0.001009
# 46828      6.859593e-04         0.000938
# 43660      6.859593e-04         0.000713
# 27666      6.859593e-04         0.000663
# 50106      6.859593e-04         0.000916
# 17285      6.645230e-04         0.001533
# 22307      6.645230e-04         0.000728
# 4513       6.645230e-04         0.000615
# 14737      6.645230e-04         0.000593
# 8190       6.645230e-04         0.000842
# 43514      6.430868e-04         0.000470
# 10265      6.430868e-04         0.000621
# 9629       6.430868e-04         0.000389
# 7384       6.430868e-04         0.000498
# 28315      6.216506e-04         0.001888
# 43798      6.216506e-04         0.000415
# 347        6.216506e-04         0.000491
# 7664       6.216506e-04         0.001435
# 45872      6.002144e-04         0.000854
# 6695       5.787781e-04         0.000528
# 34770      5.787781e-04         0.001420
# 37184      5.787781e-04         0.000716
# 9944       5.573419e-04         0.000626
# 32491      5.573419e-04         0.000476
# 45987      5.573419e-04         0.000491
# 5229       5.573419e-04         0.000765
# 33242      5.144695e-04         0.000672
# 49969      5.144695e-04         0.000467
# 39368      4.930332e-04         0.001403
# 31375      4.930332e-04         0.000523
# 13200      4.930332e-04         0.000496
# 9313       4.930332e-04         0.000510
# 23174      4.715970e-04         0.000778
# 21405      4.715970e-04         0.000597
# 50460      4.715970e-04         0.000231
# 3226       4.715970e-04         0.000329
# 49883      4.501608e-04         0.000672
# 39230      4.501608e-04         0.001427
# 34811      4.501608e-04         0.000467
# 1278       4.501608e-04         0.000436
# 29674      4.501608e-04         0.001312
# 41119      4.501608e-04         0.002970
# 46344      4.501608e-04         0.000368
# 6272       4.287245e-04         0.000973
# 29047      4.287245e-04         0.001543
# 44898      4.287245e-04         0.000465
# 45340      4.287245e-04         0.000739
# 27370      4.287245e-04         0.000303
# 3416       4.072883e-04         0.000476
# 34431      4.072883e-04         0.000783
# 48902      4.072883e-04         0.000709
# 44772      4.072883e-04         0.000380
# 25888      4.072883e-04         0.000248
# 16423      3.858521e-04         0.000697
# 38161      3.858521e-04         0.001109
# 36408      3.858521e-04         0.001187
# 31659      3.858521e-04         0.000553
# 29191      3.858521e-04         0.000375
# 40222      3.858521e-04         0.000336
# 47185      3.858521e-04         0.000487
# 33442      3.644159e-04         0.000778
# 9643       3.644159e-04         0.000967
# 30906      3.644159e-04         0.000369
# 27163      3.644159e-04         0.000482
# 15537      3.644159e-04         0.000421
# 14318      3.644159e-04         0.000561
# 3569       3.429796e-04         0.000834
# 47623      3.429796e-04         0.000569
# 28917      3.429796e-04         0.000398
# 15241      3.429796e-04         0.000476
# 36165      3.429796e-04         0.000321
# 6595       3.215434e-04         0.000761
# 283        3.215434e-04         0.000855
# 46366      3.215434e-04         0.000939
# 2509       3.215434e-04         0.000423
# 38563      3.215434e-04         0.000235
# 43918      3.215434e-04         0.000352
# 2845       3.001072e-04         0.000689
# 15495      3.001072e-04         0.000738
# 6888       3.001072e-04         0.000581
# 30861      3.001072e-04         0.000476
# 26306      3.001072e-04         0.000476
# 6061       3.001072e-04         0.000447
# 49663      3.001072e-04         0.000380
# 19825      3.001072e-04         0.000774
# 8029       3.001072e-04         0.000626
# 1617       3.001072e-04         0.000361
# 16942      3.001072e-04         0.000321
# 15803      2.786710e-04         0.000693
# 21291      2.786710e-04         0.000453
# 18408      2.786710e-04         0.000829
# 12624      2.786710e-04         0.000967
# 13014      2.786710e-04         0.000663
# 24226      2.786710e-04         0.000421
# 18212      2.786710e-04         0.000778
# 24811      2.572347e-04         0.000578
# 48143      2.572347e-04         0.000948
# 39399      2.572347e-04         0.000442
# 44323      2.572347e-04         0.000393
# 12587      2.572347e-04         0.000410
# 4749       2.572347e-04         0.000375
# 23072      2.572347e-04         0.000984
# 22139      2.572347e-04         0.000566
# 32877      2.357985e-04         0.001267
# 30626      2.357985e-04         0.000531
# 13145      2.357985e-04         0.000615
# 35606      2.357985e-04         0.000505
# 541        2.357985e-04         0.000615
# 18837      2.357985e-04         0.000431
# 43556      2.357985e-04         0.000921
# 19072      2.357985e-04         0.000299
# 17373      2.143623e-04         0.000994
# 1051       2.143623e-04         0.000681
# 50619      2.143623e-04         0.000640
# 25822      2.143623e-04         0.000836
# 48127      2.143623e-04         0.000507
# 36648      2.143623e-04         0.000507
# 27896      2.143623e-04         0.000571
# 32458      2.143623e-04         0.000418
# 50583      2.143623e-04         0.000418
# 19862      2.143623e-04         0.000583
# 35725      2.143623e-04         0.000254
# 41462      1.929260e-04         0.000368
# 14697      1.929260e-04         0.000196
# 42532      1.929260e-04         0.000257
# 25759      1.714898e-04         0.000878
# 10391      1.714898e-04         0.000437
# 44606      1.714898e-04         0.000259
# 22743      1.714898e-04         0.000510
# 39868      1.714898e-04         0.000259
# 28057      1.500536e-04         0.001009
# 36193      1.500536e-04         0.001225
# 13227      1.500536e-04         0.000453
# 16886      1.500536e-04         0.000804
# 2363       1.500536e-04         0.000421
# 32329      1.286174e-04         0.001106
# 5094       1.286174e-04         0.000608
# 14419      1.286174e-04         0.000386
# 43596      1.286174e-04         0.000759
# 8870       1.286174e-04         0.000861
# 23721      1.286174e-04         0.000328
# 1411       1.286174e-04         0.000283
# 45714      1.286174e-04         0.000158
# 1189       1.286174e-04         0.000158
# 33176      1.286174e-04         0.000386
# 13250      1.286174e-04         0.000257
# 6789       1.071811e-04         0.000618
# 20889      1.071811e-04         0.000465
# 27304      1.071811e-04         0.000559
# 2295       1.071811e-04         0.000225
# 45953      1.071811e-04         0.001327
# 40577      1.071811e-04         0.000254
# 49186      1.071811e-04         0.000192
# 31174      8.574491e-05         0.000979
# 12456      8.574491e-05         0.000518
# 50175      8.574491e-05         0.000518
# 1723       8.574491e-05         0.000882
# 3280       8.574491e-05         0.000321
# 42948      8.574491e-05         0.000380
# 15667      8.574491e-05         0.000380
# 10948      8.574491e-05         0.000342
# 37174      8.574491e-05         0.000800
# 29032      8.574491e-05         0.000699
# 16195      6.430868e-05         0.001839
# 32356      6.430868e-05         0.000677
# 35443      6.430868e-05         0.000473
# 48848      6.430868e-05         0.000375
# 13106      6.430868e-05         0.000514
# 42894      6.430868e-05         0.000375
# 36316      6.430868e-05         0.001077
# 5698       6.430868e-05         0.000210
# 6204       6.430868e-05         0.000487
# 37730      6.430868e-05         0.000473
# 8528       4.287245e-05         0.000574
# 46724      4.287245e-05         0.000421
# 14596      4.287245e-05         0.000404
# 15036      4.287245e-05         0.000284
# 8648       4.287245e-05         0.000109
# 1773       4.287245e-05         0.000109
# 3238       4.287245e-05         0.000199
# 6731       4.287245e-05         0.000437
# 33422      4.287245e-05         0.000510
# 50405      2.143623e-05         0.000637
# 41522      2.143623e-05         0.000342
# 35541      2.143623e-05         0.000248
# 9962       2.143623e-05         0.000184
# 46411      2.143623e-05         0.000080
# 32433      2.143623e-05         0.000321
# 23605      2.143623e-05         0.000275
# 21157      2.143623e-05         0.000342
# 18169      2.143623e-05         0.001278
# 23352      2.143623e-05         0.000826
# 20015      1.480297e-17         0.000587
# 41225      7.401487e-18         0.000551
# 16251      7.401487e-18         0.000909
from sklearn.inspection import permutation_importance
result = permutation_importance(rsf, X_test, structured_y_test, n_repeats=15, random_state=random_state)
fi = pd.DataFrame(
    {
        k: result[k]
        for k in (
            "importances_mean",
            "importances_std",
        )
    },
    index=X_test.columns,
).sort_values(by="importances_mean", ascending=False)
print(fi[fi['importances_mean'] > 0].index)