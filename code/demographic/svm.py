import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV, StratifiedKFold
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM, HingeLossSurvivalSVM, FastKernelSurvivalSVM, MinlipSurvivalAnalysis, NaiveSurvivalSVM
from sklearn import set_config
set_config(display="text")
from sklearn.model_selection import train_test_split

# Read data
raw_df = pd.read_csv("../../data/ml_inputs/raw_data3.csv", index_col=False)
raw_df = raw_df.drop(['case_submitter_id'], axis = 1)
raw_df['survival_in_days'] = raw_df['survival_in_days'].astype(float)

# Preprocess raw data
random_state = 8
y_status = raw_df.pop('status').to_list()
y_survival_in_days = raw_df.pop('survival_in_days').to_list()
y = [(y_status[i], y_survival_in_days[i]) for i in range(len(raw_df))]
y = np.array(y, dtype=[('status', '?'), ('survival_in_days', '<f8')])
X_train, X_test, structured_y_train, structured_y_test = train_test_split(raw_df, y, test_size=0.2, random_state=random_state, stratify=y["status"])

y_cont = pd.qcut(structured_y_train['survival_in_days'], q=75, labels=False)
y_combined = [(structured_y_train['status'][i], y_cont[i]) for i in range(len(structured_y_train))]
y_combined = np.array(y_combined, dtype=[('status', '?'), ('survival_in_days', '<f8')])
skf = StratifiedKFold(n_splits=6, shuffle = True, random_state = 8)
cv = list(skf.split(X_train, y_combined))

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y["status"], y["survival_in_days"], prediction)
    return result[0]

#### 6 cvs scores #####
scores = []
for train_index, test_index in cv:
    X_train_data, X_test_data = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_data, y_test_data = structured_y_train[train_index], structured_y_train[test_index]
    # Fit model to train 
    estimator = MinlipSurvivalAnalysis(max_iter=10000)
    estimator.set_params(**{'alpha': 0.1, 'kernel': 'poly', 'solver': 'osqp'})
    estimator.fit(X_train_data, y_train_data)
    # Add test score to scores 
    scores.append(score_survival_model(estimator, X_test_data, y_test_data))
    # export scores to csv 
    pd.DataFrame({'score': scores}).to_csv("../../data/signed_ranks_test/demographic/svm.csv", index=False)


# Train model
# {'alpha': 0.1, 'kernel': 'poly', 'solver': 'osqp'}
# param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'solver':['ecos', 'osqp'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']}
# kssvm = MinlipSurvivalAnalysis(max_iter=10000)
# kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)
# kgcv = kgcv.fit(X_train, structured_y_train)
# print(kgcv.best_score_, kgcv.best_params_)
# estimator = MinlipSurvivalAnalysis(max_iter=10000)
# estimator.set_params(**kgcv.best_params_)
# estimator.fit(X_train, structured_y_train)
# print(score_survival_model(estimator, X_train, structured_y_train)) # 0.6983804311403998
# print(score_survival_model(estimator, X_test, structured_y_test)) # 0.585455764075067

##### feature importance
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

#age_at_index                                                0.059581   
#ajcc_pathologic_n                                           0.054131   
#ajcc_pathologic_stage                                       0.044233   
#ajcc_clinical_stage                                         0.015372   
#race_white                                                  0.007768   
#ajcc_clinical_n                                             0.004732   
#site_of_resection_or_biopsy_Mouth, NOS                      0.003978   
#tissue_or_organ_of_origin_Mouth, NOS                        0.003978   
#race_black or african american                              0.003837   
#prior_malignancy_yes                                        0.003225   
#tissue_or_organ_of_origin_Floor of mouth, NOS               0.002801   
#site_of_resection_or_biopsy_Floor of mouth, NOS             0.002801   
#gender_male                                                 0.002519   
#site_of_resection_or_biopsy_Tonsil, NOS                     0.002472   
#tissue_or_organ_of_origin_Tonsil, NOS                       0.002472   
#site_of_resection_or_biopsy_Larynx, NOS                     0.002142   
#tissue_or_organ_of_origin_Larynx, NOS                       0.002142   
#prior_malignancy_no                                         0.001742   
#site_of_resection_or_biopsy_Tongue, NOS                     0.001365   
#tissue_or_organ_of_origin_Tongue, NOS                       0.001365   
#site_of_resection_or_biopsy_Gum, NOS                        0.001224   
#tissue_or_organ_of_origin_Gum, NOS                          0.001224   
#ajcc_clinical_m                                             0.000400   
#tissue_or_organ_of_origin_Hypopharynx, NOS                  0.000188   
#site_of_resection_or_biopsy_Hypopharynx, NOS                0.000188   
#ajcc_clinical_t                                             0.000024