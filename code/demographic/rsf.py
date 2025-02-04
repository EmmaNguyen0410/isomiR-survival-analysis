from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

######## Data processing ########
# Read data
raw_df = pd.read_csv("../../data/ml_inputs/raw_data3.csv", index_col=False)
raw_df = raw_df.drop(['case_submitter_id'], axis=1)
raw_df['survival_in_days'] = raw_df['survival_in_days'].astype(float)

# Preprocess raw data
random_state = 8
y_status = raw_df.pop('status').to_list()
y_survival_in_days = raw_df.pop('survival_in_days').to_list()
y = [(y_status[i], y_survival_in_days[i]) for i in range(len(raw_df))]
y = np.array(y, dtype=[('status', '?'), ('survival_in_days', '<f8')])
X_train, X_test, structured_y_train, structured_y_test = train_test_split(raw_df, y, test_size=0.2, random_state=random_state, stratify=y["status"])

##### 6 cvs scores #####
y_cont = pd.qcut(structured_y_train['survival_in_days'], q=75, labels=False)
y_combined = [(structured_y_train['status'][i], y_cont[i]) for i in range(len(structured_y_train))]
y_combined = np.array(y_combined, dtype=[('status', '?'), ('survival_in_days', '<f8')])
skf = StratifiedKFold(n_splits=6, shuffle = True, random_state = 8)
cv = list(skf.split(X_train, y_combined))
scores = []
for train_index, test_index in cv:
    X_train_data, X_test_data = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_data, y_test_data = structured_y_train[train_index], structured_y_train[test_index]
    # Fit model to train 
    estimator = RandomSurvivalForest(
        n_estimators=50, max_features = 10, min_samples_leaf= 10, random_state=random_state
    )
    estimator.fit(X_train_data, y_train_data)
    # Add test score to scores 
    scores.append(estimator.score(X_test_data, y_test_data))
    # export scores to csv 
    pd.DataFrame({'score': scores}).to_csv("../../data/signed_ranks_test/demographic/rsf.csv", index=False)


########## Grid Search ############
y_cont = pd.qcut(structured_y_train['survival_in_days'], q=75, labels=False)
y_combined = [(structured_y_train['status'][i], y_cont[i]) for i in range(len(structured_y_train))]
y_combined = np.array(y_combined, dtype=[('status', '?'), ('survival_in_days', '<f8')])
skf = StratifiedKFold(n_splits=6, shuffle = True, random_state = 8)
cv = list(skf.split(X_train, y_combined))

hyperparams_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': [5, 10, 20, 50],
    'min_samples_leaf': [3, 10, 20]
}

grid_search_tree = GridSearchCV(RandomSurvivalForest(random_state=random_state), hyperparams_grid,cv=cv, verbose=1)
grid_search_tree.fit(X_train, structured_y_train)
best_params = grid_search_tree.best_params_
# Best params 
print(best_params)
# Best score
print(grid_search_tree.best_score_)

#### Train model using best params #####
rsf = RandomSurvivalForest(
    n_estimators=50, max_features = 10, min_samples_leaf= 10, random_state=random_state
)
rsf.fit(X_train, structured_y_train)
# Score on train set 
print(rsf.score(X_train, structured_y_train)) # 0.7528984697866636
# Score on test set 
print(rsf.score(X_test, structured_y_test)) # 0.6462801608579088

###### Permutation to find important features #######
# ajcc_pathologic_n                                           0.098505   
# age_at_index                                                0.066643   
# ajcc_pathologic_t                                           0.015207   
# ajcc_clinical_stage                                         0.010664   
# tissue_or_organ_of_origin_Floor of mouth, NOS               0.009487   
# gender_male                                                 0.007486   
# site_of_resection_or_biopsy_Floor of mouth, NOS             0.005815   
# gender_female                                               0.005014   
# ajcc_clinical_t                                             0.004802   
# site_of_resection_or_biopsy_Overlapping lesion ...          0.004355   
# tissue_or_organ_of_origin_Larynx, NOS                       0.003814   
# race_white                                                  0.003319   
# tissue_or_organ_of_origin_Overlapping lesion of...          0.003131   
# site_of_resection_or_biopsy_Larynx, NOS                     0.002895   
# race_black or african american                              0.001412   
# site_of_resection_or_biopsy_Tonsil, NOS                     0.000353   
# tissue_or_organ_of_origin_Mouth, NOS                        0.000282   
# race_asian                                                  0.000235   
# site_of_resection_or_biopsy_Cheek mucosa                    0.000141   
# site_of_resection_or_biopsy_Hypopharynx, NOS                0.000094   
# tissue_or_organ_of_origin_Hypopharynx, NOS                  0.000094   
# tissue_or_organ_of_origin_Oropharynx, NOS                   0.000024   
# site_of_resection_or_biopsy_Mouth, NOS                      0.000024  
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