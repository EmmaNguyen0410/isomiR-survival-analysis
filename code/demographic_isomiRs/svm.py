import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV, StratifiedKFold
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM, HingeLossSurvivalSVM, FastKernelSurvivalSVM, MinlipSurvivalAnalysis, NaiveSurvivalSVM
from sklearn import set_config
set_config(display="text")
from sklearn.model_selection import train_test_split

# Read data
raw_df = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/demographic_isomiRs.csv", index_col=False)
raw_df = raw_df[raw_df['survival_in_days'] != "'--"]
raw_df['survival_in_days'] = raw_df['survival_in_days'].astype(float)

# Preprocess raw data
random_state = 8
y_status = raw_df.pop('status').to_list()
y_survival_in_days = raw_df.pop('survival_in_days').to_list()
y = [(y_status[i], y_survival_in_days[i]) for i in range(len(raw_df))]
y = np.array(y, dtype=[('status', '?'), ('survival_in_days', '<f8')])
raw_df = raw_df.astype(float)
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
    estimator.set_params(**{'alpha': 10, 'kernel': 'linear', 'solver': 'osqp'})
    estimator.fit(X_train_data, y_train_data)
    # Add test score to scores 
    scores.append(score_survival_model(estimator, X_test_data, y_test_data))
    # export scores to csv 
    pd.DataFrame({'score': scores}).to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/signed_ranks_test/demographic_isomiRs/svm.csv", index=False)

# # {'alpha': 10, 'kernel': 'linear', 'solver': 'osqp'}
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
print(score_survival_model(estimator, X_train, structured_y_train)) # 0.6228750139618006
print(score_survival_model(estimator, X_test, structured_y_test)) # 0.6591823056300268