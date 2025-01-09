import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV, StratifiedKFold
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM, HingeLossSurvivalSVM, FastKernelSurvivalSVM, MinlipSurvivalAnalysis, NaiveSurvivalSVM
from sklearn import set_config
set_config(display="text")
from sklearn.model_selection import train_test_split

# Read data
raw_df = pd.read_csv("../../data/raw_data.csv", index_col=False)
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

#### Train model #####
param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'optimizer':['avltree', 'direct-count', 'PRSVM', 'rbtree', 'simple']}
skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = random_state)
cv = list(skf.split(X_train, structured_y_train))
kssvm = FastSurvivalSVM(max_iter=10000)
kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)
kgcv = kgcv.fit(X_train, structured_y_train)
print(kgcv.best_score_, kgcv.best_params_)
estimator = FastSurvivalSVM(max_iter=10000)
estimator.set_params(**kgcv.best_params_)
estimator.fit(X_train, structured_y_train)
print(score_survival_model(estimator, X_train, structured_y_train)) # 0.5892296767986807
print(score_survival_model(estimator, X_test, structured_y_test)) # 0.5929260450160772

### feature importance ######
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