from sklearn import set_config
from sksurv.linear_model import CoxnetSurvivalAnalysis
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

######## Data processing ########
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

######## Choose penalty strength alpha ########
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold

coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.5, max_iter=1000))
coxnet_pipe.fit(X_train, structured_y_train)
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

########## Grid Search ############
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    cv=cv,
    error_score=0.5,
    n_jobs=1,
).fit(X_train, structured_y_train)
cv_results = pd.DataFrame(gcv.cv_results_)

####### Feature important of best model #######
best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs = pd.DataFrame(best_model.coef_, index=X_train.columns, columns=["coefficient"])

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print(f"Number of non-zero coefficients: {non_zero}")

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

print(coef_order)

_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
plt.show()

