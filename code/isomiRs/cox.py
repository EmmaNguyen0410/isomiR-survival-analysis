from sklearn import set_config
from sksurv.linear_model import CoxnetSurvivalAnalysis
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Read data
raw_df = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data.csv", index_col=False)
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

# Choose penalty strength alphaimport warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.5, max_iter=1000))
coxnet_pipe.fit(X_train, structured_y_train)
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    cv=cv,
    error_score=0.5,
    n_jobs=1,
).fit(X_train, structured_y_train)
cv_results = pd.DataFrame(gcv.cv_results_)

# alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
# mean = cv_results.mean_test_score
# std = cv_results.std_test_score

# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(alphas, mean)
# ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
# ax.set_xscale("log")
# ax.set_ylabel("concordance index")
# ax.set_xlabel("alpha")
# ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
# ax.axhline(0.5, color="grey", linestyle="--")
# ax.grid(True)
# plt.show()

# ####### Feature important of best model #######
# '4324', '48737', '18723', '40510', '7656', '49997', '42807', '28204', '39080', '47850', '8125', '1278', '38895', '4626', '18133', '5633', '27881', '44026', '38197', '35448', '28564'

best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs = pd.DataFrame(best_model.coef_, index=X_train.columns, columns=["coefficient"])

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print(f"Number of non-zero coefficients: {non_zero}")

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
print(coef_order)
plt.show()

# ####### Train best model  #######
coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
coxnet_pred.set_params(**gcv.best_params_)
coxnet_pred.fit(X_train, structured_y_train)
print(coxnet_pred.score(X_train, structured_y_train)) # 0.6810820716036374
print(coxnet_pred.score(X_test, structured_y_test)) # 0.6086816720257234

# https://github.com/sebp/scikit-survival/issues/41
