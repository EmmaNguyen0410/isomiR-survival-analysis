from sklearn import set_config
from sksurv.linear_model import CoxnetSurvivalAnalysis
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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

y_cont = pd.qcut(structured_y_train['survival_in_days'], q=75, labels=False)
y_combined = [(structured_y_train['status'][i], y_cont[i]) for i in range(len(structured_y_train))]
y_combined = np.array(y_combined, dtype=[('status', '?'), ('survival_in_days', '<f8')])
skf = StratifiedKFold(n_splits=6, shuffle = True, random_state = 8)
cv = list(skf.split(X_train, y_combined))

#### 6 cvs scores #####
scores = []
for train_index, test_index in cv:
    X_train_data, X_test_data = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_data, y_test_data = structured_y_train[train_index], structured_y_train[test_index]
    # Fit model to train 
    estimator = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
    estimator.set_params(**{'coxnetsurvivalanalysis__alphas': [0.08397149001726541]})
    estimator.fit(X_train_data, y_train_data)
    # Add test score to scores 
    scores.append(estimator.score(X_test_data, y_test_data))
    # export scores to csv 
    pd.DataFrame({'score': scores}).to_csv("../../data/signed_ranks_test/demographic/cox.csv", index=False)

# Choose penalty strength alpha
coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.5, max_iter=1000))
coxnet_pipe.fit(X_train, structured_y_train)
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    cv=cv,
    error_score=0.5,
    n_jobs=1,
).fit(X_train, structured_y_train)
cv_results = pd.DataFrame(gcv.cv_results_)

alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)
plt.show()

###### Feature important of best model ######
# tissue_or_organ_of_origin_Ventral surface of tongue, NOS
# gender_male
# site_of_resection_or_biopsy_Ventral surface of tongue, NOS
# gender_female
# tissue_or_organ_of_origin_Floor of mouth, NOS
# site_of_resection_or_biopsy_Floor of mouth, NOS
# age_at_index
# ajcc_pathologic_n
# ajcc_pathologic_stage
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
plt.show()


# ####### Train best model  #######
coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
print(gcv.best_params_)  # {'coxnetsurvivalanalysis__alphas': [0.08397149001726541]}
coxnet_pred.set_params(**gcv.best_params_)
coxnet_pred.fit(X_train, structured_y_train)
print("Train: ", coxnet_pred.score(X_train, structured_y_train)) # 0.6251982575672959
print("Test: ", coxnet_pred.score(X_test, structured_y_test)) # 0.6028820375335121