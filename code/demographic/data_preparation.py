import pandas as pd
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# get patient clinical 
def get_clinical_details():
    """
    """
    # Read clinical dataset
    clinical_df = pd.read_csv("../../data/clinical.cart.isomiRs/clinical.tsv", sep='\t')
    # Retain useful columns
    clinical_df = clinical_df[["case_submitter_id", "age_at_index", "gender", "race", "ajcc_clinical_m", "ajcc_clinical_n", "ajcc_clinical_stage", "ajcc_clinical_t", "ajcc_pathologic_n", "ajcc_pathologic_stage", "ajcc_pathologic_t", "prior_malignancy", "site_of_resection_or_biopsy", "tissue_or_organ_of_origin", "days_to_death", "vital_status", "days_to_last_follow_up"]]
    # Get max day of following up 
    max_days_to_last_follow_up = clinical_df['days_to_last_follow_up']
    max_days_to_last_follow_up = max(max_days_to_last_follow_up[max_days_to_last_follow_up != "'--"].astype(float))
    # Add status and survival_in_days columns. 
    clinical_df[['status', 'survival_in_days']] = clinical_df.apply(lambda r: set_survival(r, max_days_to_last_follow_up), axis = 1)
    # Drop patient having solid tissue normal only
    clinical_df = clinical_df[clinical_df['case_submitter_id'] != 'TCGA-WA-A7GZ']
    # Drop unused columns
    clinical_df = clinical_df.drop(['vital_status', 'days_to_death', 'days_to_last_follow_up'], axis = 1)
    # Drop duplicates 
    clinical_df = clinical_df.drop_duplicates(ignore_index = True)
    # Change datatype of columns 
    clinical_df = clinical_df.astype({
        'gender': str,
        'race': str,
        'ajcc_clinical_m': str,
        'ajcc_clinical_n': str,
        'ajcc_clinical_stage': str,
        'ajcc_clinical_t': str,
        'ajcc_pathologic_n': str,
        'ajcc_pathologic_stage': str,
        'ajcc_pathologic_t': str,
        'prior_malignancy': str,
        'site_of_resection_or_biopsy': str,
        'tissue_or_organ_of_origin': str
    })
    # Fill missing values of age_at_index with mean value 
    age_at_index_mean = clinical_df[clinical_df['age_at_index'] != "'--"]['age_at_index'].astype(int).mean()
    clinical_df.loc[clinical_df['age_at_index'] == "'--", 'age_at_index'] = round(age_at_index_mean)
    # Fill missing values of race, ajcc_clinical_m, ajcc_clinical_n, ajcc_clinical_stage, ajcc_clinical_t, ajcc_pathologic_n, ajcc_pathologic_stage, ajcc_pathologic_t with most frequent values
    for col in ['race', 'ajcc_clinical_m', 'ajcc_clinical_n', 'ajcc_clinical_stage', 'ajcc_clinical_t', 'ajcc_pathologic_n', 'ajcc_pathologic_stage', 'ajcc_pathologic_t']:
        # most frequent value 
        frequent_value = ""
        # highest count
        highest_count = 0
        # find most frequent value of each feature 
        freq = Counter(list(clinical_df[col]))
        for k,v in freq.items():
            if v > highest_count:
                frequent_value = k
                highest_count = v
        # replace missing values with most frequent value 
        clinical_df[col] = clinical_df[col].apply(lambda r: frequent_value if r == "'--" or r == "not reported" else r)

    # Export demographic data to csv
    clinical_df.to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/demographic_only_profiles.csv", index=False)

def set_survival(row, max_days_to_last_follow_up):
    """
    Set values for status and survival_in_days columns for each patient based on their vital_status
    If patient is alive without having days_to_last_follow_up, set days_to_last_follow_up to be the maximum day
    If patient is dead without having days_to_death, set it to be days_to_death
    """
    if (row['vital_status'] == "Alive" and row['days_to_last_follow_up'] != "'--"):
        return pd.Series([False, row['days_to_last_follow_up']])
    elif (row['vital_status'] == "Alive" and row['days_to_last_follow_up'] == "'--"):
        return pd.Series([False, max_days_to_last_follow_up])
    elif (row['vital_status'] == "Dead" and row['days_to_death'] != "'--"):
        return pd.Series([True, row['days_to_death']]) 
    else:
        return pd.Series([True, row['days_to_last_follow_up']])

def encode_categorical_features():
    clinical_df = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/demographic_only_profiles.csv")
    # gender, race, prior_malignancy: onehot 
    clinical_df = pd.get_dummies(clinical_df, columns=['gender', 'race', 'prior_malignancy', 'site_of_resection_or_biopsy', 'tissue_or_organ_of_origin'])
    # ajcc_clinical_m: ordinal
    clinical_df['ajcc_clinical_m'] = OrdinalEncoder(categories=[['MX', 'M0', 'M1']]).fit_transform(clinical_df[['ajcc_clinical_m']])
    # ajcc_clinical_n: ordinal
    clinical_df['ajcc_clinical_n'] = OrdinalEncoder(categories=[['NX', 'N0', 'N1', 'N2', 'N2a', 'N2b', 'N2c', 'N3']]).fit_transform(clinical_df[['ajcc_clinical_n']])
    # ajcc_clinical_stage: ordinal
    clinical_df['ajcc_clinical_stage'] = OrdinalEncoder(categories=[['Stage I', 'Stage II', 'Stage III', 'Stage IVA', 'Stage IVB', 'Stage IVC']]).fit_transform(clinical_df[['ajcc_clinical_stage']])
    # ajcc_clinical_t: ordinal
    clinical_df['ajcc_clinical_t'] = OrdinalEncoder(categories=[['TX', 'T1', 'T2', 'T3', 'T4', 'T4a', 'T4b']]).fit_transform(clinical_df[['ajcc_clinical_t']])
    # ajcc_pathologic_n: ordinal
    clinical_df['ajcc_pathologic_n'] = OrdinalEncoder(categories=[['NX', 'N0', 'N1', 'N2', 'N2a', 'N2b', 'N2c', 'N3']]).fit_transform(clinical_df[['ajcc_pathologic_n']])
    # ajcc_pathologic_stage: ordinal
    clinical_df['ajcc_pathologic_stage'] = OrdinalEncoder(categories=[['Stage I', 'Stage II', 'Stage III', 'Stage IVA', 'Stage IVB', 'Stage IVC']]).fit_transform(clinical_df[['ajcc_pathologic_stage']])
    # ajcc_pathologic_t: ordinal
    clinical_df['ajcc_pathologic_t'] = OrdinalEncoder(categories=[['TX', 'T0', 'T1', 'T2', 'T3', 'T4', 'T4a', 'T4b']]).fit_transform(clinical_df[['ajcc_pathologic_t']])
    print(clinical_df)
    clinical_df.to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data3.csv", index=False)

def select_feature():
    cox_features = [  
        'tissue_or_organ_of_origin_Ventral surface of tongue, NOS',
        'gender_male',
        'site_of_resection_or_biopsy_Ventral surface of tongue, NOS',
        'gender_female',
        'tissue_or_organ_of_origin_Floor of mouth, NOS',
        'site_of_resection_or_biopsy_Floor of mouth, NOS',
        'age_at_index',
        'ajcc_pathologic_n',
        'ajcc_pathologic_stage']
    rsf_features = [
        'ajcc_pathologic_n',
        'age_at_index',
        'ajcc_pathologic_t',
        'ajcc_clinical_stage',
        'tissue_or_organ_of_origin_Floor of mouth, NOS',
        'gender_male', 
        'site_of_resection_or_biopsy_Floor of mouth, NOS',
        'gender_female',
        'ajcc_clinical_t',
        'site_of_resection_or_biopsy_Overlapping lesion of lip, oral cavity and pharynx',
        'tissue_or_organ_of_origin_Larynx, NOS',
        'race_white',
        'tissue_or_organ_of_origin_Overlapping lesion of lip, oral cavity and pharynx',
        'site_of_resection_or_biopsy_Larynx, NOS', 
        'race_black or african american',
        'site_of_resection_or_biopsy_Tonsil, NOS',
        'tissue_or_organ_of_origin_Mouth, NOS',
        'race_asian',
        'site_of_resection_or_biopsy_Cheek mucosa',
        'site_of_resection_or_biopsy_Hypopharynx, NOS',
        'tissue_or_organ_of_origin_Hypopharynx, NOS',
        'tissue_or_organ_of_origin_Oropharynx, NOS',
        'site_of_resection_or_biopsy_Mouth, NOS'
    ]
    svm_features = [
        'age_at_index',
        'ajcc_pathologic_n',
        'ajcc_pathologic_stage',
        'ajcc_clinical_stage',
        'race_white',
        'ajcc_clinical_n',
        'site_of_resection_or_biopsy_Mouth, NOS',
        'tissue_or_organ_of_origin_Mouth, NOS',
        'race_black or african american',
        'prior_malignancy_yes',
        'tissue_or_organ_of_origin_Floor of mouth, NOS', 
        'site_of_resection_or_biopsy_Floor of mouth, NOS',
        'gender_male',
        'site_of_resection_or_biopsy_Tonsil, NOS',
        'tissue_or_organ_of_origin_Tonsil, NOS',
        'site_of_resection_or_biopsy_Larynx, NOS',
        'tissue_or_organ_of_origin_Larynx, NOS',
        'prior_malignancy_no',
        'site_of_resection_or_biopsy_Tongue, NOS',
        'tissue_or_organ_of_origin_Tongue, NOS', 
        'site_of_resection_or_biopsy_Gum, NOS',
        'tissue_or_organ_of_origin_Gum, NOS',
        'ajcc_clinical_m',
        'tissue_or_organ_of_origin_Hypopharynx, NOS',
        'site_of_resection_or_biopsy_Hypopharynx, NOS',
        'ajcc_clinical_t'
    ]
    # combine 3 list 
    all_features = cox_features + rsf_features + svm_features
    
    #### Approach 1 ####
    # count the occurence of each feature
    freq = Counter(all_features)
    # select features appearing in at least 2 models 
    # selected_features = [k for k,v in freq.items() if v >= 2]

    #### Approach 2 ####
    # selected_features = list(set(all_features))

    #### Approach 3 ####
    demographic_profile = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data3.csv")
    demographic_profile = demographic_profile.drop(['status', 'survival_in_days'], axis=1)
    selected_features = list(demographic_profile.columns)

    pd.DataFrame({'feature': selected_features}).to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/selected_features/demographic.csv", index=False)

###### main ######
get_clinical_details()
encode_categorical_features()
select_feature()