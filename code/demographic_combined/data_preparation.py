import pandas as pd

# combine isomiRs, miRNAs and demographic together 
def combine_isomiRs_miRNAs_demographic():
    # read isomiR_profile 
    isomiR_df = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/isomiR_profiles.csv")
    # read sample_sheet to get file_name and case_submitter_id 
    sample_isomiR_df = get_sample_details("../../data/clinical.cart.isomiRs/sample_sheet.tsv")
    isomiR_df = isomiR_df.merge(sample_isomiR_df, how='inner', on='file_name')
    isomiR_df = isomiR_df.drop(['file_name'], axis=1)
    isomiR_df.columns = [col + "_isomiR" if col != "case_submitter_id" else col for col in isomiR_df.columns]
    # read miRNA_profile 
    miRNA_df = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/miRNA_profiles.csv")
    # read sample_sheet to get file_name and case_submitter_id 
    sample_miRNA_df = get_sample_details("../../data/clinical.cart.miRNAs/sample_sheet.tsv")
    miRNA_df = miRNA_df.merge(sample_miRNA_df, how='inner', on='file_name')
    miRNA_df = miRNA_df.drop(['file_name'],axis=1)
    miRNA_df.columns = [col + "_miRNA" if col != "case_submitter_id" else col for col in miRNA_df.columns]
    # read demographic 
    demographic_df = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data3.csv")
    # combine isomiR_df with demographic_df
    combined_df = demographic_df.merge(isomiR_df, how='inner', on='case_submitter_id')
    # combined isomiR_df, demographic_df with miRNA_df
    combined_df = combined_df.merge(miRNA_df, how='inner', on='case_submitter_id')
    # drop case_submitter_id
    combined_df = combined_df.drop(['case_submitter_id'], axis=1)
    # export to csv
    combined_df.to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data4.csv", index=False)

def get_sample_details(file):
    """
    """
    # Read the sample metadata dataset
    sample_df = pd.read_csv(file, sep='\t')
    # Rename columns
    sample_df = sample_df.rename(columns={"File Name": "file_name", "Case ID": "case_submitter_id", "Sample Type": "sample_type"})
    # Retain Primary Tumor samples and Solid Tissue Normal only, remove Metastatic
    sample_df = sample_df[sample_df['sample_type'].isin(['Primary Tumor'])]
    # Retain file_name and case_submitter_id columns only 
    sample_df = sample_df[['file_name', 'case_submitter_id']]
    return sample_df

combine_isomiRs_miRNAs_demographic()