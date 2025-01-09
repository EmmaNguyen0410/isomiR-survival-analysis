import pandas as pd 
import os
from collections import Counter


data_path = "../../data"

def get_survival_details():
    """
    Create a file patient_id, surival time 
    Days to death = number of days that passed from initial diagnosis to death => death
    Days to last follow up = number of days that passed from initial diagnosis to last visit => alive
    """
    # Read clinical dataset
    clinical_df = pd.read_csv("../../data/clinical.cart.miRNAs/clinical.tsv", sep='\t')
    # Retain case_submitter_id, days_to_death, vital_status, days_to_last_follow_up from the clinical dataset.
    clinical_df = clinical_df[["case_submitter_id", "days_to_death", "vital_status", "days_to_last_follow_up"]]
    # Add status and survival_in_days columns. 
    clinical_df[['status', 'survival_in_days']] = clinical_df.apply(lambda r: set_survival(r), axis = 1)
    # Drop unused columns
    clinical_df = clinical_df.drop(['vital_status', 'days_to_death', 'days_to_last_follow_up'], axis = 1)
    # Drop duplicates 
    clinical_df = clinical_df.drop_duplicates(ignore_index = True)
    return clinical_df

def set_survival(row):
    """
    Set values for status and survival_in_days columns for each patient based on their vital_status
    """
    if (row['vital_status'] == "Alive"):
        return pd.Series([False, row['days_to_last_follow_up']])
    else:
        return pd.Series([True, row['days_to_death']])
       
def get_sample_details():
    """
    """
    # Read the sample metadata dataset
    sample_df = pd.read_csv("../../data/clinical.cart.miRNAs/sample_sheet.tsv", sep='\t')
    # Rename columns
    sample_df = sample_df.rename(columns={"File Name": "file_name", "Case ID": "case_submitter_id", "Sample Type": "sample_type"})
    # Retain Primary Tumor samples only
    sample_df = sample_df[sample_df['sample_type'].isin(['Primary Tumor', 'Solid Tissue Normal'])]
    # Retain file_name and case_submitter_id columns only 
    sample_df = sample_df[['file_name', 'case_submitter_id', 'sample_type']]
    return sample_df

def get_miR_details(file_path):
    """
    There are 4801
    """
    miRBase_details_df = pd.read_csv(file_path, sep='\t', header=None, skiprows=[0,1,2,3,4,5,6,7,8,9,10,11,12], names=["chr", "unknown1", "type", "start", "end", "unknown2", "strand", "unknown3", "details"])
    miRBase_details_df[['id', 'name', 'derives_from']] = miRBase_details_df.apply(lambda r: extract_miR_info(r), axis = 1)
    return miRBase_details_df

def extract_miR_info(row):
    details = row['details'].split(';')
    if (row['type'] == "miRNA_primary_transcript"):
        return pd.Series([details[1].split('=')[1], details[2].split('=')[1], ''])
    else:
        return pd.Series([details[1].split('=')[1], details[2].split('=')[1], details[3].split('=')[1]])

def get_miR_seqs(file_path):
    """
    1917 hsa sequences
    """
    miR_seq_dicts = []
    current_stem_loop_id = ""
    current_stem_loop_name = ""
    current_miR_seq = ""
    with open(file_path, 'r') as f:
        for line in f: 
            if line[0] == '>':
                if current_stem_loop_name != ""  and current_stem_loop_id != "":
                    miR_seq_dicts.append({"id": current_stem_loop_id, "name": current_stem_loop_name, "seq": current_miR_seq})
                
                seq_info = line[1:-1].split(" ")
                current_stem_loop_id = seq_info[1]
                current_stem_loop_name = seq_info[0]
                current_miR_seq = ""
            else:
                current_miR_seq += line.strip()
    miR_seq_dicts.append({"id": current_stem_loop_id, "name": current_stem_loop_name, "seq": current_miR_seq})
    return pd.DataFrame.from_records(miR_seq_dicts)
    
def curate_miRNA_quantification(miR_seq_df):
    # Path to folder 
    raw_miRNAs_quantification_path = "../../data/gdc_download_20240526_080506.757472"

    # List of mirnas.quantification.txt files 
    sub_folders = os.listdir(raw_miRNAs_quantification_path)
    # sub_folders.remove(".DS_Store")
    sub_folders.remove("MANIFEST.txt")
    files = []
    for sub_folder in sub_folders:
        for file in os.listdir(f"{raw_miRNAs_quantification_path}/{sub_folder}"):
            if file != "annotations.txt":
                files.append(file)

    for i in range(len(files)): 
        print(f"{raw_miRNAs_quantification_path}/{sub_folders[i]}/{files[i]}")
        print(f"{data_path}/curated_miRNAs_quantification_v1/{files[i]}.csv")
        # Raw quantification of miRNAs from TCGA
        patient_miRNAs_df = pd.read_csv(f"{raw_miRNAs_quantification_path}/{sub_folders[i]}/{files[i]}", sep='\t')
        # Rename column miRNA_ID to name
        patient_miRNAs_df.rename(columns={'miRNA_ID': 'name'}, inplace=True)
        # Add miRNA_seq 
        patient_miRNAs_df = patient_miRNAs_df.merge(miR_seq_df, how='inner', on='name')
        # group miRNA_seq with the same seq, aggregate read count 
        curated_miRNA_df = patient_miRNAs_df.groupby(['seq']).agg({'read_count': 'sum'})
        # reset index to be miRNA_seq column
        curated_miRNA_df = curated_miRNA_df.reset_index()
        curated_miRNA_df = curated_miRNA_df.rename(columns={'seq': 'miRNA_seq'})
        # remove empty seq 
        curated_miRNA_df = curated_miRNA_df[curated_miRNA_df["miRNA_seq"] != ""]
        # export to file 
        curated_miRNA_df.to_csv(f"{data_path}/curated_miRNAs_quantification_v1/{files[i]}.csv", index=False)

def annotate_miRNA():
    # unique miRNA sequences 
    miRNA_seqs = set()
    
    # read all files
    curated_miRNAs_quantification_path = "../../data/curated_miRNAs_quantification_v1"
    # Loop through all curated files. For each file, get the list of miRNA_seq
    for file in os.listdir(curated_miRNAs_quantification_path):
        # Read the curated quantification of miRNAs
        curated_miRNAs_df = pd.read_csv(f"{curated_miRNAs_quantification_path}/{file}")
        # Add miRNA sequences to set of unique miRNA sequences
        miRNA_seqs.update(curated_miRNAs_df["miRNA_seq"].to_list())

    # Export unique miRNAs to excel with id 
    annotated_miRNAs = pd.DataFrame({"miRNA_ID": range(1, len(miRNA_seqs) + 1), "miRNA_seq": list(miRNA_seqs)})
    annotated_miRNAs.to_csv(f"../../data/annotation_guide/annotated_miRNAs.csv", index=False)
    
def add_miRNA_ID_to_curated_miRNA_quantification():
    # read annoted miRNAs 
    annotated_miRNAs = pd.read_csv("../../data/annotation_guide/annotated_miRNAs.csv")
    # read all files
    curated_miRNAs_quantification_path = "../../data/curated_miRNAs_quantification_v1"
    # Loop through all curated files. For each file, add miRNA ID
    for file in os.listdir(curated_miRNAs_quantification_path):
        # Curated quantification of miRNAs
        curated_miRNAs_df = pd.read_csv(f"{curated_miRNAs_quantification_path}/{file}")   
        # Add miRNA sequences to set of unique miRNA sequences
        curated_miRNAs_df = curated_miRNAs_df.merge(annotated_miRNAs, how="inner", on="miRNA_seq")
        # Export to csv 
        curated_miRNAs_df.to_csv(f"{data_path}/curated_miRNAs_quantification_v2/{file}", index=False)

def filter_miRNA_ids_appear_in_n_samples(ratio):
    sample_miRNAs = []
    # read all files
    curated_miRNAs_quantification_path = "../../data/curated_miRNAs_quantification_v2"
    # Loop through all curated files. For each file, get the list of miRNA_seq
    for file in os.listdir(curated_miRNAs_quantification_path):
        # Read the curated quantification of miRNAs
        curated_miRNAs_df = pd.read_csv(f"{curated_miRNAs_quantification_path}/{file}")
        if file in ['289bb3f2-f828-4ce4-8b9a-bb4c19f5b2d0.mirbase21.mirnas.quantification.txt.csv', '89997d58-4819-41a0-8f35-8f519cb5e483.mirbase21.mirnas.quantification.txt.csv']:
            continue
        sample_miRNAs.extend(list(curated_miRNAs_df['miRNA_seq']))
    miRNAs_freq = Counter(sample_miRNAs)
    # filter miRNAs appears in all files 
    dominant_miRNAs = [k for k,v in miRNAs_freq.items() if v >= 567*ratio]
    # Read annotated miRNAs
    annotated_miRNAs = pd.read_csv("../../data/annotation_guide/annotated_miRNAs.csv")
    dominant_miRNA_ids = annotated_miRNAs[annotated_miRNAs['miRNA_seq'].isin(dominant_miRNAs)]['miRNA_ID']
    return set(dominant_miRNA_ids)
    
def set_up_normal_primary_tumour():
    sample_df = get_sample_details()
    normal_sample_df = sample_df[sample_df['sample_type'] == 'Solid Tissue Normal']
    primary_tumour_sample_df = sample_df[sample_df['sample_type'] == 'Primary Tumor']
    merged = primary_tumour_sample_df.merge(normal_sample_df, how='inner', on='case_submitter_id')

    curated_miRNAs_quantification_path = "../../data/curated_miRNAs_quantification_v2"

    # copy each file to folder corresponding to sample type and rename to be case id
    for _, r in merged.iterrows():
        # read primary tissue file
        primary_df = pd.read_csv(f"{curated_miRNAs_quantification_path}/{r['file_name_x']}.csv")
        # only retain miRNA_ID, read_count and reoroder columns
        primary_df = primary_df[['miRNA_ID', 'read_count']]
        # save to primary folder with new name 
        primary_df.to_csv(f"../../data/deseq2/primary2/{r['case_submitter_id']}_primary.csv", index=False, header=False)
        # read normal tissue file
        normal_df = pd.read_csv(f"{curated_miRNAs_quantification_path}/{r['file_name_y']}.csv")
        # only retain miRNA_ID, read_count and reoroder columns
        normal_df = normal_df[['miRNA_ID', 'read_count']]
        # save to normal folder with new name 
        normal_df.to_csv(f"../../data/deseq2/normal2/{r['case_submitter_id']}_normal.csv", index=False, header=False)

def filter_de_dominant_miRNAs():
    de_miRNAs = pd.read_csv("../../data/deseq2/summary2.tabular", delim_whitespace=True, header=None)
    de_miRNA_ids = set(de_miRNAs[0])
    dominant_miRNA_ids = filter_miRNA_ids_appear_in_n_samples(0.8)
    de_dominant_miRNA_ids = dominant_miRNA_ids.intersection(de_miRNA_ids)
    de_dominant_miRNA_ids_df = pd.DataFrame({'miRNA_ID': list(de_dominant_miRNA_ids)})
    de_dominant_miRNA_ids_df.to_csv("../../data/annotation_guide/de_dominant_miRNA_ids.csv", index=False)
    curated_miRNAs_quantification_path = "../../data/curated_miRNAs_quantification_v2"

    for file in os.listdir(curated_miRNAs_quantification_path):
        # Read the curated quantification of miRNAs
        curated_miRNAs_df = pd.read_csv(f"{curated_miRNAs_quantification_path}/{file}")   
        curated_miRNAs_df = curated_miRNAs_df[curated_miRNAs_df['miRNA_ID'].isin(de_dominant_miRNA_ids)]    
        curated_miRNAs_df = curated_miRNAs_df[['read_count', 'miRNA_ID']]
        curated_miRNAs_df.to_csv(f"../../data/curated_miRNAs_quantification_v3/{file}", index=False)

def create_miRNA_profiles(): 
    # read de miRNA_ID 
    de_dominant_miRNAs = pd.read_csv("../../data/annotation_guide/de_dominant_miRNA_ids.csv")
    de_dominant_miRNA_ids = list(de_dominant_miRNAs['miRNA_ID'])
    combined_curated_miRNAs = pd.DataFrame(columns = de_dominant_miRNA_ids + ['file_name'])

    curated_miRNAs_quantification_path = "../../data/curated_miRNAs_quantification_v3"

    for file in os.listdir(curated_miRNAs_quantification_path):
        # Curated quantification of miRNAs
        curated_miRNAs_df = pd.read_csv(f"{curated_miRNAs_quantification_path}/{file}")
        # Dict to store miRNAs' counts 
        curated_miRNAs_dict = {}        
        for _, r in curated_miRNAs_df.iterrows():
            curated_miRNAs_dict[int(r['miRNA_ID'])] = int(r['read_count'])
        # Add new row to combined_curated_miRNAs
        curated_miRNAs_dict['file_name'] = file.replace(".csv", "")
        combined_curated_miRNAs = combined_curated_miRNAs.append(curated_miRNAs_dict, ignore_index=True)
    
    combined_curated_miRNAs.to_csv("../../data/ml_inputs/miRNA_profiles.csv", index=False)

def create_ml_input(combined_curated_miRNAs, clinical_sample_df):
    # Merge ml_input with clinical_sample_df to get status and survival time 
    ml_input = combined_curated_miRNAs.merge(clinical_sample_df, how = 'inner', on = "file_name")
    # Drop file_name, case_submitter_id columns
    ml_input = ml_input.drop(['file_name', 'case_submitter_id', 'sample_type'], axis = 1)
    # Export to csv
    ml_input.to_csv("../../data/ml_inputs/raw_data2.csv", index=False)

def select_feature():
    cox_features = [  
        '1642', '1733', '1071', '678', '1239', '560', '1388', '964', '440',
        '324', '1034', '1293', '1400', '1297', '605', '326', '1100', '1802',
        '1049', '1740', '231', '1322'
    ]
    # get selected rsf features
    rsf_features = list(pd.read_csv("../../data/selected_features/miRNAs_raw_rsf.csv", sep=' ')['miRNA_ID'])
    rsf_features = [str(id) for id in rsf_features]

    # get selected svm features 
    svm_features = list(pd.read_csv("../../data/selected_features/miRNAs_raw_svm.csv", sep=' ')['miRNA_ID'])
    svm_features = [str(id) for id in svm_features]

    # combine 3 list 
    all_features = cox_features + rsf_features + svm_features

    #### Approach 1 #####
    # count the occurence of each feature
    freq = Counter(all_features)
    # select features appearing in at least 2 models 
    selected_features = [k for k,v in freq.items() if v >= 2]

    #### Approach 2 #####
    # selected_features = list(set(all_features))

    pd.DataFrame({'feature': selected_features}).to_csv("../../data/selected_features/miRNAs.csv", index=False)

## main ##
# survival_details_df = get_survival_details()
# sample_details_df = get_sample_details()
# sample_details_df = sample_details_df[sample_details_df['sample_type'] == 'Primary Tumor']
# sample_survival_details_df = survival_details_df.merge(sample_details_df, how = 'inner', on = "case_submitter_id")

# miR_seq_df = get_miR_seqs("../../data/annotation_guide/hairpin.fa")
# curate_miRNA_quantification(miR_seq_df)
# annotate_miRNA()
# add_miRNA_ID_to_curated_miRNA_quantification()
# set_up_normal_primary_tumour()
# filter_de_dominant_miRNAs()
# create_miRNA_profiles()
# combined_curated_miRNAs = pd.read_csv("../../data/ml_inputs/miRNA_profiles.csv")
# create_ml_input(combined_curated_miRNAs, sample_survival_details_df)
select_feature()


