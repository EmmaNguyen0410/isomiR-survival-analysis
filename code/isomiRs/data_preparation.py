import pandas as pd 
import os
from collections import Counter


data_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data"


def get_survival_details():
    """
    Create a file patient_id, surival time 
    Days to death = number of days that passed from initial diagnosis to death => death
    Days to last follow up = number of days that passed from initial diagnosis to last visit => alive
    """
    # Read clinical dataset
    clinical_df = pd.read_csv("../../data/clinical.cart.isomiRs/clinical.tsv", sep='\t')
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
    sample_df = pd.read_csv("../../data/clinical.cart.isomiRs/sample_sheet.tsv", sep='\t')
    # Rename columns
    sample_df = sample_df.rename(columns={"File Name": "file_name", "Case ID": "case_submitter_id", "Sample Type": "sample_type"})
    # Retain Primary Tumor samples and Solid Tissue Normal only, remove Metastatic
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

def get_isomiR_seq(row):
    isomiR_coords = row["isoform_coords"].split(":")[2].split("-")
    isomiR_start = int(isomiR_coords[0]) 
    isomiR_end = int(isomiR_coords[1]) - 1 
    stemloop_start = int(row["start"])
    stemloop_end = int(row["end"])
    start_diff = isomiR_start - stemloop_start
    end_diff = stemloop_end - isomiR_end
    if row["strand"] == "+":
        return pd.Series([row["seq"][start_diff:len(row["seq"]) - end_diff]])
    else:
        return pd.Series([row["seq"][end_diff:len(row["seq"]) - start_diff]])
    
def curate_isomiR_quantification(miR_stemloop_seq_details_df):
    # Path to folder 
    raw_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/gdc_download_20240217_032435.276336"

    # List of isoforms.quantification.txt files 
    sub_folders = os.listdir(raw_isomiRs_quantification_path)
    sub_folders.remove(".DS_Store")
    sub_folders.remove("MANIFEST.txt")
    files = []
    for sub_folder in sub_folders:
        for file in os.listdir(f"{raw_isomiRs_quantification_path}/{sub_folder}"):
            if file != "annotations.txt":
                files.append(file)

    for i in range(len(files)): 
        print(f"{raw_isomiRs_quantification_path}/{sub_folders[i]}/{files[i]}")
        print(f"{data_path}/curated_isomiRs_quantification_v2/{files[i]}.csv")
        # Raw quantification of isomiRs from TCGA
        patient_isomiRs_df = pd.read_csv(f"{raw_isomiRs_quantification_path}/{sub_folders[i]}/{files[i]}", sep='\t')
        # rename column miRNA_ID to name
        patient_isomiRs_df.rename(columns={'miRNA_ID': 'name'}, inplace=True)
        # Merge patient_isomiRs_df with miR_stemloop_seq_details_df by mirID, name
        merged = patient_isomiRs_df.merge(miR_stemloop_seq_details_df, how='inner', on="name")
        # add isomiR_seq for each isomiR
        merged[["isomiR_seq"]] = merged.apply(lambda r: get_isomiR_seq(r), axis = 1)
        # group isomiRs with the same isomiR_seq, aggregate read count 
        curated_isomiR_df = merged.groupby(['isomiR_seq']).agg({'read_count': 'sum'})
        # reset index to be isomiR_seq column
        curated_isomiR_df = curated_isomiR_df.reset_index()
        curated_isomiR_df = curated_isomiR_df.rename(columns={'index': 'isomiR_seq'})
        # remove empty seq 
        curated_isomiR_df = curated_isomiR_df[curated_isomiR_df["isomiR_seq"] != ""]
        # export to file 
        curated_isomiR_df.to_csv(f"{data_path}/curated_isomiRs_quantification_v1/{files[i]}.csv", index=False)

def annotate_isomiR():
    # unique isomiR sequences 
    isomiR_seqs = set()
    
    # read all files
    curated_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v1"
    # Loop through all curated files. For each file, get the list of isomiR_seq
    for file in os.listdir(curated_isomiRs_quantification_path):
        # Read the curated quantification of isomiRs
        curated_isomiRs_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{file}")
        # Add isomiR sequences to set of unique isomiR sequences
        isomiR_seqs.update(curated_isomiRs_df["isomiR_seq"].to_list())

    # Export unique isomiRs to excel with id 
    annotated_isomiRs = pd.DataFrame({"isomiR_ID": range(1, len(isomiR_seqs) + 1), "isomiR_seq": list(isomiR_seqs)})
    annotated_isomiRs.to_csv(f"/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/annotation_guide/annotated_isomiRs.csv", index=False)
    
def add_isomiR_ID_to_curated_isomiR_quantification():
    # read annoted isomiRs 
    annotated_isomiRs = pd.read_csv("../../data/annotation_guide/annotated_isomiRs.csv")
    # read all files
    curated_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v1"
    # Loop through all curated files. For each file, add isomiR ID
    for file in os.listdir(curated_isomiRs_quantification_path):
        # Curated quantification of isomiRs
        curated_isomiRs_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{file}")   
        # Add isomiR sequences to set of unique isomiR sequences
        curated_isomiRs_df = curated_isomiRs_df.merge(annotated_isomiRs, how="inner", on="isomiR_seq")
        # Export to csv 
        curated_isomiRs_df.to_csv(f"{data_path}/curated_isomiRs_quantification_v2/{file}", index=False)

def filter_isomiR_ids_appear_in_n_samples(ratio):
    sample_isomiRs = []
    # read all files
    curated_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v2"
    # Loop through all curated files. For each file, get the list of isomiR_seq
    for file in os.listdir(curated_isomiRs_quantification_path):
        # Read the curated quantification of isomiRs
        curated_isomiRs_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{file}")
        if file in ['89997d58-4819-41a0-8f35-8f519cb5e483.mirbase21.isoforms.quantification.txt.csv', '289bb3f2-f828-4ce4-8b9a-bb4c19f5b2d0.mirbase21.isoforms.quantification.txt.csv']:
            continue
        sample_isomiRs.extend(list(curated_isomiRs_df['isomiR_seq']))
    isomiRs_freq = Counter(sample_isomiRs)
    # filter isomiRs appears in all files 
    dominant_isomiRs = [k for k,v in isomiRs_freq.items() if v >= 567*ratio]
    # Read annotated isomiRs
    annotated_isomiRs = pd.read_csv("../../data/annotation_guide/annotated_isomiRs.csv")
    dominant_isomiR_ids = annotated_isomiRs[annotated_isomiRs['isomiR_seq'].isin(dominant_isomiRs)]['isomiR_ID']
    return set(dominant_isomiR_ids)
    
def set_up_normal_primary_tumour():
    sample_df = get_sample_details()
    normal_sample_df = sample_df[sample_df['sample_type'] == 'Solid Tissue Normal']
    primary_tumour_sample_df = sample_df[sample_df['sample_type'] == 'Primary Tumor']
    merged = primary_tumour_sample_df.merge(normal_sample_df, how='inner', on='case_submitter_id')
    curated_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v3"

    # copy each file to folder corresponding to sample type and rename to be case id
    for _, r in merged.iterrows():
        # read primary tissue file
        primary_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{r['file_name_x']}.csv")
        # only retain isomiR_ID, read_count and reoroder columns
        primary_df = primary_df[['isomiR_ID', 'read_count']]
        # save to primary folder with new name 
        primary_df.to_csv(f"/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/deseq2/primary/{r['case_submitter_id']}_primary.csv", index=False, header=False)
        # read normal tissue file
        normal_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{r['file_name_y']}.csv")
        # only retain isomiR_ID, read_count and reoroder columns
        normal_df = normal_df[['isomiR_ID', 'read_count']]
        # save to normal folder with new name 
        normal_df.to_csv(f"/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/deseq2/normal/{r['case_submitter_id']}_normal.csv", index=False, header=False)

def populate_all_isomiRs():
    """
    Make sure that each has all isomiRs
    """
    # Read annotated isomiRs
    annotated_isomiRs = pd.read_csv("../../data/annotation_guide/annotated_isomiRs.csv")
    # Get all isomiR ids 
    annotated_isomiR_ids = set(annotated_isomiRs['isomiR_ID'])
    # Read all files
    curated_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v2"
    # For each curated isomiR file
    for file in os.listdir(curated_isomiRs_quantification_path):
        # Curated quantification of isomiRs
        curated_isomiRs_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{file}")  
        # isomiR ids of the file
        curated_isomiR_ids = set(curated_isomiRs_df['isomiR_ID'])
        # annotated isomiRs that are not in the file 
        missing_isomiR_ids = annotated_isomiR_ids - curated_isomiR_ids
        # create a dataframe with missing_isomiR_ids 
        missing_isomiRs_df = pd.DataFrame([{'read_count': 0, 'isomiR_ID': isomiR_id} for isomiR_id in missing_isomiR_ids])
        # add missing isomiRs to current curated isomiR df 
        curated_isomiRs_df = curated_isomiRs_df[['read_count', 'isomiR_ID']]
        curated_isomiRs_df = pd.concat([curated_isomiRs_df, missing_isomiRs_df])
        # sort dataframe by isomiR_ID
        curated_isomiRs_df = curated_isomiRs_df.sort_values(by="isomiR_ID", ascending=True)
        # export to v3 
        curated_isomiRs_df.to_csv(f"/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v3/{file}", index=False)

def filter_de_dominant_isomiRs():
    de_isomiRs = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/deseq2/summary.tabular", delim_whitespace=True, header=None)
    de_isomiR_ids = set(de_isomiRs[0])
    dominant_isomiR_ids = filter_isomiR_ids_appear_in_n_samples(0.8)
    de_dominant_isomiR_ids = dominant_isomiR_ids.intersection(de_isomiR_ids)
    #de_dominant_isomiR_ids_df = pd.DataFrame({'isomiR_ID': list(de_dominant_isomiR_ids)})
    #de_dominant_isomiR_ids_df.to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/annotation_guide/de_dominant_isomiR_ids.csv", index=False)
    curated_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v3"

    for file in os.listdir(curated_isomiRs_quantification_path):
        # Read the curated quantification of isomiRs
        curated_isomiRs_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{file}")   
        curated_isomiRs_df = curated_isomiRs_df[curated_isomiRs_df['isomiR_ID'].isin(de_dominant_isomiR_ids)]    
        curated_isomiRs_df.to_csv(f"/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v4/{file}", index=False)

def create_isomiR_profiles(): 
    # read de isomiR_IDs 
    de_dominant_isomiRs = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/annotation_guide/de_dominant_isomiR_ids.csv")
    de_dominant_isomiR_ids = list(de_dominant_isomiRs['isomiR_ID'])
    combined_curated_isomiRs = pd.DataFrame(columns = de_dominant_isomiR_ids + ['file_name'])

    curated_isomiRs_quantification_path = "/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/curated_isomiRs_quantification_v4"

    for file in os.listdir(curated_isomiRs_quantification_path):
        # Curated quantification of isomiRs
        curated_isomiRs_df = pd.read_csv(f"{curated_isomiRs_quantification_path}/{file}")
        # Dict to store isomiRs' counts 
        curated_isomiRs_dict = {}        
        for _, r in curated_isomiRs_df.iterrows():
            curated_isomiRs_dict[int(r['isomiR_ID'])] = int(r['read_count'])
        # Add new row to combined_curated_isomiRs
        curated_isomiRs_dict['file_name'] = file.replace(".csv", "")
        combined_curated_isomiRs = combined_curated_isomiRs.append(curated_isomiRs_dict, ignore_index=True)
    
    combined_curated_isomiRs.to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/isomiR_profiles.csv", index=False)

def create_ml_input(combined_curated_isomiRs, clinical_sample_df):
    print(combined_curated_isomiRs)
    print(clinical_sample_df)
    # Merge ml_input with clinical_sample_df to get status and survival time 
    ml_input = combined_curated_isomiRs.merge(clinical_sample_df, how = 'inner', on = "file_name")
    # Drop file_name, case_submitter_id columns
    ml_input = ml_input.drop(['file_name', 'case_submitter_id', 'sample_type'], axis = 1)
    # Export to csv
    ml_input.to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data.csv", index=False)

def select_feature():
    cox_features = [  
        '4324', '48737', '18723', '40510', '7656', '49997', '42807', '28204', '39080', '47850', '8125', '1278', '38895', '4626', '18133', '5633', '27881', '44026', '38197', '35448', '28564'
    ]
    # get selected rsf features
    rsf_features = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/selected_features/isomiRs_raw_rsf.csv", sep=' ')['isomiR_ID'])
    rsf_features = [str(id) for id in rsf_features]

    # get selected svm features 
    svm_features = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/selected_features/isomiRs_raw_svm.csv", sep=' ')['isomiR_ID'])
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

    pd.DataFrame({'feature': selected_features}).to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/selected_features/isomiRs.csv", index=False)

## main ##
# survival_details_df = get_survival_details()
# sample_details_df = get_sample_details()
# sample_details_df = sample_details_df[sample_details_df['sample_type'] == 'Primary Tumor']
# sample_survival_details_df = survival_detsails_df.merge(sample_details_df, how = 'inner', on = "case_submitter_id")


# miR_seq_df = get_miR_seqs("../../data/annotation_guide/hairpin.fa")
# miR_details_df = get_miR_details("../../data/annotation_guide/hsa.gff3")
# miR_stemloop_details_df = miR_details_df[miR_details_df['type'] == 'miRNA_primary_transcript']
# miR_stemloop_seq_details_df = miR_details_df.merge(miR_seq_df, how='inner', on=["id", "name"])
# curate_isomiR_quantification(miR_stemloop_seq_details_df)
# annotate_isomiR()
# add_isomiR_ID_to_curated_isomiR_quantification()
# populate_all_isomiRs()
# set_up_normal_primary_tumour()
# filter_de_dominant_isomiRs()
# create_isomiR_profiles()
#combined_curated_isomiRs = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/isomiR_profiles.csv")
#create_ml_input(combined_curated_isomiRs, sample_survival_details_df)
select_feature()

