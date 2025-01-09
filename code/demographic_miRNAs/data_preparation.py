import pandas as pd

def create_ml_input():
    # read the combined input file 
    combined_input = pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/raw_data4.csv")

    # read the selected demographic features 
    selected_demographic = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/selected_features/demographic.csv")['feature'])

    # read the selected miRNA features
    selected_miRNAs = list(pd.read_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/selected_features/miRNAs.csv")['feature'])
    selected_miRNAs = [str(id) for id in selected_miRNAs]
    # rename miRNA ids 
    selected_miRNAs = [id + "_miRNA" for id in selected_miRNAs]
    # keep the selected features only 
    combined_input = combined_input[selected_demographic + selected_miRNAs + ['status', 'survival_in_days']]   
    # export to csv 
    combined_input.to_csv("/Users/nguyenthao/Desktop/UTS/TranLab/research_project/data/ml_inputs/demographic_miRNAs.csv", index=False)

create_ml_input()
