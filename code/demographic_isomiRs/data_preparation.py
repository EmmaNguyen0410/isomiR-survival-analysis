import pandas as pd 

def create_ml_input():
    # read the combined input file 
    combined_input = pd.read_csv("../../data/ml_inputs/raw_data4.csv")

    # read the selected demographic features 
    selected_demographic = list(pd.read_csv("../../data/selected_features/demographic.csv")['feature'])

    # read the selected isomiR features
    selected_isomiRs = list(pd.read_csv("../../data/selected_features/isomiRs.csv")['feature'])
    selected_isomiRs = [str(id) for id in selected_isomiRs]
    # rename isomiR ids 
    selected_isomiRs = [id + "_isomiR" for id in selected_isomiRs]
    # keep the selected features only 
    combined_input = combined_input[selected_demographic + selected_isomiRs + ['status', 'survival_in_days']]   
    # export to csv 
    combined_input.to_csv("../../data/ml_inputs/demographic_isomiRs.csv", index=False)

create_ml_input()
