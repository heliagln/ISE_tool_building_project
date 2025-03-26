import pandas as pd
import ast
import numpy as np


def main():
    # Load information on dataset
    info_dataset = pd.read_csv('../info_datasets.csv')
    info_dataset['sensitive_attributes'] = (info_dataset['sensitive_attributes']).apply(ast.literal_eval)

    # Concatenate all data
    df = pd.DataFrame(columns=['IDI_baseline','time_baseline','IDI_tool','time_tool'])

    for name in info_dataset['name']:
        # Get the data for this dataset
        df_dataset = pd.read_csv(f'performance_comparison/{name}_results.csv')

        # extract median 
        print(f"The median for the baseline for the runtime for the dataset {name} : {np.median(df_dataset['time_baseline'])}")
        print(f"The median for the tool for the runtime for the dataset {name} : {np.median(df_dataset['time_tool'])}")

        # Concatenate the dataset's data
        df = pd.concat([df, df_dataset], ignore_index=True)

    # extract median 
    print(f"The median for the baseline for the runtime : {np.median(df['time_baseline'])}")
    print(f"The median for the tool for the runtime : {np.median(df['time_tool'])}")
    

if __name__ == "__main__":
    main()