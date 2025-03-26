import pandas as pd
import ast
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Load information on dataset
    info_dataset = pd.read_csv('../../info_datasets.csv')
    info_dataset['sensitive_attributes'] = (info_dataset['sensitive_attributes']).apply(ast.literal_eval)

    # Concatenate all data
    df = pd.DataFrame(columns=['IDI_second_baseline','IDI_second_tool'])

    for name in info_dataset['name']:
        # Get the data for this dataset
        df_dataset = pd.read_csv(f'performance_comparison/{name}_results.csv')

        # Compute IDI per second
        IDI_second_baseline = np.array(1000*(np.divide(df_dataset['IDI_baseline'], df_dataset['time_baseline'])))
        IDI_second_tool = np.array(1000*(np.divide(df_dataset['IDI_tool'], df_dataset['time_tool'])))

        # extract p-values
        if name != 'communities_crime':
            # Wilcoxon signed rank test
            two_sided_res, two_sided_pvalue = wilcoxon(IDI_second_baseline, IDI_second_tool)
            print(f"two sided hypothesis for {name} : {two_sided_pvalue}")
            less_res, less_pvalue = wilcoxon(IDI_second_baseline, IDI_second_tool, alternative='less')
            print(f"less hypothesis for {name} : {less_pvalue}")
            greater_res, greater_pvalue = wilcoxon(IDI_second_baseline, IDI_second_tool, alternative='greater')
            print(f"greater hypothesis for {name} : {greater_pvalue}")

        # extract median 
        print(f"The median for the baseline for the IDI per second for the dataset {name} : {np.median(IDI_second_baseline)}")
        print(f"The median for the tool for the IDI per second for the dataset {name} : {np.median(IDI_second_tool)}")

        # form dataframe
        df_second_dataset = pd.DataFrame({'IDI_second_baseline' : IDI_second_baseline ,'IDI_second_tool' : IDI_second_tool})

        # figure
        df_melted = df_second_dataset.melt(var_name="method", value_name="IDI per second")

        plt.figure(figsize=(8, 6))
        sns.boxplot(x="method", y="IDI per second", data=df_melted, width=0.4, palette="Set2")
        sns.swarmplot(x="method", y="IDI per second", data=df_melted, color="black", size=6, alpha=0.7)

        plt.title(f"Comparison of the IDI per second as a function of the method used for the dataset {name}")
        plt.savefig(f'IDI_second_plt/figure_{name}.png')
        plt.close()

        # Concatenate the dataset's data
        df = pd.concat([df, df_second_dataset], ignore_index=True)
    
    # Wilcoxon signed rank test across all datasets
    two_sided_res, two_sided_pvalue = wilcoxon(df['IDI_second_baseline'], df['IDI_second_tool'])
    print(f"two sided hypothesis across datasets : {two_sided_pvalue}")
    less_res, less_pvalue = wilcoxon(df['IDI_second_baseline'], df['IDI_second_tool'], alternative='less')
    print(f"less hypothesis across datasets : {less_pvalue}")
    greater_res, greater_pvalue = wilcoxon(df['IDI_second_baseline'], df['IDI_second_tool'], alternative='greater')
    print(f"greater hypothesis across datasets : {greater_pvalue}")

    # extract median 
    print(f"The median for the baseline for the IDI per second : {np.median(df['IDI_second_baseline'])}")
    print(f"The median for the tool for the IDI per second : {np.median(df['IDI_second_tool'])}")

    

if __name__ == "__main__":
    main()