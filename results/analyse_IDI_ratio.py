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
    df = pd.DataFrame(columns=['IDI_baseline','time_baseline','IDI_tool','time_tool'])

    for name in info_dataset['name']:
        # Get the data for this dataset
        df_dataset = pd.read_csv(f'performance_comparison/{name}_results.csv')

        # extract p-values
        if name != 'communities_crime':
            # Wilcoxon signed rank test
            two_sided_res, two_sided_pvalue = wilcoxon(df_dataset['IDI_baseline'], df_dataset['IDI_tool'])
            print(f"two sided hypothesis for {name} : {two_sided_pvalue}")
            less_res, less_pvalue = wilcoxon(df_dataset['IDI_baseline'], df_dataset['IDI_tool'], alternative='less')
            print(f"less hypothesis for {name} : {less_pvalue}")
            greater_res, greater_pvalue = wilcoxon(df_dataset['IDI_baseline'], df_dataset['IDI_tool'], alternative='greater')
            print(f"greater hypothesis for {name} : {greater_pvalue}")

        # extract median 
        print(f"The median for the baseline for the IDI ratio for the dataset {name} : {np.median(df_dataset['IDI_baseline'])}")
        print(f"The median for the tool for the IDI ratio for the dataset {name} : {np.median(df_dataset['IDI_tool'])}")

        # figure 
        df_melted = (df_dataset.drop(columns=['time_baseline', 'time_tool'])).melt(var_name="method", value_name="IDI ratio")

        plt.figure(figsize=(8, 6))
        sns.boxplot(x="method", y="IDI ratio", data=df_melted, width=0.4, palette="Set2")
        sns.swarmplot(x="method", y="IDI ratio", data=df_melted, color="black", size=6, alpha=0.7)

        plt.title(f"Comparison of the IDI ratio as a function of the method used for the dataset {name}")
        plt.savefig(f'IDI_ratio_plt/figure_{name}.png')
        plt.close()


        # Concatenate the dataset's data
        df = pd.concat([df, df_dataset], ignore_index=True)
    
    # Wilcoxon signed rank test across all datasets
    two_sided_res, two_sided_pvalue = wilcoxon(df['IDI_baseline'], df['IDI_tool'])
    print(f"two sided hypothesis across datasets : {two_sided_pvalue}")
    less_res, less_pvalue = wilcoxon(df['IDI_baseline'], df['IDI_tool'], alternative='less')
    print(f"less hypothesis across datasets : {less_pvalue}")
    greater_res, greater_pvalue = wilcoxon(df['IDI_baseline'], df['IDI_tool'], alternative='greater')
    print(f"greater hypothesis across datasets : {greater_pvalue}")

    # extract median 
    print(f"The median for the baseline for the IDI ratio : {np.median(df['IDI_baseline'])}")
    print(f"The median for the tool for the IDI ratio : {np.median(df['IDI_tool'])}")
    

if __name__ == "__main__":
    main()