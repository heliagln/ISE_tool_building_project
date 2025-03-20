from scipy.stats import ranksums
import pandas as pd


file_path_result_1 = "result_sample_1.csv"
file_path_result = "result_sample.csv"

df_result_1 = pd.read_csv(file_path_result_1)
df_result = pd.read_csv(file_path_result)

df_result_1 = df_result_1.drop(columns=['500 samples'])
df_result = df_result.drop(columns=['2000 samples'])

print(df_result_1)
print(df_result)

df = pd.concat([df_result_1,df_result], ignore_index=True)

rs = ranksums(df['1000 samples'], df['1500 samples'], alternative='greater')

print(rs)