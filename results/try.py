import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Import baseline and tool
chemin_methodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(chemin_methodes)
from lab4_solution import calculate_idi_ratio_baseline
from tool import calculate_idi_ratio_tool

datasets = pd.DataFrame({
    'name' : ['adult', 'compas_cleaned', 'law_school_cleaned', 'kdd_cleaned', 'dutch', 'credit', 'greman_cleaned'],
    'sensitive_attributes' : [['gender', 'race', 'age'], ['Sex', 'Race'], ['male', 'race'], ['sex', 'race'], ['sex', 'age'], ['SEX', 'EDUCATION', 'MARRIAGE'], ['PersonStatusSex','AgeInYears']],
    'target_label' : ['Class-label', 'Recidivism', 'pass_bar', 'income', 'occupation', 'class', 'CREDITRATING']
})

print(datasets)


def load_and_preprocess_data(row):
    file_path = '../dataset/processed_' + row['name'] + '.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
    model_path = '../DNN/model_processed_'+ row['name'] + '.h5'  # Model path
    df = pd.read_csv(file_path)
    # Splitting the dataset into features and target
    target_column = row['target_label']# 'income'  # Modify the target column name if necessary
    X = df.drop(columns=[target_column])  # Features (drop the target column)
    y = df[target_column]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_test = X_test.astype('float64')
    model = keras.models.load_model(model_path)

    # Define sensitive and non-sensitive columns
    sensitive_columns = row['sensitive_attributes']  # Example sensitive column(s)
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

    return X_test, model, sensitive_columns, non_sensitive_columns

n = 2

for index, row in datasets.iterrows():
    X_test, model, sensitive_columns, non_sensitive_columns = load_and_preprocess_data(row)

    csv_file = os.path.join('result_csv', f"{row['name']}_results.csv")

    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=['baseline', 'tool'])
        df.to_csv(csv_file, index=False)
    
    for i in range(n):
        idi_baseline = calculate_idi_ratio_baseline(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000)
        idi_tool, discrimination = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples = 1000, num_seed = 100)
        print(idi_baseline)
        print(idi_tool)
        df_csv = pd.DataFrame({ 'baseline' : [idi_baseline], 'tool' : [idi_tool]})
        df_csv.to_csv(csv_file, mode='a', header=False, index=False)

'''
idi_ratio_baseline = calculate_idi_ratio_baseline(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000)
print(f"the result for baseline : {idi_ratio_baseline}")
idi_ratio_tool, discrimination_samples = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples = 1000, num_seed = 100)
print(f"the result for tool : {idi_ratio_tool}")
'''