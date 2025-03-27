import sys
import os
import pandas as pd
import ast
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Import baseline and tool
chemin_methodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(chemin_methodes)
from lab4_solution import calculate_idi_ratio_baseline
from decision_tree import calculate_idi_ratio_tool


def load_and_preprocess_data(row):
    file_path = '../dataset/processed_' + row['name'] + '.csv' 
    model_path = '../DNN/model_processed_'+ row['name'] + '.h5' 
    df = pd.read_csv(file_path)

    # Splitting the dataset into attributes and class
    target_column = row['target_label']
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_test = X_test.astype('float64')
    model = keras.models.load_model(model_path)

    # Define sensitive and non-sensitive columns
    sensitive_columns = row['sensitive_attributes']  # Example sensitive column(s)
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

    return X_test, model, sensitive_columns, non_sensitive_columns

def main():
    # Load information on dataset
    info_dataset = pd.read_csv('../info_datasets.csv')
    info_dataset['sensitive_attributes'] = (info_dataset['sensitive_attributes']).apply(ast.literal_eval)

    # Number of runs
    n = 10

    # Foreach datasets
    for index, row in info_dataset.iterrows():
        X_test, model, sensitive_columns, non_sensitive_columns = load_and_preprocess_data(row)

        csv_file = os.path.join(f"performance_comparison/{row['name']}_results.csv")

        if not os.path.exists(csv_file):
            df = pd.DataFrame(columns=['IDI_baseline','time_baseline', 'IDI_tool', 'time_tool'])
            df.to_csv(csv_file, index=False)
        
        for i in range(n):
            start_baseline = time.perf_counter()
            idi_baseline = calculate_idi_ratio_baseline(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000)
            end_baseline = time.perf_counter()
            time_baseline = end_baseline - start_baseline
            start_tool = time.perf_counter()
            idi_tool = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples = 1000, num_training=8000)
            end_tool = time.perf_counter()
            time_tool = end_tool - start_tool
            print(f"IDI ratio for baseline : {idi_baseline}")
            print(f"runtime for the baseline : {time_baseline}")
            print(f"IDI ratio for tool : {idi_tool}")
            print(f"runtime for tool : {time_tool}")
            df_csv = pd.DataFrame({ 'IDI_baseline' : [idi_baseline], 'time_baseline' : [time_baseline], 'IDI_tool' : [idi_tool], 'time_tool' : [time_tool]})
            df_csv.to_csv(csv_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    main()
