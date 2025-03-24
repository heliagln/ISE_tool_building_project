import sys
import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split


# Import baseline and tool
chemin_methodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(chemin_methodes)
# from lab4_solution import load_and_preprocess_data  # Remplace par le nom réel de la méthode
from decision_tree import calculate_idi_ratio_tool  # Remplace par le nom réel de la méthode

def extract_data_model(dataset_name, info_dataset):
  file_path = f'../../dataset/processed_{dataset_name}.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
  model_path = f'../../DNN/model_processed_{dataset_name}.h5'  # Model path
  df = pd.read_csv(file_path)
  dataset = info_dataset.loc[info_dataset['name'] == dataset_name]
  # Splitting the dataset into features and target
  target_column = (dataset['target_label']).iloc[0] 
  X = df.drop(columns=[target_column])  # Features (drop the target column)
  y = df[target_column]  # Target variable
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  X_test = X_test.astype('float64')
  model = keras.models.load_model(model_path)
  # Define sensitive and non-sensitive columns
  sensitive_columns = np.array((dataset['sensitive_attributes']).iloc[0])  # Example sensitive column(s)
  non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]
  return model, X_test, sensitive_columns, non_sensitive_columns


def main():
  # Load information on dataset
  info_dataset = pd.read_csv('../../info_datasets.csv')
  info_dataset['sensitive_attributes'] = (info_dataset['sensitive_attributes']).apply(ast.literal_eval)
  
  # go through all datasets
  for index, row in info_dataset.iterrows():
    # load model and data
    model, X_test, sensitive_columns, non_sensitive_columns = extract_data_model(row['name'], info_dataset)

    # See for different size of seeds
    n = 20
    num_training_1 = 6000
    num_training_2 = 8000
    num_training_3 = 10000

    df = pd.DataFrame()

    def make_runs(df, num_runs, num_train):
      IDI_list = []
      for i in range(num_runs):
        idi_ratio = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_training=num_train)
        IDI_list.append(idi_ratio)
      print(f"mean for {num_train} : {sum(IDI_list)/len(IDI_list)} ")
      df[f'{num_train} training data'] = IDI_list
      return df

    df = make_runs(df, n, num_training_1)
    df = make_runs(df, n, num_training_2)
    df = make_runs(df, n, num_training_3)

    dataset_name = row['name']
    csv_file = os.path.join(f'result_training_{dataset_name}.csv')
    df.to_csv(csv_file, index=False)

    df_melted = df.melt(var_name="number of training data", value_name="IDI ratio")

    # plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="number of training data", y="IDI ratio", data=df_melted, width=0.4, palette="Set2")
    sns.swarmplot(x="number of training data", y="IDI ratio", data=df_melted, color="black", size=6, alpha=0.7)

    plt.title(f"Comparison of the IDI ratio as a function of the initial number of training data for the dataset {row['name']}")
    plt.savefig(f'figure_training_{dataset_name}.png')
    plt.close()

if __name__ == "__main__":
  main()