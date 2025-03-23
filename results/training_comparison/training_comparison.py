import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split


# Import baseline and tool
chemin_methodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(chemin_methodes)
# from lab4_solution import load_and_preprocess_data  # Remplace par le nom réel de la méthode
from decision_tree import calculate_idi_ratio_tool  # Remplace par le nom réel de la méthode

# 1. Data loading and preprocessing
def load_and_preprocess_data(file_path):
  df = pd.read_csv(file_path)
  # Splitting the dataset into features and target
  target_column = 'Class-label'# 'income'  # Modify the target column name if necessary
  X = df.drop(columns=[target_column])  # Features (drop the target column)
  y = df[target_column]  # Target variable
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  return X_train, X_test, y_train, y_test

# load model and dataset
file_path = '../../dataset/processed_adult.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
model_path = '../../DNN/model_processed_adult.h5'  # Model path
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
X_test = X_test.astype('float64')
model = keras.models.load_model(model_path)

# Define sensitive and non-sensitive columns
sensitive_columns = ['age', 'gender', 'race']  # Example sensitive column(s)
non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

# See for different size of seeds
n = 100
num_training_0 = 4000
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

df = make_runs(df, n, num_training_0)
df = make_runs(df, n, num_training_1)
df = make_runs(df, n, num_training_2)
df = make_runs(df, n, num_training_3)

csv_file = os.path.join('result_training.csv')
df.to_csv(csv_file, index=False)

df_melted = df.melt(var_name="number of training data", value_name="IDI ratio")

# plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="number of training data", y="IDI ratio", data=df_melted, width=0.4, palette="Set2")
sns.swarmplot(x="number of training data", y="IDI ratio", data=df_melted, color="black", size=6, alpha=0.7)

plt.title("Comparison of the IDI ratio as a function of the initial number of training data")
plt.show()