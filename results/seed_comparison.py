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
file_path = '../dataset/processed_adult.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
model_path = '../DNN/model_processed_adult.h5'  # Model path
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
X_test = X_test.astype('float64')
model = keras.models.load_model(model_path)

# Define sensitive and non-sensitive columns
sensitive_columns = ['age', 'gender', 'race']  # Example sensitive column(s)
non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

# See for different size of seeds
n = 10
discrimination_1500 = []
discrimination_1000 = []
discrimination_500 = []

for i in range(n):
  idi_ratio_1500 = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1500)
  discrimination_1500.append(idi_ratio_1500)
  idi_ratio_1000 = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000)
  discrimination_1000.append(idi_ratio_1000)
  idi_ratio_500 = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=500)
  discrimination_500.append(idi_ratio_500)

print(f"mean for 1500 : {sum(discrimination_1500)/len(discrimination_1500)} ")
print(f"mean for 1000 : {sum(discrimination_1000)/len(discrimination_1000)} ")
print(f"mean for 500 : {sum(discrimination_500)/len(discrimination_500)} ")
graph = {
    '500 samples': discrimination_500,
    '1000 samples': discrimination_1000,
    '1500 samples':discrimination_1500,
}
df = pd.DataFrame(graph)
df_melted = df.melt(var_name="number of samples", value_name="IDI ratio")

# plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="number of samples", y="IDI ratio", data=df_melted, width=0.4, palette="Set2")
sns.swarmplot(x="number of samples", y="IDI ratio", data=df_melted, color="black", size=6, alpha=0.7)

plt.title("Comparison of the IDI ratio as a function of the initial number of samples")
plt.show()