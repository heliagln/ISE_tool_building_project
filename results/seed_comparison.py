import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras


# Import baseline and tool
chemin_methodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(chemin_methodes)
from lab4_solution import load_and_preprocess_data  # Remplace par le nom réel de la méthode
from tool import calculate_idi_ratio_tool  # Remplace par le nom réel de la méthode


# load model and dataset
file_path = '../dataset/processed_adult.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
model_path = '../DNN/model_processed_adult.h5'  # Model path
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
X_test = X_test.astype('float64')
model = keras.models.load_model(model_path)

# Define sensitive and non-sensitive columns
sensitive_columns = ['age']  # Example sensitive column(s)
non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

# See for different size of seeds
n = 20
discrimination_125 = []
discrimination_100 = []
discrimination_50 = []

for i in range(n):
  idi_ratio_125, discrimination_samples = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_seed = 125)
  discrimination_125.append(idi_ratio_125)
  idi_ratio_100, discrimination_samples = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_seed = 100)
  discrimination_100.append(idi_ratio_100)
  idi_ratio_50, discrimination_samples = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_seed = 50)
  discrimination_50.append(idi_ratio_50)

print(f"mean for 125 : {sum(discrimination_125)/len(discrimination_125)} ")
print(f"mean for 100 : {sum(discrimination_100)/len(discrimination_100)} ")
print(f"mean for 50 : {sum(discrimination_50)/len(discrimination_50)} ")
graph = {
    '50 seeds': discrimination_50,
    '100 seeds': discrimination_100,
    '125 seeds':discrimination_125,
}
df = pd.DataFrame(graph)
df_melted = df.melt(var_name="number of seeds", value_name="IDI ratio")

# plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="number of seeds", y="IDI ratio", data=df_melted, width=0.4, palette="Set2")
sns.swarmplot(x="number of seeds", y="IDI ratio", data=df_melted, color="black", size=6, alpha=0.7)

plt.title("Comparison of the IDI ratio as a function of the initial number of seeds")
plt.show()