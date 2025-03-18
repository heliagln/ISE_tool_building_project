import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

# Import baseline and tool
chemin_methodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(chemin_methodes)
from lab4_solution import load_and_preprocess_data
from lab4_solution import calculate_idi_ratio_baseline
from decision_tree import calculate_idi_ratio_tool

# Load model and dataset
file_path = '../dataset/processed_adult.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
model_path = '../DNN/model_processed_adult.h5'  # Model path
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
X_test = X_test.astype('float64')
model = keras.models.load_model(model_path)

# Define sensitive and non-sensitive columns
sensitive_columns = ['age']  # Example sensitive column(s)
non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

# Compare performance
n = 10

IDI_ratio_tool = []

for i in range(n):
    idi_ratio_baseline = calculate_idi_ratio_baseline(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000)
    IDI_ratio_baseline.append(idi_ratio_baseline)
    idi_ratio_tool, discrimination_samples = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples = 1000)
    IDI_ratio_tool.append(idi_ratio_tool)

print(f"mean for baseline : {sum(IDI_ratio_baseline)/len(IDI_ratio_baseline)} ")
print(f"mean for tool : {sum(IDI_ratio_tool)/len(IDI_ratio_tool)} ")

graph = {
    'baseline': IDI_ratio_baseline,
    'tool': IDI_ratio_tool,
}

df = pd.DataFrame(graph)
df_melted = df.melt(var_name="method used", value_name="IDI ratio")

# plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="method used", y="IDI ratio", data=df_melted, width=0.4, palette="Set2")
sns.swarmplot(x="method used", y="IDI ratio", data=df_melted, color="black", size=6, alpha=0.7)

plt.title("Comparison of the IDI ratio between the baseline and the tool")
plt.show()