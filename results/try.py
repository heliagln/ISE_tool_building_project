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

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Splitting the dataset into features and target
    target_column = 'occupation'# 'income'  # Modify the target column name if necessary
    X = df.drop(columns=[target_column])  # Features (drop the target column)
    y = df[target_column]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# Load model and dataset
file_path = '../dataset/processed_dutch.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
model_path = '../DNN/model_processed_dutch.h5'  # Model path
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
X_test = X_test.astype('float64')
model = keras.models.load_model(model_path)

# Define sensitive and non-sensitive columns
sensitive_columns = ['sex', 'age']  # Example sensitive column(s)
non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

idi_ratio_baseline = calculate_idi_ratio_baseline(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000)
print(f"the result for baseline : {idi_ratio_baseline}")
idi_ratio_tool, discrimination_samples = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples = 1000, num_seed = 100)
print(f"the result for tool : {idi_ratio_tool}")
