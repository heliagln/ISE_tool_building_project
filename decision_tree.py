import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier

# 1. Data loading and preprocessing
def load_and_preprocess_data(file_path):
  df = pd.read_csv(file_path)
  # Splitting the dataset into features and target
  target_column = 'Class-label'# 'income'  # Modify the target column name if necessary
  X = df.drop(columns=[target_column])  # Features (drop the target column)
  y = df[target_column]  # Target variable
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  return X_train, X_test, y_train, y_test

# Draw some random samples (must be higher than num_samples) and slightly modify them
def draw_samples(model, X_test, sensitive_columns, non_sensitive_columns, num_samples):
  # Prendre à peu près 1 quart du set
  df_random = X_test.sample(n=int(num_samples*6), replace=True)
  df_random = df_random.reset_index(drop=True)
  # Apply perturbation on non sensitive columns
  for index, sample in df_random.iterrows():
    for col in non_sensitive_columns:
        if col in X_test.columns:  # Ensure the column exists
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))  # Small perturbation
            sample[col] = np.clip(sample[col] + perturbation, min_val, max_val)
  # Prédit tous les tests
  prediction = pd.DataFrame(model.predict(np.array(df_random)), index=df_random.index, columns=['prediction'])
  return df_random, prediction

# random path
def random_path(model, tree, X_test, sensitive_columns):
    sample = X_test.sample(n=1) # X_test.iloc[np.random.choice(len(X_test))]

    predict_sample = (model.predict((np.array(sample)).reshape(1, -1)))[0][0]

    decision_path = tree.decision_path(sample)

    node_indices = decision_path.indices[decision_path.indptr[0]:decision_path.indptr[1]]

    borne = pd.DataFrame({
        'max' : [X_test[col].max() for col in sensitive_columns],
        'min' : [X_test[col].min() for col in sensitive_columns]
        }, index=sensitive_columns)

    sample_a = sample.iloc[0]

    tryout = 0
    disc_number = 0
    disc_sample = []

    for node in node_indices:
        columns = X_test.columns
        feature_name = columns[tree.tree_.feature[node]]
        threshold = tree.tree_.threshold[node]
        if feature_name in sensitive_columns:
            if sample_a[feature_name] <= threshold:
                sample_b = sample_a.copy()
                sample_b[feature_name] = random.uniform(threshold, borne.loc[feature_name, 'max'])
                predict_sample_b = (model.predict((np.array(sample_b)).reshape(1, -1)))[0][0]
                if abs(predict_sample - predict_sample_b) >= 0.05:
                    disc_number += 1
                    disc_sample.append((sample_a, sample_b))
                    print('yes')
                borne.loc[feature_name, 'max'] = threshold
            else:
                sample_b = sample_a.copy()
                sample_b[feature_name] = random.uniform(borne.loc[feature_name, 'min'], threshold)
                predict_sample_b = (model.predict((np.array(sample_b)).reshape(1, -1)))[0][0]
                if abs(predict_sample - predict_sample_b) >= 0.05:
                    disc_number += 1
                    disc_sample.append((sample_a, sample_b))
                    print('yes')
                borne.loc[feature_name, 'min'] = threshold
            tryout += 1
    return tryout, disc_number, disc_sample


# evaluate discrimination knowing the difference in predictions
def evaluate_discrimination(pred_diff, boundary_samples, modified_samples, threshold = 0.05):
  # check for discrimination
  mask = (pred_diff > threshold)
  first_half = mask[:(int(len(mask)/2))]
  # print(sum(first_half))
  second_half = mask[(int(len(mask)/2)):]
  # print(sum(second_half))
  indices = np.where(mask)[0]
  number_discrimination = len(indices)
  discrimination_samples = [(boundary_samples.loc[i], modified_samples.loc[i]) for i in indices]
  return number_discrimination, discrimination_samples


# Calculate the IDI ratio
def calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_seed = 100):
    # Draw sample
    df_random, prediction = draw_samples(model, X_test, sensitive_columns, non_sensitive_columns, num_samples)
    prediction_class = np.where((np.array(prediction))>=0.5, 1, 0)

    # decision tree training
    tree = DecisionTreeClassifier()
    tree.fit(df_random, prediction_class)

    # needed information
    try_number = 0
    disc = 0
    discrimination_pairs = []

    for i in range(num_samples):
        tryout, disc_number, disc_sample = random_path(model, tree, X_test, sensitive_columns)
        try_number += tryout
        disc += disc_number
        discrimination_pairs = discrimination_pairs + disc_sample
    
    return disc, try_number # discrimination_number / num_samples, discrimination_samples

# 6. Main function
def main():
    # 1. Load dataset and model
    file_path = 'dataset/processed_adult.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
    model_path = 'DNN/model_processed_adult.h5'  # Model path
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    X_test = X_test.astype('float64')
    model = keras.models.load_model(model_path)

    # 2. Define sensitive and non-sensitive columns
    sensitive_columns = ['age', 'gender', 'race']  # Example sensitive column(s)
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

    # 3. Calculate and print the Individual Discrimination Instance Ratio
    disc, tryout = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=500, num_seed=50)
    print(f"number of tryout : {tryout} and number for discrimination : {disc}")
    print(f"IDI Ratio: {disc/tryout}")

if __name__ == "__main__":
    main()