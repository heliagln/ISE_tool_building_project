import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
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

# Modify samples
def modify_training_samples(row, X_test, non_sensitive_columns):
    for col in non_sensitive_columns:
        if col in X_test.columns:  # Ensure the column exists
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))  # Small perturbation
            row[col] = np.clip(row[col] + perturbation, min_val, max_val)
    return row

# Draw some random samples (must be higher than num_samples) and slightly modify them
def draw_samples(model, X_test, sensitive_columns, non_sensitive_columns, num_samples, num_training):
  # Prendre à peu près 1 quart du set
  df_random = X_test.sample(n=int(num_training), replace=True) # num_samples*6
  df_random = df_random.reset_index(drop=True)
  # Apply perturbation on non sensitive columns
  df_random = df_random.apply(lambda row: modify_training_samples(row, X_test, non_sensitive_columns), axis=1)
  # Prédit tous les tests
  prediction = pd.DataFrame(model.predict(np.array(df_random)), index=df_random.index, columns=['prediction'])
  return df_random, prediction

def analyze_threshold(sample_a, borne, feature_name, minmax, threshold):
    series = pd.Series({
        'sample' : sample_a.copy(),
        'feature_name' : feature_name,
        'first_threshold' : threshold,
        'second_threshold' : borne.loc[feature_name, minmax],
    })
    borne.loc[feature_name, minmax] = threshold
    return series


# random path
def random_path(model, tree, X_test, sensitive_columns, non_sensitive_columns):
    sample = X_test.sample(n=1) # X_test.iloc[np.random.choice(len(X_test))]

    decision_path = tree.decision_path(sample)

    node_indices = decision_path.indices[decision_path.indptr[0]:decision_path.indptr[1]]

    borne = pd.DataFrame({
        'max' : [X_test[col].max() for col in sensitive_columns],
        'min' : [X_test[col].min() for col in sensitive_columns]
        }, index=sensitive_columns)
    
    sample_a = sample.iloc[0]
    columns = X_test.columns
    for col in non_sensitive_columns:
        if col in columns:
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
            sample_a[col] = np.clip(sample_a[col] + perturbation, min_val, max_val)
    
    sensible_nodes = [(columns[tree.tree_.feature[node]], tree.tree_.threshold[node]) for node in node_indices if columns[tree.tree_.feature[node]] in sensitive_columns]
    df_path = pd.DataFrame(columns=['sample', 'feature_name', 'first_threshold', 'second_threshold'])

    if sensible_nodes != []:
        for feature_name, threshold in sensible_nodes:
            if sample_a[feature_name] <= threshold:
                series = analyze_threshold(sample_a, borne, feature_name, 'max', threshold)
                df_path = pd.concat([df_path, series.to_frame().T], ignore_index= True)
            else:
                series = analyze_threshold(sample_a, borne, feature_name, 'min', threshold)
                df_path = pd.concat([df_path, series.to_frame().T], ignore_index= True)
    return df_path
    
def collect_sample(row):
    return row['sample'].copy()

def modify_sample(row):
    sample = row['sample'].copy()
    col = row['feature_name']
    first_threshold, second_threshold = row['first_threshold'], row['second_threshold']
    new_value = random.uniform(first_threshold, second_threshold)
    sample[col] = new_value
    return sample


# Calculate the IDI ratio
def calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_seed = 100, threshold=0.05, num_training=6000):
    # Draw sample
    df_random, prediction = draw_samples(model, X_test, sensitive_columns, non_sensitive_columns, num_samples, num_training)
    prediction_class = np.where((np.array(prediction))>=0.5, 1, 0)

    # decision tree training
    tree = DecisionTreeClassifier()
    tree.fit(df_random, prediction_class)

    # is it dependant of sensible parameters
    # Récupérer les indices des colonnes utilisées dans les nœuds
    features_used = tree.tree_.feature

    # Supprimer les valeurs -2 (indiquant les feuilles de l'arbre)
    features_used = features_used[features_used != -2]

    # Obtenir les indices uniques des colonnes utilisées
    features_used = set(features_used)

    columns = X_test.columns

    sensible_nodes = [(columns[feature]) for feature in features_used if columns[feature] in sensitive_columns]
    print(sensible_nodes)

    if sensible_nodes == []: # if there is no disciminating space
        return 0
    else:
        # extract paths
        df_path = pd.DataFrame(columns=['sample', 'feature_name', 'first_threshold', 'second_threshold', 'prediction'])

        while len(df_path) < num_seed:
            df_result = random_path(model, tree, X_test, sensitive_columns, non_sensitive_columns)
            df_path = pd.concat([df_path, df_result], ignore_index=True)
            # know advancement
            # print(len(df_path))
        
        df_path = df_path.head(num_seed)

        df_samples = df_path.sample(n=num_samples, replace=True)

        # prediction
        array_for_prediction = np.stack(df_samples['sample'].values)
        prediction = model.predict(array_for_prediction)

        # needed information
        samples = df_samples.apply(collect_sample, axis=1)

        # modify the sample thanks to the 
        df_modifed_samples = df_samples.apply(modify_sample, axis=1)

        # Predict the new samples
        modified_prediction = model.predict(np.array(df_modifed_samples))

        # check for difference in prediction
        pred_diff = abs(prediction - modified_prediction)
        mask = (pred_diff > threshold)
        indices = np.where(mask)[0]

        # extract discriminatory information
        number_discrimination = len(indices)
        discrimination_pairs = [(samples.iloc[i], df_modifed_samples.iloc[i]) for i in indices]
        try_number = len(df_modifed_samples)

        return number_discrimination/ try_number

# 6. Main function
def main():
    # 1. Load dataset and model
    file_path = 'dataset/processed_adult.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
    model_path = 'DNN/model_processed_adult.h5'  # Model path
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    X_test = X_test.astype('float64')
    model = keras.models.load_model(model_path)

    # 2. Define sensitive and non-sensitive columns
    sensitive_columns = ['race', 'gender', 'age']  # Example sensitive column(s)
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

    # 3. Calculate and print the Individual Discrimination Instance Ratio
    IDI_ratio = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_seed=200, num_training=6000)
    # print(f"number of tryout : {tryout} and number for discrimination : {disc}")
    print(f"IDI Ratio: {IDI_ratio}")

if __name__ == "__main__":
    main()