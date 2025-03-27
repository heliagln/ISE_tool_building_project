import numpy as np
import random
import time
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier


# Draw some random samples and predict them
def draw_samples(model, X_test, sensitive_columns, non_sensitive_columns, num_samples, num_training):
  # Take num_training samples from X_text
  df_random = X_test.sample(n=int(num_training), replace=True)
  df_random = df_random.reset_index(drop=True)

  # Predict them
  prediction = pd.DataFrame(model.predict(np.array(df_random)), index=df_random.index, columns=['prediction'])
  return df_random, prediction

# Preselect the inputs that go through sensitive nodes
def preselection(tree, X_test, sensitive_columns):
    # Take decision path for each of the inputs
    decision_path = tree.decision_path(X_test)

    # Extract which node is sensitive
    sensitive_indices = [X_test.columns.get_loc(col) for col in sensitive_columns]
    sensitive_nodes = np.where(np.isin(tree.tree_.feature, sensitive_indices))[0]

    # Find samples passing through these nodes
    selected_samples = pd.DataFrame(columns=X_test.columns)
    for i in range(len(X_test)): 
        path_nodes = decision_path.indices[decision_path.indptr[i]:decision_path.indptr[i+1]]
        sensible = [node for node in path_nodes if node in sensitive_nodes]
        if sensible != []:
            selected_samples.loc[len(selected_samples)] = X_test.iloc[i]

    return selected_samples

# Modify slightly the samples
def modify_samples(row, X_test, non_sensitive_columns):
    for col in non_sensitive_columns:
        if col in X_test.columns:
            min = X_test[col].min()
            max = X_test[col].max()
            modification = np.random.uniform(-0.1 * (max - min), 0.1 * (max - min))
            row[col] = np.clip(row[col] + modification, min, max)
    return row

# Search for discriminatory pairs in the decision tree
def find_random_discrimination(tree, X_test, sensitive_columns, non_sensitive_columns, num_samples):
    # Sample some input
    sample = X_test.sample(n=num_samples, replace=True)

    # Slightly modify them
    sample = sample.apply(lambda row: modify_samples(row, X_test, non_sensitive_columns), axis=1)

    # Extract the decision path for each sample
    decision_path = tree.decision_path(sample)

    # Needed information
    columns = X_test.columns
    df_path = pd.DataFrame(columns=['sample_a', 'sample_b'])

    for i in range(len(sample)):
        # needed information
        sample_a = sample.iloc[i]
        borne = pd.DataFrame({
            'max' : [X_test[col].max() for col in sensitive_columns],
            'min' : [X_test[col].min() for col in sensitive_columns]
            }, index=sensitive_columns)
        node_indices = decision_path.indices[decision_path.indptr[i]:decision_path.indptr[i+1]]

        # Extract sensitive nodes
        sensible_nodes = [(columns[tree.tree_.feature[node]], tree.tree_.threshold[node]) for node in node_indices if columns[tree.tree_.feature[node]] in sensitive_columns]
        
        # Check if there is sensible nodes
        if sensible_nodes != []:
            for feature_name, threshold in sensible_nodes:
                if sample_a[feature_name] <= threshold:
                    # Pick a new value
                    new_value = random.uniform(threshold, borne.loc[feature_name, 'max'])

                    # Change the value for the modified sample
                    sample_b = sample_a.copy()
                    sample_b[feature_name] = new_value

                    # Predict the class of each sample thanks to the decision tree
                    dt = pd.concat([sample_a, sample_b], axis=1).T
                    pred = tree.predict(dt)

                    # Register them if the classes are not the same
                    if pred[0] != pred[1]:
                        series = pd.Series({'sample_a':sample_a, 'sample_b':sample_b})
                        df_path = pd.concat([df_path, series.to_frame().T], ignore_index= True)
                    borne.loc[feature_name, 'max'] = threshold
                else:
                    # Pick a new value
                    new_value = random.uniform(threshold, borne.loc[feature_name, 'min'])

                    # Change the value for the modified sample
                    sample_b = sample_a.copy()
                    sample_b[feature_name] = new_value

                    # Predict the class of each sample thanks to the decision tree
                    dt = pd.concat([sample_a, sample_b], axis=1).T
                    pred = tree.predict(dt)

                    # Register them if the classes are not the same
                    if pred[0] != pred[1]:
                        series = pd.Series({'sample_a':sample_a, 'sample_b':sample_b})
                        df_path = pd.concat([df_path, series.to_frame().T], ignore_index= True)
                    borne.loc[feature_name, 'min'] = threshold
    return df_path


# Calculate the IDI ratio
def calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, threshold=0.05, num_training=8000):
    # Draw sample
    df_random, prediction = draw_samples(model, X_test, sensitive_columns, non_sensitive_columns, num_samples, num_training)
    # Predict the class of each sample
    prediction_class = np.where((np.array(prediction))>=0.5, 1, 0)

    # Show the percentage of predictions between 0.45 and 0.55
    pred = np.array(prediction['prediction'])
    mask = (pred > 0.45) & (pred < 0.55)
    print(f"The percentage of predictions between 0.45 and 0.55 is {sum(mask)/len(mask)}")

    # Decision tree training
    tree = DecisionTreeClassifier()
    tree.fit(df_random, prediction_class)

    # Is it dependant of sensible parameters
    features_used = tree.tree_.feature
    features_used = features_used[features_used != -2]
    features_used = set(features_used)

    columns = X_test.columns

    sensible_nodes = [(columns[feature]) for feature in features_used if columns[feature] in sensitive_columns]
    print("The sensitive columns identified by the decision tree classifier : ")
    print(sensible_nodes)

    if sensible_nodes == []: # if there is no disciminating nodes
        return 0
    else:
        # Extract sample
        df_path = pd.DataFrame(columns=['sample_a', 'sample_b'])

        preselected_samples = preselection(tree, X_test, sensitive_columns)
        
        while len(df_path) < num_samples:
            df_result = find_random_discrimination(tree, preselected_samples, sensitive_columns, non_sensitive_columns, num_samples)
            df_path = pd.concat([df_path, df_result], ignore_index=True)
            # know advancement
            print(f"\rProgression : {(len(df_path)/num_samples)*100}%", end='', flush=True)
        
        df_path = df_path.head(num_samples)

        df_samples = df_path

        # Prediction for each sample of a pair
        array_for_prediction_a = np.stack(df_samples['sample_a'].values)
        prediction_a = model.predict(array_for_prediction_a)

        array_for_prediction_b = np.stack(df_samples['sample_b'].values)
        prediction_b = model.predict(array_for_prediction_b)

        # check for difference in prediction
        pred_diff = abs(prediction_a - prediction_b)
        mask = (pred_diff > threshold)
        indices = np.where(mask)[0]

        # extract discriminatory information
        number_discrimination = len(indices)
        discrimination_pairs = [((df_samples['sample_a']).iloc[i], (df_samples['sample_b']).iloc[i]) for i in indices]
        try_number = len(df_samples)

        return number_discrimination/ try_number

# Main function
def main():
    # Load dataset and model
    dataset_name = 'adult'
    file_path = f'dataset/processed_{dataset_name}.csv' 
    model_path = f'DNN/model_processed_{dataset_name}.h5' 
    df = pd.read_csv(file_path)

    # Take information on dataset
    info_dataset = pd.read_csv('info_datasets.csv')
    info_dataset['sensitive_attributes'] = (info_dataset['sensitive_attributes']).apply(ast.literal_eval)
    dataset = info_dataset.loc[info_dataset['name'] == dataset_name]
    
    # Splitting the dataset into attributes and class
    target_column = (dataset['target_label']).iloc[0] 
    X = df.drop(columns=[target_column])
    y = df[target_column] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test = X_test.astype('float64') # change the type of the columns
    model = keras.models.load_model(model_path) # extract the model

    # Define sensitive and non-sensitive columns
    sensitive_columns = np.array((dataset['sensitive_attributes']).iloc[0])  # Example sensitive column(s)
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

    # Calculate and print the IDI Ratio and runtime
    start = time.perf_counter()
    IDI_ratio = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_training=8000)
    end = time.perf_counter()
    print(f"runtime : {end-start}")
    print(f"IDI Ratio: {IDI_ratio}")

if __name__ == "__main__":
    main()