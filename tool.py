import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from tensorflow import keras

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
  df_random = X_test.sample(n=int(num_samples*4), replace=True)
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

# Aplply the clustering to get the cluster's center
def apply_clustering(data):
  # Normalize
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)
  # fit and predict kmeans
  kmeans = KMeans(n_clusters=2, random_state=42)
  labels = kmeans.fit_predict(scaled_data)
  # take the centers of the clusters
  centers = kmeans.cluster_centers_
  return centers

# Select the boundary points, the ones that are at the boundary between the two clusters
def select_boundary_points(df_random, centers, num_samples):
  # distances entre les points et les centres
  dist_to_centers = pairwise_distances(df_random, centers)
  # Reconnaître les points à la frontière
  dist_to_first, dist_to_second = dist_to_centers[:, 0], dist_to_centers[:, 1]
  diff_centers = abs(dist_to_first - dist_to_second)
  boundary_index = np.argsort(diff_centers)[:num_samples]
  return boundary_index

# Randomly flip the sensitive parameters of the selected samples
def random_flip(boundary_samples, X_test, sensitive_columns):
  modified_samples = boundary_samples.copy()
  for ind in modified_samples.index:
    for col in sensitive_columns:
      if col in X_test.columns:  # Ensure the column exists
          unique_values = X_test[col].unique()
          modified_samples.loc[ind,col] = np.random.choice(unique_values)
  return modified_samples

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

def crossover(population):
  new_population = pd.DataFrame(columns=population.columns)
  nb_columns = len(population.columns)
  # for each
  for index, s in population.iterrows():
    # select another sample
    s1 = s.copy()
    s2 = population.sample(n=1).iloc[0]

    # select random features
    n = np.random.randint(0, nb_columns - 1)
    random_columns = np.random.choice(population.columns, n, replace=False)

    # create two new instances by exchanging the features of each sample
    s1[random_columns], s2[random_columns] = s2[random_columns].values, s1[random_columns].values

    # add them to the new population
    new_population.loc[len(new_population)] = s1
    new_population.loc[len(new_population)] = s2
  # return the new population
  return new_population

def selection(model, population, centers, sample_number):
  # prediction
  prediction = pd.DataFrame(model.predict(np.array(population)), index=population.index, columns=['prediction'])
  population['prediction'] = prediction['prediction']

  # take the "sample_number" closest point to each center
  selected_population_index = select_boundary_points(population, centers, sample_number)
  population = population.drop(columns=['prediction'])
  selected_population = (population.iloc[selected_population_index]).reset_index(drop=True)
  return selected_population


# Calculate the IDI ratio
def calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000, num_seed = 100):
  # Draw sample
  df_random, prediction = draw_samples(model, X_test, sensitive_columns, non_sensitive_columns, num_samples)
  df_random['prediction'] = prediction['prediction']

  # Clustering algorithm
  centers = apply_clustering(df_random)

  # Select boundary points
  boundary_index = select_boundary_points(df_random, centers, num_seed)

  # Extract needed information
  df_random = df_random.drop(columns=['prediction'])
  boundary_samples = (df_random.iloc[boundary_index]).reset_index(drop=True)

  number_of_samples = len(boundary_samples.index)
  discrimination_number = 0
  discrimination_samples = []
  while number_of_samples < num_samples:
    # Predict the samples
    boundary_pred = pd.DataFrame(model.predict(np.array(boundary_samples)), index=boundary_samples.index, columns=['prediction'])# (prediction.iloc[boundary_index]).reset_index(drop=True)

    # randomly flip sensible parameters of the samples
    modified_samples = random_flip(boundary_samples, X_test, sensitive_columns)

    # new prediction for the modified samples
    new_pred = pd.DataFrame(model.predict(np.array(modified_samples)), index=boundary_samples.index, columns=['prediction'])

    # check for discrimination
    pred_diff = np.array(abs(boundary_pred['prediction'] - new_pred['prediction']))
    disc, disc_samp = evaluate_discrimination(pred_diff, boundary_samples, modified_samples)
    discrimination_samples = discrimination_samples + disc_samp
    discrimination_number += disc

    # update parameters
    number_of_samples += num_seed

    if number_of_samples < num_samples:
      # crossover
      new_pop = crossover(boundary_samples)

      # selection
      boundary_samples = selection(model, new_pop, centers, num_seed)

  return discrimination_number / num_samples, discrimination_samples

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
    idi_ratio, discrimination_samples = calculate_idi_ratio_tool(model, X_test, sensitive_columns, non_sensitive_columns)
    print(idi_ratio)
    print(f"IDI Ratio: {idi_ratio}")

if __name__ == "__main__":
    main()