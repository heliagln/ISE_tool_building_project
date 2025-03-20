import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 1. Data loading and preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Splitting the dataset into features and target
    target_column = 'class'# 'income'  # Modify the target column name if necessary
    X = df.drop(columns=[target_column])  # Features (drop the target column)
    y = df[target_column]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


# 2. Load pre-trained DNN model
def load_model(model_path):
    return load_model(model_path)


# 3. Generate test sample pairs (Random Search)
def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    sample_a = X_test.iloc[np.random.choice(len(X_test))]
    sample_b = sample_a.copy()

    # Apply perturbation on sensitive features (random flipping)
    for col in sensitive_columns:
        if col in X_test.columns:  # Ensure the column exists
            unique_values = X_test[col].unique()
            sample_b[col] = np.random.choice(unique_values)  # Randomly select a new value

    # Apply perturbation on non-sensitive features
    for col in non_sensitive_columns:
        if col in X_test.columns:  # Ensure the column exists
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))  # Small perturbation
            sample_a[col] = np.clip(sample_a[col] + perturbation, min_val, max_val)
            sample_b[col] = np.clip(sample_b[col] + perturbation, min_val, max_val)

    return sample_a, sample_b


# 4. Model prediction and individual discrimination evaluation
def evaluate_discrimination(model, sample_a, sample_b, threshold=0.05, discrimination_pairs=None):
    if discrimination_pairs is None:
        discrimination_pairs = []  # Initialize list to store discriminatory sample pairs

    # Convert sample_a and sample_b to numpy arrays and reshape
    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)

    # Model predictions
    prediction_a = model.predict(sample_a.reshape(1, -1))  # Reshape to fit model input format
    prediction_b = model.predict(sample_b.reshape(1, -1))

    # Get prediction results (usually probability values)
    pred_a = prediction_a[0][0]  # Get the value from the output (shape: (1, 1))
    pred_b = prediction_b[0][0]

    # Check if the difference in predictions is greater than the threshold
    if abs(pred_a - pred_b) > threshold:
        discrimination_pairs.append((sample_a, sample_b))  # Store the discriminatory sample pair
        return 1  # Individual discriminatory instance
    else:
        return 0  # Not a discriminatory instance


# 5. Calculate Individual Discrimination Instance Ratio (IDI ratio)
def calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000):
    discrimination_count = 0

    for _ in range(num_samples):
        sample_a, sample_b = generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)
        discrimination_count += evaluate_discrimination(model, sample_a, sample_b)

    # Total number of generated samples
    total_generated = num_samples

    # Calculate Individual Discrimination Instance Ratio
    IDI_ratio = discrimination_count / total_generated
    return IDI_ratio


# 6. Main function
def main():
    file_path = 'dataset/processed_credit.csv' # 'model/processed_kdd_cleaned.csv'  # Dataset path
    model_path = 'DNN/model_processed_credit.h5'  # Model path
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    X_test = X_test.astype('float64')
    model = keras.models.load_model(model_path)

    # 2. Define sensitive and non-sensitive columns
    sensitive_columns = ['SEX', 'EDUCATION', 'MARRIAGE']  # Example sensitive column(s)
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

    # 3. Calculate and print the Individual Discrimination Instance Ratio
    idi_ratio = calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns)
    print(f"IDI Ratio: {idi_ratio}")


if __name__ == "__main__":
    main()