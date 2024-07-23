import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    """
    Load and preprocess data from the specified directory.

    Args:
        data_dir (str): Path to the directory containing the data.

    Returns:
        dict: Dictionary containing training and validation data.
    """
    x_data = []
    y_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            x_data.append(df.values)
            y_data.append(int(filename.split('_')[1]))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val
    }

    return data

def normalize_data(data):
    """
    Normalize data to the range [0, 1].

    Args:
        data (np.ndarray): Data to normalize.

    Returns:
        np.ndarray: Normalized data.
    """
    return data / np.max(data)

if __name__ == "__main__":
    data_dir = './data_generation/synthetic_data/data'
    data = load_data(data_dir)

    print("Data loaded and split into training and validation sets.")
    print(f"Training data shape: {data['x_train'].shape}")
    print(f"Validation data shape: {data['x_val'].shape}")
