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


def load_flattened_samples(data_dir):
    """
    Load CSVs and return flattened samples with per-row labels.

    Each CSV is treated as one device, and each row becomes a sample.
    The label is the device id parsed from the filename.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_samples, y_labels)
    """
    x_samples = []
    y_labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            device_id = int(filename.split('_')[1].split('.')[0])
            x_samples.append(df.values)
            y_labels.append(np.full(df.shape[0], device_id, dtype=int))

    if not x_samples:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    x_samples = np.concatenate(x_samples, axis=0)
    y_labels = np.concatenate(y_labels, axis=0)
    return x_samples, y_labels


def create_federated_splits(
    data_dir,
    num_clients=5,
    val_size=0.2,
    random_state=42,
):
    """
    Create a federated dataset dictionary with per-client splits.

    Returns:
        dict: Data dictionary with keys:
            x_train, y_train, x_val, y_val, and per-client keys.
    """
    x_samples, y_labels = load_flattened_samples(data_dir)

    x_train, x_val, y_train, y_val = train_test_split(
        x_samples, y_labels, test_size=val_size, random_state=random_state, stratify=y_labels
    )

    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
    }

    client_splits = np.array_split(np.arange(x_train.shape[0]), num_clients)
    for client_id, indices in enumerate(client_splits):
        data[f"x_train_client_{client_id}"] = x_train[indices]
        data[f"y_train_client_{client_id}"] = y_train[indices]
        data[f"x_val_client_{client_id}"] = x_val
        data[f"y_val_client_{client_id}"] = y_val

    return data

if __name__ == "__main__":
    data_dir = './data_generation/synthetic_data/data'
    data = load_data(data_dir)

    print("Data loaded and split into training and validation sets.")
    print(f"Training data shape: {data['x_train'].shape}")
    print(f"Validation data shape: {data['x_val'].shape}")
