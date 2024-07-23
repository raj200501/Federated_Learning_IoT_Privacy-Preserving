import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_sample_data(x_data, y_data, sample_indices):
    """
    Plot sample data points.

    Args:
        x_data (np.ndarray): Feature data.
        y_data (np.ndarray): Labels.
        sample_indices (list): List of indices for the samples to plot.
    """
    num_samples = len(sample_indices)
    plt.figure(figsize=(10, num_samples * 2))

    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(x_data[idx])
        plt.title(f"Sample {idx} - Class {y_data[idx]}")
    
    plt.tight_layout()
    plt.show()

def plot_data_distribution(y_data):
    """
    Plot the distribution of the data classes.

    Args:
        y_data (np.ndarray): Labels.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_data)
    plt.title("Data Distribution")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    # Example usage
    # Sample data (replace with actual data)
    x_data = np.random.rand(100, 10)
    y_data = np.random.randint(0, 5, size=100)

    plot_sample_data(x_data, y_data, sample_indices=[0, 1, 2, 3, 4])
    plot_data_distribution(y_data)
