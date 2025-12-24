import numpy as np
import pandas as pd

def generate_iot_data(data_dir):
    num_devices = 100
    num_samples = 1000

    for device_id in range(num_devices):
        data = np.random.rand(num_samples, 10)
        df = pd.DataFrame(data, columns=[f'sensor_{i}' for i in range(10)])
        df.to_csv(f"{data_dir}/device_{device_id}.csv", index=False)

    print(f"Generated data for {num_devices} devices with {num_samples} samples each.")


def generate_iot_data_configurable(data_dir, num_devices=100, num_samples=1000, num_sensors=10, seed=42):
    """
    Generate synthetic IoT data with configurable dimensions.

    Args:
        data_dir (str): Output directory for CSV files.
        num_devices (int): Number of devices (files) to generate.
        num_samples (int): Number of samples per device.
        num_sensors (int): Number of sensor features per sample.
        seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    for device_id in range(num_devices):
        data = rng.random((num_samples, num_sensors))
        df = pd.DataFrame(data, columns=[f'sensor_{i}' for i in range(num_sensors)])
        df.to_csv(f"{data_dir}/device_{device_id}.csv", index=False)

    print(
        f"Generated data for {num_devices} devices with {num_samples} samples each "
        f"and {num_sensors} sensors."
    )
