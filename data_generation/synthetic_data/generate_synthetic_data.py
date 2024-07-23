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
