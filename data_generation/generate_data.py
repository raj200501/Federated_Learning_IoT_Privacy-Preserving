import os
from synthetic_data.generate_synthetic_data import generate_iot_data

def main():
    data_dir = './data_generation/synthetic_data/data'
    os.makedirs(data_dir, exist_ok=True)
    generate_iot_data(data_dir)

if __name__ == "__main__":
    main()
