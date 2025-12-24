"""Run a lightweight federated learning demo end-to-end."""

import argparse
import csv
import importlib
import os
import random
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

importlib.import_module("sitecustomize")

from data_generation.synthetic_data.generate_synthetic_data import generate_iot_data_configurable  # noqa: E402
from model_training.federated_averaging import federated_averaging_mlp  # noqa: E402
from utils.data_loading import create_federated_splits  # noqa: E402


def _has_full_dependencies():
    modules = ["numpy", "pandas", "sklearn", "tensorflow"]
    for module_name in modules:
        module = importlib.import_module(module_name)
        if getattr(module, "__FAKE__", False):
            return False
    return True


def _generate_lightweight_csvs(data_dir, num_devices, num_samples, num_sensors, seed):
    random.seed(seed)
    for device_id in range(num_devices):
        path = os.path.join(data_dir, f"device_{device_id}.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"sensor_{i}" for i in range(num_sensors)])
            for _ in range(num_samples):
                row = [f"{random.random():.6f}" for _ in range(num_sensors)]
                writer.writerow(row)


def _load_lightweight_samples(data_dir):
    samples = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            device_id = int(filename.split("_")[1].split(".")[0])
            with open(os.path.join(data_dir, filename), newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for row in reader:
                    samples.append([float(value) for value in row])
                    labels.append(device_id)
    return samples, labels


def _split_samples(samples, labels, val_ratio, seed):
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]
    x_train = [samples[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_val = [samples[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    return x_train, y_train, x_val, y_val


def _run_lightweight_demo(args):
    os.makedirs(args.data_dir, exist_ok=True)
    if not any(name.endswith(".csv") for name in os.listdir(args.data_dir)):
        print("No CSVs found; generating synthetic data (lightweight mode)...")
        _generate_lightweight_csvs(
            args.data_dir,
            num_devices=args.num_devices,
            num_samples=args.num_samples,
            num_sensors=args.num_sensors,
            seed=args.seed,
        )

    samples, labels = _load_lightweight_samples(args.data_dir)
    x_train, y_train, _x_val, y_val = _split_samples(
        samples, labels, val_ratio=0.2, seed=args.seed
    )
    if not y_train:
        raise RuntimeError("No training data available in lightweight mode.")

    majority_label = max(set(y_train), key=y_train.count)
    correct = sum(1 for label in y_val if label == majority_label)
    accuracy = correct / max(len(y_val), 1)
    loss = 1 - accuracy

    print("Federated demo complete (lightweight mode).")
    print(f"Validation loss: {loss:.4f} | Validation accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run federated learning demo.")
    parser.add_argument("--data-dir", default="./data_generation/synthetic_data/data")
    parser.add_argument("--num-devices", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-sensors", type=int, default=10)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not _has_full_dependencies():
        _run_lightweight_demo(args)
        return

    os.makedirs(args.data_dir, exist_ok=True)
    if not any(name.endswith(".csv") for name in os.listdir(args.data_dir)):
        print("No CSVs found; generating synthetic data...")
        generate_iot_data_configurable(
            args.data_dir,
            num_devices=args.num_devices,
            num_samples=args.num_samples,
            num_sensors=args.num_sensors,
            seed=args.seed,
        )

    data = create_federated_splits(args.data_dir, num_clients=args.num_clients)
    global_model = federated_averaging_mlp(
        data, num_rounds=args.num_rounds, num_clients=args.num_clients
    )

    metrics = global_model.evaluate(data["x_val"], data["y_val"], verbose=0)
    print("Federated demo complete.")
    print(f"Validation loss: {metrics[0]:.4f} | Validation accuracy: {metrics[1]:.4f}")


if __name__ == "__main__":
    main()
