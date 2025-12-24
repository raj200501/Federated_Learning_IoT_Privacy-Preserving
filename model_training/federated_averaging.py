from model_training.local_training import create_model
from model_training.model_aggregation import aggregate_models
from utils.data_loading import load_data
from model_training.local_training import create_mlp_model

def federated_averaging(data, num_rounds=10, num_clients=10):
    global_model = create_model(data['x_train'].shape[1:])
    client_models = [create_model(data['x_train'].shape[1:]) for _ in range(num_clients)]

    for round_num in range(num_rounds):
        for client_id in range(num_clients):
            client_data = {
                'x_train': data[f'x_train_client_{client_id}'],
                'y_train': data[f'y_train_client_{client_id}'],
                'x_val': data[f'x_val_client_{client_id}'],
                'y_val': data[f'y_val_client_{client_id}']
            }
            client_models[client_id].fit(client_data['x_train'], client_data['y_train'], epochs=1)
        
        global_model = aggregate_models(client_models)

    return global_model

if __name__ == "__main__":
    data = load_data('./data_generation/synthetic_data/data')
    global_model = federated_averaging(data)
    print("Federated Averaging completed and global model is trained.")


def federated_averaging_mlp(data, num_rounds=5, num_clients=5):
    """
    Federated averaging using an MLP model for tabular IoT data.

    Args:
        data (dict): Data dictionary with per-client splits.
        num_rounds (int): Number of federated rounds.
        num_clients (int): Number of clients.
    """
    num_classes = len(set(data["y_train"]))
    global_model = create_mlp_model((data["x_train"].shape[1],), num_classes)
    client_models = [
        create_mlp_model((data["x_train"].shape[1],), num_classes) for _ in range(num_clients)
    ]

    for _ in range(num_rounds):
        for client_id in range(num_clients):
            client_models[client_id].fit(
                data[f"x_train_client_{client_id}"],
                data[f"y_train_client_{client_id}"],
                epochs=1,
                verbose=0,
            )
        global_model = aggregate_models(client_models)

    return global_model
