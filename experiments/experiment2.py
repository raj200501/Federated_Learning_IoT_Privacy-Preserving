import tensorflow as tf
from utils.data_loading import load_data, normalize_data
from model_training.federated_averaging import federated_averaging
from evaluation.evaluate_model import evaluate_model, print_evaluation_metrics
from utils.visualization import plot_training_history, plot_confusion_matrix

def experiment2(data_dir):
    # Load and normalize data
    data = load_data(data_dir)
    for key in data.keys():
        if 'x' in key:
            data[key] = normalize_data(data[key])

    # Federated learning
    global_model = federated_averaging(data, num_rounds=20, num_clients=10)

    # Evaluate global model
    metrics = evaluate_model(global_model, data['x_val'], data['y_val'])
    print_evaluation_metrics(metrics)

    # Predict and plot confusion matrix
    y_pred_prob = global_model.predict(data['x_val'])
    y_pred = tf.argmax(y_pred_prob, axis=1)
    plot_confusion_matrix(data['y_val'], y_pred, labels=[f'Class {i}' for i in range(10)])

if __name__ == "__main__":
    data_dir = './data_generation/synthetic_data/data'
    experiment2(data_dir)
