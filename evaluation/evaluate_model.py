from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model using accuracy, precision, recall, and F1-score.

    Args:
        model (tf.keras.Model): Trained model to evaluate.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test set.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics

def print_evaluation_metrics(metrics):
    """
    Print evaluation metrics.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    # Example usage
    # Load a sample test set (replace with actual test set)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # Load a sample model (replace with actual model)
    model = tf.keras.models.load_model('path_to_model.h5')

    metrics = evaluate_model(model, x_test, y_test)
    print_evaluation_metrics(metrics)
