import tensorflow as tf
from utils.data_loading import load_data, normalize_data
from model_training.local_training import create_model
from evaluation.evaluate_model import evaluate_model, print_evaluation_metrics
from utils.visualization import plot_training_history, plot_confusion_matrix

def experiment1(data_dir):
    # Load and normalize data
    data = load_data(data_dir)
    data['x_train'] = normalize_data(data['x_train'])
    data['x_val'] = normalize_data(data['x_val'])

    # Create model
    input_shape = data['x_train'].shape[1:]
    model = create_model(input_shape)

    # Train model
    history = model.fit(data['x_train'], data['y_train'], epochs=20, validation_data=(data['x_val'], data['y_val']))

    # Evaluate model
    metrics = evaluate_model(model, data['x_val'], data['y_val'])
    print_evaluation_metrics(metrics)

    # Plot training history
    plot_training_history(history)

    # Predict and plot confusion matrix
    y_pred_prob = model.predict(data['x_val'])
    y_pred = tf.argmax(y_pred_prob, axis=1)
    plot_confusion_matrix(data['y_val'], y_pred, labels=[f'Class {i}' for i in range(10)])

if __name__ == "__main__":
    data_dir = './data_generation/synthetic_data/data'
    experiment1(data_dir)
