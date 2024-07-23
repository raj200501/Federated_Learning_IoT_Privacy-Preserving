import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        labels (list): List of label names.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.

    Args:
        history (tf.keras.callbacks.History): Training history.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    # Example usage
    # Sample data (replace with actual data)
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 1, 0])
    labels = ['Class 0', 'Class 1', 'Class 2']

    plot_confusion_matrix(y_true, y_pred, labels)

    # Example history (replace with actual training history)
    class History:
        def __init__(self):
            self.history = {
                'accuracy': [0.1, 0.2, 0.3, 0.4],
                'val_accuracy': [0.15, 0.25, 0.35, 0.45],
                'loss': [2.0, 1.5, 1.0, 0.5],
                'val_loss': [1.8, 1.4, 0.9, 0.4]
            }

    history = History()
    plot_training_history(history)
