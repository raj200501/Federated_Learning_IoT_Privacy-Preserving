import tensorflow as tf
from utils.data_loading import load_data

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_mlp_model(input_shape, num_classes):
    """
    Create a lightweight MLP model for tabular IoT data.

    Args:
        input_shape (tuple): Shape of a single sample (num_features,).
        num_classes (int): Number of output classes.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    data = load_data('./data_generation/synthetic_data/data')
    input_shape = data['x_train'].shape[1:]
    model = create_model(input_shape)
    model.fit(data['x_train'], data['y_train'], epochs=10, validation_data=(data['x_val'], data['y_val']))

if __name__ == "__main__":
    main()
