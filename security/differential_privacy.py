import numpy as np

def add_differential_privacy(model_weights, epsilon=1.0):
    noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=model_weights.shape)
    return model_weights + noise

def apply_differential_privacy(models, epsilon=1.0):
    for model in models:
        for layer in model.layers:
            weights, biases = layer.get_weights()
            weights = add_differential_privacy(weights, epsilon)
            biases = add_differential_privacy(biases, epsilon)
            layer.set_weights([weights, biases])
    return models
