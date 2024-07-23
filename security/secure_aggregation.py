import numpy as np

def secure_aggregation(models):
    num_models = len(models)
    aggregated_model = models[0]
    for layer in range(len(aggregated_model.layers)):
        weights = np.zeros_like(aggregated_model.layers[layer].get_weights()[0])
        for model in models:
            weights += model.layers[layer].get_weights()[0]
        aggregated_weights = weights / num_models
        aggregated_model.layers[layer].set_weights([aggregated_weights])
    return aggregated_model
