import numpy as np

def aggregate_models(models):
    aggregated_model = models[0]
    for layer in range(len(aggregated_model.layers)):
        weights = np.mean([model.layers[layer].get_weights() for model in models], axis=0)
        aggregated_model.layers[layer].set_weights(weights)
    return aggregated_model
