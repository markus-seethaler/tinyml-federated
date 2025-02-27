"""
Neural network configuration for the federated learning system.
"""

class NNConfig:
    # Network Architecture - should match Arduino client
    LAYERS = [11, 60, 3]
    
    @classmethod
    def calculate_total_weights(cls):
        """Calculate the total number of weights in the neural network."""
        return sum(cls.LAYERS[i] * cls.LAYERS[i + 1] 
                  for i in range(len(cls.LAYERS) - 1))
    
    @classmethod
    def get_layer_sizes(cls):
        """Return a list of (inputs, outputs) for each layer."""
        return [(cls.LAYERS[i], cls.LAYERS[i+1]) 
                for i in range(len(cls.LAYERS) - 1)]