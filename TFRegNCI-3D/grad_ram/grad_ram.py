import numpy as np
from grad_ram.base_ram import BaseRAM

class GradRAM(BaseRAM):
    def __init__(self, model, target_layers, use_cuda=False,  reshape_transform=None):
        super(
            GradRAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_ram_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3, 4))
