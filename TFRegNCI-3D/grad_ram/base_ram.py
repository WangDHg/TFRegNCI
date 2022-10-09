import numpy as np
from pkg_resources import require
import torch
import ttach as tta
from typing import Callable, List, Tuple
from grad_ram.activations_and_gradients import ActivationsAndGradients
from grad_ram.utils.svd_on_activations import get_2d_projection
from grad_ram.utils.image import scale_ram_image

def loss_func(x, label):
    mse = torch.mean(torch.pow((x-label),2))
    rmse = mse.sqrt()
    return rmse

class BaseRAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    def get_ram_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_ram_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_ram_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        
        weighted_activations = weights[:, :, None, None, None] * activations
        
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                chem_input: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()
            chem_input = chem_input.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)
            chem_input = torch.autograd.Variable(chem_input, 
                                                 requires_grad=True)

        outputs = self.activations_and_grads(input_tensor, chem_input)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([1/(1e-9+loss_func(output, target)) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        ram_per_layer = self.compute_ram_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(ram_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        length, width, height = input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)
        # return width, height
        return length, width, height

    def compute_ram_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        ram_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            ram = self.get_ram_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            ram = np.maximum(ram, 0)
            scaled = scale_ram_image(ram, target_size)
            ram_per_target_layer.append(scaled[:, None, :])

        return ram_per_target_layer

    def aggregate_multi_layers(self, ram_per_target_layer: np.ndarray) -> np.ndarray:
        ram_per_target_layer = np.concatenate(ram_per_target_layer, axis=1)
        ram_per_target_layer = np.maximum(ram_per_target_layer, 0)
        result = np.mean(ram_per_target_layer, axis=1)
        return scale_ram_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        rams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            ram = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            ram = ram[:, None, :, :, :]
            ram = torch.from_numpy(ram)
            ram = transform.deaugment_mask(ram)

            # Back to numpy float32, HxW
            ram = ram.numpy()
            ram = ram[:, 0, :, :, :]
            rams.append(ram)

        ram = np.mean(np.float32(rams), axis=0)
        return ram

    def __call__(self,
                 input_tensor: torch.Tensor,
                 chem_input: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor,chem_input, targets, eigen_smooth)

        return self.forward(input_tensor,chem_input ,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
