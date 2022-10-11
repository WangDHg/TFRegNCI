import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import math
import torch.nn.functional as F

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return img

def scale_ram_image(ram, target_size=None):
    result = []

    for img in ram:
        img = img - np.min(img)
        img = img / (1e-9 + np.max(img))
        result.append(img)
    result = np.float32(result)

    if target_size is not None:
        result = result[:,None]

        result = F.interpolate(torch.tensor(result), size=target_size, mode='trilinear')
        result = result.numpy()

    return result

def scale_accross_batch_and_channels(tensor, target_size):
    batch_size, channel_size = tensor.shape[:2]
    reshaped_tensor = tensor.reshape(
        batch_size * channel_size, *tensor.shape[2:])
    result = scale_ram_image(reshaped_tensor, target_size)
    result = result.reshape(
        batch_size,
        channel_size,
        target_size[1],
        target_size[0])
    return result
