from torch.utils.data import Dataset
from torch.nn import Module
import torch 
import numpy as np
import scipy.ndimage as nd

def convert_to_numpy(tensor) -> np.array: 
    assert len(tensor.shape) == 5   # Ensure we have 3D Data
    tensor = tensor.squeeze(0)
    tensor = tensor.squeeze(0)
    return tensor.numpy()

def compare_model_with_interpolation(evaluation_dataset: Dataset, model: Module, upscale_factor: int = 2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    for lr_image, hr_image in evaluation_dataset: 
        lr_image_np = convert_to_numpy(lr_image)
        linear_image = nd.zoom(lr_image_np, zoom=upscale_factor, order=1)
        cubic_image = nd.zoom(lr_image_np, zoom=upscale_factor, order=3)

        model.to(device)
        sr_image = model(lr_image)
