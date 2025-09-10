from torch.utils.data import Dataset
from torch.nn import Module
import torch 
import numpy as np

def convert_to_numpy(tensor) -> np.array: 
    assert len(tensor.shape) == 5   # Ensure we have 3D Data
    tensor = tensor.squeeze(0)
    tensor = tensor.squeeze(0)

def compare_model_with_interpolation(evaluation_dataset: Dataset, model: Module):
