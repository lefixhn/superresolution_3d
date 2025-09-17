from torch.utils.data import Dataset
from torch.nn import Module
import torch 
import numpy as np
import scipy.ndimage as nd



def convert_tensor_to_numpy(tensor) -> np.array: 
    ''' Only accepts 5d (B, C, D, H, W) tensors'''
    assert len(tensor.shape) == 5   # Ensure we have 3D Data
    tensor = tensor.squeeze(0)
    tensor = tensor.squeeze(0)
    return tensor.numpy()

def interpolate_tensor(tensor, upscale_factor: int = 2, order : int =3): 
    ''' Only works with one channel (B, 1, D, H, W) '''
    assert tensor.shape[1] == 1, "Tensor can only have one channel dimension"
    batch_size = tensor.shape[0]
    interpolated_tensor_images = []
    for batch_index in range(batch_size): 
        numpy_image = convert_tensor_to_numpy(tensor[batch_index, :, :, :, :].unsqueeze(0))
        interpolated_numpy_image = nd.zoom(numpy_image, zoom=upscale_factor, order=order)
        interpolated_tensor_image = torch.as_tensor(interpolated_numpy_image).unsqueeze(0).unsqueeze(0)
        interpolated_tensor_images.append(interpolated_tensor_image)
    interpolated_tensor = torch.cat(interpolated_tensor_images, dim=0)
    return interpolated_tensor


def compare_models(
    lr_hr_tuples, 
    models: , 
    require_compare_mae=True, 
    require_compare_mse=True, 
    require_compare_psnr=True, 
    require_compare_ssim=True, 
    require_compare_lpips=True, 
    
):
    '''
    Accepts 5d torch tensors as input images 
    '''
    

