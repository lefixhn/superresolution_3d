from torch.utils.data import Dataset
from torch.nn import Module
import torch 
import numpy as np
import scipy.ndimage as nd
import comparing_metrics as cm


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
    models: Dict[str, callable], 
    metrics: Dict[str, callable]
):
    '''
    Accepts 5d torch tensors as input images 
    '''
    # Generate dict for results
    models_results = {model_name : {metric_name : 0.0 for metric_name in metrics.keys()} for model_name in models.keys()}

    # Sum up the metric results over the models     
    for model_name, model in models.items(): 
        # TODO: Bring model and images to GPU if it is a Module
        # Bring it to eval mode if it is a module
        for (lr_image, hr_image) in lr_hr_tuples: 
            assert lr_image.dim() == 5 and hr_image.dim() == 5, "Tensors mus be 5d (B, C, D, H, W)"
                
            sr_image = model(lr_image)

            for metric_name, metric in metrics.items():
                models_results[model_name][metric_name] += metric(sr_image, hr_image)
    
               
