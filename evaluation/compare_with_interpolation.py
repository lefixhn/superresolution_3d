from torch.utils.data import Dataset
from torch.nn import Module
import torch 
import numpy as np
import scipy.ndimage as nd
#import comparing_metrics as cm
from typing import Dict, List, Tuple
from tqdm import tqdm
sys.path.append('/content/superresolution_3d/data_preprocessing')
from image_degradation import general_image_degradation_model

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


def print_metrics_of_models(results: Dict[str, Dict[str, float]]): 
    for model_name, model_results in results: 
        print(f"### {model_name}")
        metric_results_accumulated = ""
        for metric_name, metric_value in model_results: 
            metric_results_accumulated += f" {metric_name} : {metric_value} | "
        print(f"###{metric_results_accumulated}")

def print_results_of_degradations(results: Dict[str, Dict[str, Dict[str, float]]]): 
    for degradation_name, degradation_results in results: 
        print(f"DEGRADATION: {degradation_name}")
        print_metrics_of_models(degradation_results)


# Compares performance of models on given data
@torch.inference_mode()
def compare_models_performance(
    lr_hr_tuples: List[tuple], 
    models: Dict[str, callable], 
    metrics: Dict[str, callable], 
    autoprint: bool=False 
) -> Dict[str, Dict[str, float]]:
    '''
    Accepts 5d torch tensors as input images 
    '''
    # Generate dict for results
    models_results = {model_name : {metric_name : 0.0 for metric_name in metrics.keys()} for model_name in models.keys()}

    # Sum up the metric results for each model over the images
    for model_name, model in tqdm(models.items(), desc="Going trough models"): 
        target_device = 'cuda' if (torch.cuda.is_available() and isinstance(model, Module)) else 'cpu'
        if isinstance(model, Module): 
            model = model.to(target_device).eval()

        # TODO: Bring model and images to GPU if it is a Module
        # Bring it to eval mode if it is a module
        for (lr_image, hr_image) in lr_hr_tuples: 
            lr_image = lr_image.to(target_device)
            hr_image = hr_image.to(target_device)
            assert lr_image.dim() == 5 and hr_image.dim() == 5, "Tensors mus be 5d (B, C, D, H, W)"
            

            sr_image = model(lr_image)

            for metric_name, metric in metrics.items():
                models_results[model_name][metric_name] += metric(sr_image, hr_image)
    
    num_images = len(lr_hr_tuples)
    for model_name in models.keys(): 
        for metric_name in metrics.keys(): 
            models_results[model_name][metric_name] = models_results[model_name][metric_name] / num_images
    
    # Print the results 
    if autoprint: 
        print_metrics_of_models(models_results)

    return models_results          


def compare_models_on_degradation_models(
    models: Dict[str, callable], 
    hr_images: List[torch.Tensor], 
    degradation_models: Dict[str, callable]=build_degradations(), 
    metrics: Dict[str, callable], 
): 
    '''
    Comparing multiple models with multiple metrics on multiple degradation models.
    
    '''
    results = {}
    for degradation_model_name, degradation_model in degradation_models.items(): 
        lr_hr_tuples = []
        for hr_image in hr_images: 
            if hr_image.dim() == 4: 
                hr_image = hr_image.unsqueeze(0)
            lr_image = degradation_model(hr_image)
            lr_hr_tuple = (lr_image, hr_image)
            lr_hr_tuples.append(lr_hr_tuple)
        results[degradation_model_name] = compare_models_performance(
            lr_hr_tuples, 
            models=models,  
            metrics=metrics, 
            autoprint=False 
        )
    
    return results


# TODO: Implement following methods 

def build_degradations(blur_sigmas: List[float]=[0.0, 12.0/255.0, 25.0/255.0], noise_sigmas: List[float]=[0.0, 1.2, 2.4], downscale_factor=2) -> Dict[str, callable]: 
    degradations = {}
    for blur_sigma in blur_sigmas: 
        for noise_sigma in noise_sigmas: 
            degradation_name = f"GeneralDegradation - noise_sigma:{noise_sigma} | blur_sigma:{blur_sigma}"
            degradations[degradation_name]= lambda image: general_image_degradation_model(image=image, downscale_factor=downscale_factor)

def compare_model_with_interpolations_on_degradations() -> Dict[str, Dict[str, Dict[str, float]]]:
    return None