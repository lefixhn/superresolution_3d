import os
from functools import partial
import numpy as np 
import nibabel as nib
from scipy import ndimage
import torch 

np.random.seed(42)

def crop_image_for_downscale(image, downscale_factor): 
    '''Changes the shape of an image, to make all dimensions dividable by 
    downscale_factor'''
    assert len(image.shape) <= 3
    # List of integers > size of the dimensions
    cropped_shape = [(dimension - (dimension % downscale_factor)) for dimension in image.shape]
    # How much is cropped away
    cropped_overhead = [(dimension % downscale_factor) for dimension in image.shape]
    # Half it to crop both sides equaly
    crop_offset = [overhead//2 for overhead in cropped_overhead]
    # Slice is like the start:stop:step syntax
    cropped_ranges = tuple(slice(crop_offset[i], crop_offset[i] + cropped_shape[i]) for i in range(len(cropped_shape)))
    cropped_image = image[cropped_ranges]
    for dimension in cropped_image.shape:
        assert dimension % downscale_factor == 0
    return cropped_image


# Define downscale functions 
def nearest_neighbor_downscale(image, downscale_factor): 
    return ndimage.zoom(image, zoom=1/downscale_factor, order=0)

def linear_downscale(image, downscale_factor): 
    return ndimage.zoom(image, zoom=1/downscale_factor, order=1)

def cubic_downscale(image, downscale_factor): 
    return ndimage.zoom(image, zoom=1/downscale_factor, order=3)


# Define blur function 
def apply_gaus_blur(image, blur_sigma): 
    blurred_image = ndimage.gaussian_filter(image, sigma=blur_sigma, truncate=3.0)
    return blurred_image

def add_gaus_noise(image, noise_sigma): 
    # Add gaussian noise 
    noise = np.random.normal(0,noise_sigma, image.shape)
    noised_image = image + noise 
    # Clip values outside 0-1
    noised_image = np.clip(noised_image, 0, 1)
    return noised_image

# Works in 3d on np arrays and tensors 
def general_image_degradation_model_on_3d_nparray(image: np.array,noise_sigma, downscale_function ,downscale_factor=2 ,blur_sigma = 0):
    assert image.ndim == 3, "Image must be in 3D"
    degradation_image = image
    # Apply gaussian blur 
    if blur_sigma != 0: 
        degradation_image = ndimage.gaussian_filter(degradation_image, sigma=blur_sigma, truncate=3.0)
    # Scale down 
    degradation_image = crop_image_for_downscale(degradation_image, downscale_factor)
    if downscale_function is not None: 
        degradation_image = downscale_function(degradation_image, downscale_factor)
    else: 
        degradation_image = cubic_downscale(degradation_image, downscale_factor)
    degradation_image = add_gaus_noise(degradation_image, noise_sigma=noise_sigma)

    return degradation_image.astype(np.float32, copy=False)


def general_image_degradation_model(image,noise_sigma, downscale_function ,downscale_factor=2 ,blur_sigma = 0): 
    if isinstance(image, torch.Tensor):
        if image.dim() == 4: 
            image = image.unsqueeze(0)
        assert image.dim() == 5 and image.shape[image.dim() - 4] == 1, f"Tensor must have shape (1, D, H, W) or (B, 1, D, H, W) but has shape {image.shape}"
        degradated_images = []
        for batch_index in range(image.shape[0]): 
            single_image = image[batch_index, :, :, :, :]
            single_image = single_image.squeeze(0)
            degradated_images.append(
                torch.as_tensor(
                    general_image_degradation_model_on_3d_nparray(single_image.detach().cpu().numpy() ,noise_sigma, downscale_function, downscale_factor, blur_sigma)
                ).unsqueeze(0).unsqueeze(0))
        degradated_images = torch.cat(degradated_images, dim=0)
        return degradated_images
    else: 
        return general_image_degradation_model_on_3d_nparray(image, noise_sigma, downscale_function, downscale_factor, blur_sigma)



# Simple random settings for the general image degradation model 
def default_degradation(image, downscale_factor):
    # Generate number from 1/255 to 25/255
    noise_sigma = np.random.randint(1, 26) / 255
    blur_sigma = 0
    downscale_functions = (nearest_neighbor_downscale, linear_downscale, cubic_downscale) 
    # Select random downscale function 
    downscale_function = downscale_functions[np.random.randint(0, 3)]
    return general_image_degradation_model_on_3d_nparray(image, noise_sigma, downscale_function, downscale_factor, blur_sigma) 


def advanced_image_degradation_model(image, downscale_factor, sfulle_operations=False):
    '''
    Here we apply the degradation operations in random order 
    ''' 
    degradation_image = crop_image_for_downscale(image, downscale_factor)
    noise_sigma = np.random.randint(1, 26) / 255  
    blur_sigma = None
    if downscale_factor > 2:
        blur_sigma = np.random.uniform(0.1, 2.8) 
    else: 
        blur_sigma = np.random.uniform(0.1, 2.4) 
    # Select downscale operation 
    downscale_operations = (nearest_neighbor_downscale, linear_downscale, cubic_downscale)
    downscale_operation = downscale_operations[np.random.randint(0, 3)]

    degradation_operations = [
        partial(apply_gaus_blur, blur_sigma=blur_sigma),
        partial(downscale_operation, downscale_factor = downscale_factor),
        partial(add_gaus_noise, noise_sigma=noise_sigma)
    ]
    if sfulle_operations: 
        # Apply operations in random order 
        np.random.shuffle(degradation_operations)

    for operation in degradation_operations: 
        degradation_image = operation(degradation_image)
    
    return degradation_image

