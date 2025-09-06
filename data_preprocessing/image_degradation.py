import os
import numpy as np 
import nibabel as nib
from scipy import ndimage
 

def crop_image_for_downscale(image, downscale_factor): 
    '''Changes the shape of an image, to make all dimensions dividable by 
    downscale_factor'''
    cropped_shape = [(dimension - (dimension % downscale_factor)) for dimension in image.shape]
    cropped_ranges = tuple(slice(0, cropped_dimension) for cropped_dimension in cropped_shape)
    cropped_image = image[cropped_ranges]
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
    

def add_gaus_noise(image, noise_sigma): 
    # Add gaussian noise 
    noise = np.random.normal(0,noise_sigma, image.shape)
    image = image + noise 
    # Clip values outside 0-1
    image = np.clip(image, 0, 1)
    return image

def general_image_degradation_model(image: np.array ,noise_sigma, downscale_function ,downscale_factor=2 ,blur_sigma = 0):
    degradation_image = image
    # Apply gaussian blur 
    if blur_sigma != 0: 
        degradation_image = ndimage.gaussian_filter(image, sigma=blur_sigma)
    # Scale down 
    degradation_image = crop_image_for_downscale(degradation_image, downscale_factor)
    degradation_image = downscale_function(degradation_image, downscale_factor)
    
    return degradation_image

# Simple random settings for the general image degradation model 
def default_degradation(image, downscale_factor):
    # Generate number from 1/255 to 25/255
    noise_sigma = np.random.randint(1, 26) / 255
    blur_sigma = 0
    downscale_functions = (nearest_neighbor_downscale, linear_downscale, cubic_downscale) 
    # Select random downscale function 
    downsale_function = downscale_functions[np.random.randint(0, 3)]
    return general_image_degradation_model(image, noise_sigma, downsale_function, downscale_factor, blur_sigma) 


def advanced_image_degradation_model(image, downscale_factor):
    '''
    Here we apply the degradation operations in random order 
    ''' 

     



