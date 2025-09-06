import os
import numpy as np 
import nibabel as nib
from scipy import ndimage


SOURCE_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/BraTS2021_Training_Data'
STORE_BASE_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/degradation_v0'
STORE_LR_PATH = os.path.join(STORE_BASE_PATH, 'lr')
STORE_HR_PATH = os.path.join(STORE_BASE_PATH, 'hr')



# This ensures that the correct mri channel is loaded
FILE_CHANNEL_INDICATOR = 'flair'

def crop_image_for_downscale(image, downscale_factor): 
    '''Changes the shape of an image, to make all dimensions dividable by 
    downscale_factor'''
    cropped_shape = [(dimension - (dimension % downscale_factor)) for dimension in image.shape]
    cropped_ranges = tuple(slice(0, cropped_dimension) for cropped_dimension in cropped_shape)
    cropped_image = image[cropped_ranges]
    return cropped_image


def image_degradation(image: np.array ,noise_sigma, downscale_function ,downscale_factor=2 ,blur_sigma = 0):
    degradation_image = image
    # Apply gaussian blur 
    if blur_sigma != 0: 
        degradation_image = ndimage.gaussian_filter(image, sigma=blur_sigma)
    # Scale down 
    degradation_image = crop_image_for_downscale(degradation_image, downscale_factor)
    degradation_image = downscale_function(degradation_image, downscale_factor)
    # Add gaussian noise 
    noise = np.random.normal(0,noise_sigma, degradation_image.shape)
    degradation_image = degradation_image + nosie 
    # Clip values outside 0-1
    degradation_image = np.clip(degradation_image, 0, 1)
    return degradation_image
    
def scale_down_with_order(image, downscale_factor, order): 
    return ndimage.zoom(image, zoom=1/downscale_factor, order=order)

def default_degradation(image, downscale_factor):
    # Generate number from 1/255 to 25/255
    noise_sigma = np.random.randint(1, 26) / 255
    blur_sigma = 0
    downscale_orders = (0, 1, 3) # 0 
    downsale_order = downscale_orders[np.random.randint(0, 3)]


def preprocess_data(degradation_model=None, item_limit=None): 
    element_counter = 0
    # Ensure folders exist
    os.makedirs(STORE_LR_PATH, exist_ok=True)
    os.makedirs(STORE_HR_PATH, exist_ok=True)

    # Iterate though subfolders
    for sub_folder in os.listdir(SOURCE_PATH):
        sub_folder_path = os.path.join(SOURCE_PATH, sub_folder)
        
        # Search correct file in folder 
        for sub_file in os.listdir(sub_folder_path): 
            if FILE_CHANNEL_INDICATOR in sub_file:
                sub_file_path = os.path.join(sub_folder_path, sub_file)
                image_loaded = nib.load(sub_file_path)
                image = image_loaded.get_fdata().astype(np.float32)
                # Normalize 0-1
                image = (image-np.min(image)) / (np.max(image)- np.min(image))
                # Apply degradation model 
                lr_image = None
                if degradation_model is None: 
                    
                    

                    
                    
                else: 
                    image = crop_image_for_downscale(image, downscale_factor)
                    lr_iamge = degradation_model(image)

                # Store LR image as np
                np.save(os.path.join(STORE_LR_PATH, f'{element_counter}.npy'), lr_image)
                # Store HR image as np
                np.save(os.path.join(STORE_HR_PATH, f'{element_counter}.npy'), image)

                element_counter = element_counter + 1 
        if item_limit is not None: 
            if item_limit < element_counter: 
                break



if __name__ == '__main__': 
    preprocess_data(item_limit=3)
