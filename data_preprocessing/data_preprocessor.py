import os
import numpy as np 
import nibabel as nib
from scipy import ndimage
import image_degradation as imd

SOURCE_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/BraTS2021_Training_Data'
STORE_BASE_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/advanced_degradation_v0'
STORE_LR_PATH = os.path.join(STORE_BASE_PATH, 'lr')
STORE_HR_PATH = os.path.join(STORE_BASE_PATH, 'hr')

# This ensures that the correct mri channel is loaded
FILE_CHANNEL_INDICATOR = 'flair'


def preprocess_data(degradation_model=None, item_limit=None, downscale_factor=2): 
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
                # Crop to image to make dividable by downscale_factor
                image = imd.crop_image_for_downscale(image, downscale_factor=downscale_factor)
                # Apply degradation model 
                lr_image = degradation_model(image=image, downscale_factor=downscale_factor)
                # Store LR image as np
                np.save(os.path.join(STORE_LR_PATH, f'{element_counter}.npy'), lr_image)
                # Store HR image as np
                np.save(os.path.join(STORE_HR_PATH, f'{element_counter}.npy'), image)

                element_counter = element_counter + 1 
        if item_limit is not None: 
            if item_limit < element_counter: 
                break


if __name__ == '__main__': 
    preprocess_data(degradation_model=imd.advanced_image_degradation_model, item_limit=10)
