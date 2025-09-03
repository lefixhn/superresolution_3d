import os
import numpy as np 
import nibabel as nib
from scipy import ndimage


SOURCE_PATH = ''
STORE_BASE_PATH = ''
STORE_LR_PATH = os.path.join(STORE_BASE_PATH, 'hr')
STORE_HR_PATH = os.path.join(STORE_BASE_PATH, 'lr')



# This is 
FILE_CHANNEL_INDICATOR = ''

def load_mri_as_nparray(load_path): 


def default_image_degradation(image: np.array, noise_sigma, blur_sigma):
    blurred_image =  


def preprocess_data(degradation_model=None): 
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
                image = (image-np.min(image)) / np.max(image)
                # Apply degradation model 
                if degradation_model is None: 
                    blur_sigma = 0.5
                    noise_sigma = 0.03
                    downscale_factor = 2

                    # Apply gaussian blur 
                    blurred_image = ndimage.gaussian_filter(imgae, sigma=blur_sigma)
                    # Scale down 
                    downscaled_image = ndimage.zoom(blurred_image, zoom=(1/downscale_factor, 1/downscale_factor, 1/downscale_factor), order = 3)
                    # Add noise 
                    noise = np.random.normal(0,noise_sigma, downscaled_image.shape)
                    downscaled_image = downscaled_image + noise


                    







if __name__ == '__main__': 
    preprocess_data()
