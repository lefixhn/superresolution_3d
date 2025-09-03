import os
import numpy as np 
import nibabel as nib
from scipy import ndimage


SOURCE_PATH = ''
STORE_PATH = ''

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

                if degradation_model is None: 
                    # Apply gaussian blur 
                    blurred_image = ndimage.gaussian_filter(imgae, sigma=blur_sigma)
                    # Scale down 
                    downscaled_image = ndimage.zoom(blurred_image, zoom=(1/do))

                    







if __name__ == '__main__': 
    preprocess_data()
