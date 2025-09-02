import os
import numpy as np 
import nibabel as nib
SOURCE_PATH = ''
STORE_PATH = ''

# This is 
FILE_CHANNEL_INDICATOR = ''

def load_mri_as_nparray(load_path): 


def image_degradation(image: np.array): 


def preprocess_data(): 
    # Iterate though subfolders
    for sub_folder in os.listdir(SOURCE_PATH):
        sub_folder_path = os.path.join(SOURCE_PATH, sub_folder)
        # Search correct file in folder 
        for sub_file in os.listdir(sub_folder_path): 
            if FILE_CHANNEL_INDICATOR in sub_file:
                sub_file_path = os.path.join(sub_folder_path, sub_file)





if __name__ == '__main__': 
    preprocess_data()
