import nibabel as nib 
from torch.utils.data import Dataset
from typing import Tuple
import torch 
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import random
import numpy as np


def image_degradation(image: np.ndarray, scale_factor=2,blur_sigma=0, noise_sigma=0) -> np.ndarray:
    '''scale_factor 2 means, that resulting image will be half the 
    size of the original '''
    # Add blur if nessecary
    if blur_sigma > 0: 
        image = gaussian_filter(image, sigma=blur_sigma)
    # Tricubic downscaling 
    lr_image = zoom(image, 1/scale_factor, order=3)
    if noise_sigma > 0: 
        noise = np.random.normal(0, noise_sigma, lr_image)
        lr_image = lr_image + noise
        lr_image = np.clip(lr_image, 0, 1)
    return lr_image
    
def generate_paths(start_index = 3, end_index = 400): 
    return [f'/content/drive/MyDrive/superresolution_3d_data/datasets/imagesTr/BRATS_{i:03d}.nii.gz' for i in range(start_index, end_index)]


class Dataset3DMri(Dataset):

    def __init__(self, paths=generate_paths(), downscale_factor = 2, cube_side_length = 64, channel=0):
        super().__init__()
        assert cube_side_length % downscale_factor == 0
        self.channel = channel
        self.paths = paths
        self.downscale_factor = downscale_factor
        self.cube_side_length = cube_side_length

    def _image_to_tensor(self, image):
        return torch.from_numpy(image)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]: 

        path = self.paths[index]
        mri = nib.load(path)
        mri = mri.get_fdata()
        # Reduce one dimension 
        mri = mri[:, :, :, self.channel]
        # Normalize 0-1
        max_val = mri.max()
        if max_val > 0: 
            mri = mri / max_val
        # Cast from double to float 
        mri = mri.astype(np.float32)
        
        # Get sub sub cube from image 
        rand_pos = [random.randint(0, dlength-self.cube_side_length) for dlength in mri.shape]
        mri = mri[rand_pos[0]:(rand_pos[0]+self.cube_side_length), rand_pos[1]:(rand_pos[1]+self.cube_side_length), rand_pos[2]:(rand_pos[2]+self.cube_side_length)]

        mri_downsampled = image_degradation(mri, scale_factor = self.downscale_factor)

        return self._image_to_tensor(mri_downsampled).unsqueeze(0), self._image_to_tensor(mri).unsqueeze(0)

    def __len__(self): 
        return len(self.paths)

if __name__ == '__main__': 
    print("Initializing Dataset3DMri")
    dataset = Dataset3DMri()
    lr_image, hr_image = dataset[10]
    print(str(lr_image.shape) + " " + str(hr_image.shape))

    