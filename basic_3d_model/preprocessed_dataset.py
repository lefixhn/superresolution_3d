from torch.utils.data import Dataset
import numpy as np 
import torch 

class PreprocessedDataset(Dataset): 
    '''Expects the given paths to direct to .npy files. '''
    def __init__(self, 
    hr_paths=[f'/content/drive/MyDrive/superresolution_3d_data/datasets/superresolution_brats/hr_images/hr_image{i:03d}.npy' for i in range(0, 484)], 
    lr_paths=[f'/content/drive/MyDrive/superresolution_3d_data/datasets/superresolution_brats/lr_images/lr_image{i:03d}.npy' for i in range(0, 484)]
    ): 
        super().__init__()
        print("PRELOADING DATA")
        assert len(lr_paths) == len(hr_paths)
        self.data = [
            (self._get_image_as_tensor_(lr_paths[i]), 
            self._get_image_as_tensor_(hr_paths[i]) )
            for i in range(0, len(hr_paths)
        ]
        print("DATA LOADED")

    def __getitem__(self, index): 
        return self.data[index]

    def __len__(self): 
        return len(self.data)
    
    def _get_image_as_tensor_(self, path): 
        image = np.load(path)
        image = image.astype(np.float32, copy=False)
        tensor = torch.from_numpy(image)
        # Add channel dimension 
        if tensor.ndim == 3: 
            tensor = tensor.unsqueeze(0)
        return tensor
