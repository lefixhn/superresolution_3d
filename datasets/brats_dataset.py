import numpy as np 
from torch.utils.data import Dataset
import torch 
import os 

DEFAULT_LR_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/advanced_degradation_v0/lr'
DEFAULT_HR_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/advanced_degradation_v0/hr'

class PreloadedBratsDataset(Dataset): 
    def __init__(self, start_item_index=0, item_count=1251 ,lr_path=DEFAULT_LR_PATH, hr_path=DEFAULT_HR_PATH):
        super().__init__()
        assert item_count + start_item_index <= 1251

        self.lr_path = lr_path
        self.hr_path = hr_path

        self.preloaded_data = [
            (self._get_file_as_tensor(f'{lr_path}/{i:04d}.npy'),
            self._get_file_as_tensor(f'{hr_path}/{i:04d}.npy'))
            for i in range(start_item_index, start_item_index+item_count)
        ]
    

    def __len__(self): 
        return len(self.preloaded_data)
    
    def __getitem__(self, index) :
        return self.preloaded_data[index]
    
    def _get_file_as_tensor(self, path):
        # Load and convert to float32
        data = np.load(path)
        data = data.astype(np.float32, copy=False)
        tensor = torch.from_numpy(data)
        # Add dimension for channel 
        if tensor.ndim == 3: 
            tensor = tensor.unsqueeze(0) 
        return tensor 



        


