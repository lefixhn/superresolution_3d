import numpy as np 
from torch.utils.data import Dataset
import torch 
import os 

DEFAULT_LR_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/advanced_degradation_t1/lr'
DEFAULT_HR_PATH = '/content/drive/MyDrive/superresolution_3d_data/datasets/advanced_degradation_t1/hr'

class PreloadedBratsDataset(Dataset): 
    def __init__(self,cube_side_length=None ,start_item_index=0, item_count=1251 ,lr_path=DEFAULT_LR_PATH, hr_path=DEFAULT_HR_PATH):
        super().__init__()
        assert item_count + start_item_index <= 1251

        self.lr_path = lr_path
        self.hr_path = hr_path
        self.cube_side_length = cube_side_length

        self.preloaded_data = [
            (self._get_file_as_tensor(f'{lr_path}/{i:04d}.npy'),
            self._get_file_as_tensor(f'{hr_path}/{i:04d}.npy'))
            for i in range(start_item_index, start_item_index+item_count)
        ]
        num_elements = len(self.preloaded_data)
        print(f"FINISHED PRELOADING OF {num_elements} ELEMENTS")

    

    def __len__(self): 
        return len(self.preloaded_data)
    
    def __getitem__(self, index) :
        data_tuple = self.preloaded_data[index]
        if self.cube_side_length is None: 
            return data_tuple 
        else:
            lr_shape = data_tuple[0].shape[1:]
            min_shape_dim = min(lr_shape)
            safe_side_length = int(min([min_shape_dim, self.cube_side_length]))
            # Select random position 
            x, y, z = np.random.randint(0, lr_shape[0] - safe_side_length + 1), np.random.randint(0, lr_shape[1] - safe_side_length + 1), np.random.randint(0, lr_shape[2] - safe_side_length + 1)
            # Select Sub cubes
            lr_tensor = data_tuple[0][:, x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]
            x, y, z, safe_side_length = 2*x, 2*y, 2*z, 2*safe_side_length
            hr_tensor = data_tuple[1][:, x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]
            return (lr_tensor, hr_tensor)

    
    def _get_file_as_tensor(self, path):
        # Load and convert to float32
        data = np.load(path)
        data = data.astype(np.float32, copy=False)
        tensor = torch.from_numpy(data)
        # Add dimension for channel 
        if tensor.ndim == 3: 
            tensor = tensor.unsqueeze(0) 
        return tensor 


class LazyLoadingDataset(Dataset): 
    def __init__(
        self,
        cube_side_length=None,
        start_item_index=0, 
        item_count=1251, 
        lr_path=DEFAULT_LR_PATH, 
        hr_path=DEFAULT_HR_PATH, 
        datatype=torch.float32
    ):
        super().__init__()
        self.cube_side_length=cube_side_length
        self.datatype = datatype
        # Init file paths
        self.lr_files = [os.path.join(lr_path, f"{i:04d}.npy") for i in range(start_item_index, start_item_index+item_count)]
        self.hr_files = [os.path.join(hr_path, f"{i:04d}.npy") for i in range(start_item_index, start_item_index+item_count)]
        
    def __len__(self):
        return len(self.hr_files)
    
    def _convert_to_4d_tensor(self, image: np.array):
        tensor = torch.from_numpy(image, dtype=self.datatype)
        if tensor.ndim == 3: 
            return tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            return tensor
        else: 
            raise Exception(f"The given .npy file has {image.ndim} dimensions")


    # 4d > (Channel, x, y, z)
    def __getitem__(self, index):
        # Use memory mapping to only load a smaller part of the file 
        lr_memory_map = np.load(self.lr_files[index], mmap_mode="r")
        hr_memory_map = np.load(self.hr_files[index], mmap_mode="r")
        # Return directly if cube_side_length is not set
        if self.cube_side_length is None: 
            lr_tensor = self._convert_to_4d_tensor(np.asarray(lr_memory_map, dtype=np.float32))
            hr_tensor = self._convert_to_4d_tensor(np.asarray(hr_memory_map, dtype=np.float32))
            return (lr_tensor, hr_tensor)
        
        
        lr_shape = data_tuple[0].shape[1:]
        min_shape_dim = min(lr_shape)
        safe_side_length = int(min([min_shape_dim, self.cube_side_length]))
        # Select random position 
        x, y, z = np.random.randint(0, lr_shape[0] - safe_side_length + 1), np.random.randint(0, lr_shape[1] - safe_side_length + 1), np.random.randint(0, lr_shape[2] - safe_side_length + 1)
        # Select Sub cubes
        lr_tensor = data_tuple[0][:, x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]
        x, y, z, safe_side_length = 2*x, 2*y, 2*z, 2*safe_side_length
        hr_tensor = data_tuple[1][:, x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]



        
        
        


