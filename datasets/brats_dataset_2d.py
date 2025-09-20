from torch.utils.data import Dataset
impot numpy as np 
import torch 
import os 

class LazyLoadingBratsDataset2d(Dataset): 
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
        # Only load the paths 
        self.lr_files = [os.path.join(lr_path, f"{i:04d}.npy") for i in range(start_item_index, start_item_index+item_count)]
        self.hr_files = [os.path.join(hr_path, f"{i:04d}.npy") for i in range(start_item_index, start_item_index+item_count)]
        print("FINISHED INITIALIZING LazyLoadingBratsDataset")

    def __len__(self):
        return len(self.hr_files)
    
    def _get_slice_from_volume(self, volume: np.ndarray): 
        # Predetermines weather we make an axial, coronal or sagital slice 
        orientation_index = np.random.randint(3)    # 0-2
        slice_index = np.random.randint()


    def _convert_to_3d_tensor(self, image: np.ndarray):
        
        tensor = torch.as_tensor(image, dtype=torch.float32)
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
        else:
            lr_shape = lr_memory_map.shape
            safe_side_length = int(min(list(lr_shape) + [self.cube_side_length]))
            x = np.random.randint(0, lr_shape[0] - safe_side_length + 1)
            y = np.random.randint(0, lr_shape[1] - safe_side_length + 1)
            z = np.random.randint(0, lr_shape[2] - safe_side_length + 1)
            # Fetch required information from the lr memory map 
            lr_image = lr_memory_map[x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]
            # Calculate the hr_coordinates
            x, y, z, safe_side_length = 2*x, 2*y, 2*z, 2*safe_side_length
            # Fetch required information from the hr memory map 
            hr_image = hr_memory_map[x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]
            # Convert to tensor
            lr_tensor = self._convert_to_4d_tensor(lr_image)
            hr_tensor = self._convert_to_4d_tensor(hr_image)
            return (lr_tensor, hr_tensor)



      