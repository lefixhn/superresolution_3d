from torch.utils.data import Dataset
impot numpy as np 
import torch 
import os 



'''
    PROBLEM WITH USAGE OF 3D Data   

    We have two layers in the hr volume, that would 
    correspond to the layer in the lr volume

    -> Preprocess for 2d??? 
    -> Make sure to select one of both randomly

    LR      HR
    0   *2  0
            1
    1   *2  2
            3
    2   *2  4
            5
    3   *2  6
            7
    
'''




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
    
    def _get_slices_from_volumes(self, lr_volume: np.ndarray, hr_volume: np.ndarray): 
        # Predetermines weather we make an axial, coronal or sagital slice 
        orientation_index = np.random.randint(3)    # 0-2
        lr_slice_index = np.random.randint(lr_volume.shape[orientation_index])
        # Random from 0-1 ensures we select the corresponding layer randomly 
        hr_slice_index = 2 * lr_slice_index + np.random.randint(1)

        lr_sclice_selection = tuple(slice(lr_slice_index if i == orientation_index else None) for i in range(3))
        hr_sclice_selection = tuple(slice(hr_slice_index if i == orientation_index else None) for i in range(3))

        return lr_volume[lr_sclice_selection], hr_volume[hr_sclice_selection]

    def _convert_to_3d_tensor(self, image: np.ndarray):
        ''' Takes an 2d image and converts to (C, W, H)  '''
        tensor = torch.as_tensor(image, dtype=torch.float32)
        if tensor.ndim == 2: 
            return tensor.unsqueeze(0)
        elif tensor.ndim == 3:
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
            lr_slice, hr_slice = self._get_slices_from_volumes(lr_memory_map, hr_memory_map)

            lr_tensor = self._convert_to_3d_tensor(lr_slice)
            hr_tensor = self._convert_to_3d_tensor(hr_slice)
            return (lr_tensor, hr_tensor)
        else:
            # Select cube 
            lr_shape = lr_memory_map.shape
            safe_side_length = int(min(list(lr_shape) + [self.cube_side_length]))
            x = np.random.randint(0, lr_shape[0] - safe_side_length + 1)
            y = np.random.randint(0, lr_shape[1] - safe_side_length + 1)
            z = np.random.randint(0, lr_shape[2] - safe_side_length + 1)
            # Fetch required information from the lr memory map 
            lr_volume = lr_memory_map[x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]
            # Calculate the hr_coordinates
            x, y, z, safe_side_length = 2*x, 2*y, 2*z, 2*safe_side_length
            # Fetch required information from the hr memory map 
            hr_volume = hr_memory_map[x:x+safe_side_length, y:y+safe_side_length, z:z+safe_side_length]
            # Convert to tensor
            lr_image, hr_image = self._get_slices_from_volumes(lr_volume, hr_volume)
            lr_tensor = self._convert_to_4d_tensor(lr_image)
            hr_tensor = self._convert_to_4d_tensor(hr_image)
            return (lr_tensor, hr_tensor)



      