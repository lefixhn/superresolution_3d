import numpy as np 
from torch.utils.data import Dataset

DEFAULT_LR_PATH = ''
DEFAULT_HR_PATH = ''

class PreloadedBratsDataset(Dataset): 
    def __init__(self, lr_path=DEFAULT_LR_PATH, hr_path=DEFAULT_HR_PATH):
        super().__init__()
        self.lr_path = lr_path
        self.hr_path = hr_path

        self.preloaded_data = [

        ]

    
    def _get_file_as_tensor(self, path):
        # Load and convert to float32
        data = np.load(path)
        data = data.astype(np.float32, copy=False)
        


        


