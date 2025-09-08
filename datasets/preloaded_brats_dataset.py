
from torch.utils.data import Dataset

DEFAULT_LR_PATH = ''
DEFAULT_HR_PATH = ''

class PreloadedBratsDataset(Dataset): 
    def __init__(self, lr_path=DEFAULT_LR_PATH, hr_path=DEFAULT_HR_PATH):
        super().__init__()
        self.lr_path = lr_path
        self.hr_path = hr_path


