import torch 
import torch.nn.functional as F  
import numpy as np 

def _convert_to_5d_tensor(image): 
    '''
    Accepts nparray, or 3D 4D or 5d Tensor 
    '''
    if isinstance(image, np.)


def compare_mse(sr_image, hr_image): 
    return F.mse_loss(sr_image, hr_image).item()


def compare_mae(sr_image, hr_image): 
    return F.l1_loss(sr_image, hr_image).item()


def compare_psnr(sr_image, hr_image): 
    return 


def compare_ssim(sr_image, hr_image): 
    return None


def compare_lpips(sr_image, hr_image): 
    return None 