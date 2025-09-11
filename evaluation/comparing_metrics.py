import torch 
import torch.nn.functional as F  


def compare_mse(sr_image, hr_image): 
    return F.mse_loss(sr_image, hr_image).item()


def compare_mae(sr_image, hr_image): 
    return None


def compare_psnr(sr_image, hr_image): 
    return None 


def compare_ssim(sr_image, hr_image): 
    return None


def compare_lpips(sr_image, hr_image): 
    return None 