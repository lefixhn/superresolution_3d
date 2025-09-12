import torch 
import torch.nn.functional as F  
import numpy as np 
from monai.metrics import SSIMMetric

def _convert_to_5d_tensor(image): 
    '''
    Accepts nparray, or 3D 4D or 5d Tensor 
    '''
    # Convert to tensor if it is not already a tensor 
    if isinstance(image, np.ndarray): 
        image = torch.from_numpy(image)
    else: 
        image = image.detach()  # To avoid gradient calculation if it is already a tensor
    # Ensure it is 5d

    if len(image.shape) == 3: 
        image = image.unsqueeze(0).unsqueeze(0)
    if len(image.shape) == 4: 
        image = image.unsqueeze(1)
    return image


def compare_mse(sr_image, hr_image): 
    sr_iamge, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image)
    return F.mse_loss(sr_image, hr_image).item()


def compare_mae(sr_image, hr_image): 
    sr_iamge, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image)
    return F.l1_loss(sr_image, hr_image).item()


def compare_psnr(sr_image, hr_image): 
    '''
    Calcualtes the psnr. Uses the max element of the hr_image so 
    pay attention to the order of the images. 
    '''
    sr_image, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image)
    max_element = torch.max(hr_image).item()
    return 20 * np.log10(max_element) - 10 * np.log10(compare_mse(sr_image, hr_image ))

# TODO: Check parametersettings and compare with other implementation
def compare_ssim(sr_image, hr_image):
    sr_iamge, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image) 
    ssim_metric = SSIMMetric(
        spatial_dims = 3, 
        data_range = 1, 
        kernel_type = "gaussian", 
        win_size = 11, 
        kernel_sigma = 1.5, 
        k1 = 0.01, 
        k2 = 0.03
    )
    ssim_results = ssim_metric(sr_image, hr_image)
    ssim_mean_value = ssim_results.mean().item()
    return ssim_mean_value


def compare_lpips(sr_image, hr_image):
    sr_iamge, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image) 
    return None 