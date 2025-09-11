import torch 
import torch.nn.functional as F  
import numpy as np 


def _convert_to_5d_tensor(image): 
    '''
    Accepts nparray, or 3D 4D or 5d Tensor 
    '''
    # Convert to tensor if it is not already a tensor 
    if isinstance(image, np.ndarray): 
        image = torch.from_numpy(image)
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
    sr_iamge, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image)
    max_element = torch.max(hr_image)
    return 20 * np.log10(max_element) - 10 * np.log10(compare_mse(sr_image, hr_image ))


def compare_ssim(sr_image, hr_image):
    sr_iamge, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image) 
    


def compare_lpips(sr_image, hr_image):
    sr_iamge, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image) 
    return None 