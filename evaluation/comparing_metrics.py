import torch 
import torch.nn.functional as F  
import numpy as np 
from piqa import LPIPS

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
    sr_image, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image)
    return F.mse_loss(sr_image, hr_image).item()


def compare_mae(sr_image, hr_image): 
    sr_image, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image)
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
import torch
import torch.nn.functional as F

# Helper method for compare_ssim
def _gauss3d(ks=11, sigma=1.5, device=None, dtype=torch.float32):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1)/2
    g = torch.exp(-(ax**2) / (2*sigma**2))
    g = g / g.sum()
    g3 = (g[:,None,None] * g[None,:,None] * g[None,None,:]).unsqueeze(0).unsqueeze(0)  # (1,1,K,K,K)
    return g3

# compare_ssim was written through AI, because there was no good implementation that worked well
def compare_ssim(sr, hr, data_range=None, ks=11, sigma=1.5, K1=0.01, K2=0.03, eps=1e-12):
    # auf (B,1,D,H,W)
    if sr.ndim == 3: sr = sr.unsqueeze(0).unsqueeze(0)
    elif sr.ndim == 4: sr = sr.unsqueeze(1)
    if hr.ndim == 3: hr = hr.unsqueeze(0).unsqueeze(0)
    elif hr.ndim == 4: hr = hr.unsqueeze(1)

    sr = sr.detach().to(torch.float32)
    hr = hr.detach().to(device=sr.device, dtype=sr.dtype)

    # Datenbereich
    if data_range is None:
        if hr.amin().item() >= 0 and hr.amax().item() <= 1:
            L = 1.0
        else:
            L = float((hr.amax() - hr.amin()).clamp_min(1e-12).item())
    else:
        L = float(data_range)

    kernel = _gauss3d(ks, sigma, device=sr.device, dtype=sr.dtype)
    pad = ks // 2
    # depthwise 3D conv: pro Kanal (C=1) okay, für generisch -> repeat auf C und groups=C
    def conv(z): return F.conv3d(z, kernel, padding=pad)

    with torch.inference_mode():
        mu_x = conv(sr)
        mu_y = conv(hr)
        mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
        sigma_x2 = conv(sr*sr) - mu_x2
        sigma_y2 = conv(hr*hr) - mu_y2
        sigma_xy = conv(sr*hr) - mu_xy

        C1, C2 = (K1*L)**2, (K2*L)**2
        num = (2*mu_xy + C1) * (2*sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps
        ssim_map = num / den
        return float(ssim_map.mean().item())

def compare_lpips(sr_image, hr_image, lpips_2d_metric=LPIPS(network='vgg').eval().to('cuda' if torch.cuda.is_available() else 'cpu')):
    # Convert to 5D Tensors
    sr_image, hr_image = _convert_to_5d_tensor(sr_image), _convert_to_5d_tensor(hr_image) 
    # Noramlize both images from [0, 1] range to [-1, 1] range 
    sr_image, hr_image = 2 * sr_image - 1, 2 * hr_image - 1
    
    tensor_shape = hr_image.shape
    lpips_mean = 0
    # Look at D H W from the B C D H W of the tensor 
    for batch_index in range(tensor_shape[0]): 

        # Iterate through orientations coronar, axial and sagital
        for shape_dim_index in range(2, 5): 
            lpips_sum_over_slices = 0
            dimension_length = tensor_shape[shape_dim_index]
            for slice_index in range(dimension_length): 
                # Create a tuple containing the correct coordinates / slices
                # We put the slice index to the correct position inside the tuple
                # therefore we have to check the_shape_dim index, wich tells us
                # weather we are in a coronar, axial or sagital sclice
                slice_coordinates = (batch_index, 0) + tuple(slice_index if i == shape_dim_index-2 else slice(None) for i in range(3))
                sr_slice = sr_image[slice_coordinates]
                hr_slice = hr_image[slice_coordinates]
                # Add batch and channel dimension 
                sr_slice = sr_slice.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                hr_slice = hr_slice.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                lpips_sum_over_slices += lpips_2d_metric(sr_slice, hr_slice)
            lpips_mean += lpips_sum_over_slices / dimension_length
    # Divide by the amount of orientations and the number of batches
    lpips_mean /= 3 * tensor_shape[0]
    return lpips_mean

# Check weather it works properly
if __name__ == "__main__": 
    mock_hr = torch.rand(4, 1, 100, 90, 120)
    noise_tensor = torch.from_numpy(np.random.normal(0, 0.01, size=(4, 1, 100, 90, 120)))
    mock_sr = mock_hr + noise_tensor
    compare_results = {
        "MSE" : compare_mse(mock_sr, mock_hr), 
        "MAE" : compare_mae(mock_sr, mock_hr), 
        "PSNR" : compare_psnr(mock_sr, mock_hr), 
        "SSIM" : compare_ssim(mock_sr, mock_hr),
        "LPIPS" : compare_lpips(mock_sr, mock_hr) 
    }
    print("RESULTS FOR DUMMY DATA")
    for metric in compare_results: 
        print(f"{metric}: {compare_results[metric]}")


    


