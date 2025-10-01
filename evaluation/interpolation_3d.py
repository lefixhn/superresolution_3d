import torch 
from  scipy.ndimage import zoom
import numpy as np 

def interpolation_3d(
    tensor_5d : torch.Tensor, 
    order=3, 
    upscale_factor=2, 
): 
    if tensor_5d.ndim == 3:
        tensor_5d = tensor_5d.unsqueeze(0).unsqueeze(0)   # (D,H,W) -> (1,1,D,H,W)
    elif tensor_5d.ndim == 4:
        tensor_5d = tensor_5d.unsqueeze(0)                # (C,D,H,W) -> (1,C,D,H,W)
    assert tensor_5d.ndim == 5, f"Expected 5D, got {tuple(tensor_5d.shape)}"

    # Store metadata
    in_dev   = tensor_5d.device
    in_dtype = tensor_5d.dtype

    B, C, D, H, W = tensor_5d.shape
    b_list = []
    for b in range(B):
        c_list = []
        for c in range(C):
            vol = tensor_5d[b, c].detach().cpu().numpy()   # (D,H,W)
            zoom_factors = (upscale_factor, upscale_factor, upscale_factor)
            hr_vol = zoom(vol, zoom=zoom_factors, order=order)  # order=3 = cubic
            hr_vol = torch.from_numpy(hr_vol).unsqueeze(0).unsqueeze(0)  # (1,1,D',H',W')
            c_list.append(hr_vol)
        b_list.append(torch.cat(c_list, dim=1))
    
    out = torch.cat(b_list, dim=0)  # (B,C,D',H',W')
    out = out.to(device=in_dev, dtype=in_dtype).contiguous()

    return out.clamp(0, 1)
   
    return out.to(tensor_5d.device).float().clamp(0, 1)