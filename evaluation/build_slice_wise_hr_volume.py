'''
    Uses a 2D SR Model to generate a 3D HR Volume
    Two axes are upscaled though the 2D Model. 
    The remaining axe is upscaled though interpolation
'''
import torch 

def build_sclie_wise_hr_volume(
    lr_volume: torch.Tensor, 
    model: callable, 
    interpolation_order=1, 
    upscale_factor=2, 
    device = "cuda" if torch.cuda.is_available() else "cpu", 
): 
    if lr_volume.ndims() == 4: 
        lr_volume = lr_volume.unsqueeze(0)
    elif lr_volume.ndims() == 3: 
        lr_volume = lr_volume.unsqueeze(0).unsqueeze(0)
    elif not lr_volume.ndims() == 5: 
        raise TypeError("Input must have shape (B, C, D, H, W), ")

    