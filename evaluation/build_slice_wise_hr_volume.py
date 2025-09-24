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
    interpolation_axe_dim=0     # Can be 0, 1 or 2
    upscale_factor=2, 
    device = "cuda" if torch.cuda.is_available() else "cpu", 
): 
    if lr_volume.ndims() == 4: 
        lr_volume = lr_volume.unsqueeze(0)
    elif lr_volume.ndims() == 3: 
        lr_volume = lr_volume.unsqueeze(0).unsqueeze(0)
    elif not lr_volume.ndims() == 5: 
        raise TypeError("Input must have shape (B, C, D, H, W), (C, D, H, W) or (D, H, W)")

    assert lr_volume.shape[1] == 1, "Image must be greyscale"

    B, C, D, H, W = lr_volume.shape

    for batch_index in range(B): 
        lr_vomume_4d = 

    