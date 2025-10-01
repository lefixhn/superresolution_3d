import torch 
import ndimage

def interpolation_3d(
    tensor_5d : torch.Tensor, 
    order=3, 
    upscale_factor=2, 
)