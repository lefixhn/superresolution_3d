'''
    Compares a 2D Model with a 3D Model 
    The 2D Model builds a 3D volumina though slice wise superresolution
    along two axes and interpolation along one axe


''' 
import torch


def compare_pseudo_3d_with_3d(
    model_2d: torch.nn.Module, 
    model_3d: torch.nn.Module, 
    lr_hr_5d_tensor_tuples,
):