'''
    Compares a 2D Model with a 3D Model 
    The 2D Model builds a 3D volumina though slice wise superresolution
    along two axes and interpolation along one axe


''' 
import torch
from tqdm import tqdm
import comparing_metrics as cm 
from typing import Dict


def compare_pseudo_3d_with_3d(
    model_2d: torch.nn.Module, 
    model_3d: torch.nn.Module, 
    lr_hr_5d_tensor_tuples,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for lr_volume_tensor_5d, hr_volume_tensor_5d in tqdm(lr_hr_5d_tensor_tuples, "Iterating trhough lr-hr-tuples"):
         
