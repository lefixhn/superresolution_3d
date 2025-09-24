'''
    Compares a 2D Model with a 3D Model 
    The 2D Model builds a 3D volumina though slice wise superresolution
    along two axes and interpolation along one axe


''' 
import torch
from tqdm import tqdm
import comparing_metrics as cm 
from typing import Dict
from build_sclie_wise_hr_volume import build_sclie_wise_hr_volume

@torch.no_grad()
def compare_pseudo_3d_with_3d(
    model_2d: torch.nn.Module, 
    model_3d: torch.nn.Module, 
    lr_hr_5d_tensor_tuples,
    autoprint=True, 
    device="cuda" if torch.cuda.is_available() else "cpu", 
) -> Dict[str, Dict[str, float]]:
    
    results = {}
    
    mean_mse_2d = 0.0
    mean_mae_2d = 0.0
    mean_psnr_2d = 0.0

    mean_mse_3d = 0.0
    mean_mae_3d = 0.0
    mean_psnr_3d = 0.0
    

    model_3d = model_3d.to(device)

    for lr_volume_tensor_5d, hr_volume_tensor_5d in tqdm(lr_hr_5d_tensor_tuples, "Iterating trhough lr-hr-tuples"):
        # Build SR Images
        sr_image_2d_model = build_sclie_wise_hr_volume(
            lr_volume=lr_volume_tensor_5d, 
            model: model_2d, 
            interpolation_dim="D",
            device=device, 
        )

        sr_image_3d_model = model_3d(lr_volume_tensor_5d)

        # Apply metrics 
        mean_mse_2d += cm.compare_mse(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_mse_2d += cm.compare_mae(sr_image_2d_model ,hr_volume_tensor_5d)
        mean_mse_2d += cm.compare_psnr(sr_image_2d_model ,hr_volume_tensor_5d)

        mean_mse_3d += cm.compare_mse(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_mse_3d += cm.compare_mae(sr_image_3d_model ,hr_volume_tensor_5d)
        mean_mse_3d += cm.compare_psnr(sr_image_3d_model ,hr_volume_tensor_5d)

    num_tuples = len(lr_hr_5d_tensor_tuples)
    
    mean_mse_2d /= num_tuples
    mean_mae_2d /= num_tuples
    mean_psnr_2d /= num_tuples

    mean_mse_3d = /= num_tuples
    mean_mae_3d = /= num_tuples
    mean_psnr_3d = /= num_tuples

    name_model_2d = type(model_2d).__name__
    name_model_3d = type(model_3d).__name__


    results[name_model_2d]["mse"] = mean_mse_2d
    results[name_model_2d]["mae"] = mean_mse_2d
    results[name_model_2d]["psnr"] = mean_mse_2d

    results[name_model_3d]["mse"] = mean_mse_3d
    results[name_model_3d]["mae"] = mean_mse_3d
    results[name_model_3d]["psnr"] = mean_mse_3d

    if autoprint: 
        _print_compare_results(results)

    return results


def _print_compare_results(results: Dict[str, Dict[str, float]]): 
    for model_name, metric_results in results.items(): 
        model_text = str(model_name)
        for metric_name, average_metric_value in metric_results.items(): 
            model_text += f" | Average {metric_name}: {average_metric_value}"

